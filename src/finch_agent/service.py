from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from typing import Any, TypedDict

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import GroqLanguageModel, LanguageModel, MockLanguageModel
from .models import AskRequest, AskResponse, SourceCitation
from .sql_engine import ReadOnlyQueryEngine
from .vector_store import VectorStore

log = structlog.get_logger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    request_id: str
    user_id: str
    channel_id: str | None
    route: str
    history: list[dict[str, str]]
    effective_question: str
    metadata: list[dict[str, Any]]
    generated_sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    sql_unsupported: bool
    sql_valid: bool
    sql_error: str
    fatal_error: bool
    iteration_count: int
    citations: list[dict[str, Any]]
    answer: str


class FinchService:
    """LangGraph supervisor and worker pipeline with persistent Slack thread state."""

    def __init__(self, settings: Settings, *, llm: LanguageModel | None = None, vectors: Any | None = None) -> None:
        self.settings = settings
        os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
        if settings.langsmith_tracing:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
            if settings.langsmith_api_key:
                os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        if llm is not None:
            self.llm = llm
        elif settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            from .llm import OpenAILanguageModel

            self.llm: LanguageModel = OpenAILanguageModel(
                settings.openai_api_key.get_secret_value(), settings.openai_model
            )
        elif settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
            self.llm = GroqLanguageModel(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model,
                base_url=settings.groq_base_url,
            )
        else:
            self.llm = MockLanguageModel()
        self.vectors = vectors if vectors is not None else VectorStore(settings)
        self.database = ReadOnlyQueryEngine(settings)
        self._checkpoint_connection: sqlite3.Connection | None = None
        self._checkpoint_pool: Any | None = None
        if settings.checkpoint_database_url:
            from psycopg_pool import ConnectionPool
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres import PostgresSaver

            self._checkpoint_pool = ConnectionPool(
                conninfo=settings.checkpoint_database_url.get_secret_value(), min_size=1, max_size=50,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}, open=True,
            )
            self.checkpointer = PostgresSaver(self._checkpoint_pool)
        else:
            settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_connection = sqlite3.connect(
                settings.checkpoint_path, check_same_thread=False, timeout=30
            )
            self._checkpoint_connection.execute("PRAGMA journal_mode=WAL")
            self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.checkpointer.setup()
        self.graph = self._build_graph()
        self._local_graph_lock = threading.RLock()

    def close(self) -> None:
        self.database.engine.dispose()
        if self._checkpoint_pool:
            self._checkpoint_pool.close()
        if self._checkpoint_connection:
            self._checkpoint_connection.close()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("sql_writer", self._sql_writer)
        builder.add_node("sql_executor", self._sql_executor)
        builder.add_node("sql_result", self._sql_result)
        builder.add_node("sql_failure", self._sql_failure)
        builder.add_node("doc_retriever", self._doc_retriever)
        builder.add_node("doc_answer", self._doc_answer)
        builder.add_node("clarification", self._clarification)
        builder.add_node("save_history", self._save_history)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges("supervisor", lambda state: state["route"], {
            "sql_agent": "sql_writer", "doc_rag_agent": "doc_retriever", "clarification": "clarification",
        })
        builder.add_edge("sql_writer", "sql_executor")
        builder.add_conditional_edges("sql_executor", self._after_sql, {
            "success": "sql_result", "retry": "sql_writer", "failed": "sql_failure",
        })
        builder.add_edge("sql_result", "save_history")
        builder.add_edge("sql_failure", "save_history")
        builder.add_edge("doc_retriever", "doc_answer")
        builder.add_edge("doc_answer", "save_history")
        builder.add_edge("clarification", "save_history")
        builder.add_edge("save_history", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _context_question(state: AgentState) -> str:
        history = state.get("history", [])[-6:]
        previous = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        current = state["question"]
        return f"Conversation so far:\n{previous}\n\nCurrent question:\n{current}" if previous else current

    def _supervisor(self, state: AgentState) -> dict[str, Any]:
        effective = self._context_question(state)
        decision = self.llm.route(effective)
        return {"route": decision.destination, "effective_question": effective}

    def _sql_writer(self, state: AgentState) -> dict[str, Any]:
        metadata = self.vectors.search_metadata(state["effective_question"])
        metadata.insert(0, {"database_dialect": self.database.engine.dialect.name})
        try:
            sql = self.llm.generate_sql(state["effective_question"], metadata, state.get("sql_error"))
            return {"generated_sql": sql, "metadata": metadata, "sql_error": ""}
        except Exception as exc:
            return {"generated_sql": "", "metadata": metadata, "sql_error": type(exc).__name__, "fatal_error": True}

    def _sql_executor(self, state: AgentState) -> dict[str, Any]:
        if state.get("fatal_error"):
            return {"sql_valid": False}
        generated_sql = (state.get("generated_sql") or "").strip().rstrip(";").lower()
        if generated_sql == "select 1 where 1=0":
            return {"route": "clarification", "sql_valid": True, "sql_unsupported": True,
                    "columns": [], "rows": [], "truncated": False}
        try:
            columns, rows, truncated = self.database.execute(state.get("generated_sql") or "")
            return {"sql_valid": True, "columns": columns,
                    "rows": [[self._json_value(value) for value in row] for row in rows],
                    "truncated": truncated, "sql_error": ""}
        except Exception as exc:
            attempt = state.get("iteration_count", 0) + 1
            error = str(exc)[:500]
            log.warning("sql_attempt_failed", request_id=state["request_id"], attempt=attempt,
                        error_type=type(exc).__name__)
            return {"sql_valid": False, "sql_error": error, "iteration_count": attempt}

    @staticmethod
    def _after_sql(state: AgentState) -> str:
        if state.get("fatal_error"):
            return "failed"
        if state.get("sql_valid"):
            return "success"
        return "retry" if state.get("iteration_count", 0) <= 3 else "failed"

    def _sql_result(self, state: AgentState) -> dict[str, str]:
        if state.get("sql_unsupported"):
            return {"answer": "I couldn't match this question to an approved database schema. Please clarify the metric or request access to the relevant source."}
        rows = state.get("rows", [])
        count = len(rows)
        suffix = " (results truncated at the configured row limit)" if state.get("truncated") else ""
        if not rows:
            summary = "The query completed successfully and returned no matching rows."
        elif len(rows) == 1:
            summary = "Result: " + "; ".join(
                f"{column} = {value}" for column, value in zip(state.get("columns", []), rows[0])
            ) + "."
        else:
            summary = f"Query returned {count} rows across {len(state.get('columns', []))} columns. See the table below."
        return {"answer": f"{summary}{suffix}"}

    def _sql_failure(self, state: AgentState) -> dict[str, str]:
        if state.get("fatal_error"):
            return {"answer": "The SQL generation service is temporarily unavailable. Please try again later."}
        return {"answer": "The read-only query could not be completed after three corrections. Please refine the question or contact the data team."}

    def _doc_retriever(self, state: AgentState) -> dict[str, Any]:
        hits = self.vectors.search_reports(state["effective_question"])
        return {"citations": [citation.model_dump() for citation in hits]}

    def _doc_answer(self, state: AgentState) -> dict[str, str]:
        answer = self.llm.synthesize_report(state["effective_question"], state.get("citations", []))
        return {"answer": answer}

    @staticmethod
    def _clarification(state: AgentState) -> dict[str, str]:
        return {"answer": "Could you clarify whether you need a database result or an answer from a report or policy?"}

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _save_history(state: AgentState) -> dict[str, list[dict[str, str]]]:
        history = list(state.get("history", []))
        history.extend([
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state.get("answer", "")},
        ])
        return {"history": history[-12:]}

    def ask(self, request: AskRequest) -> AskResponse:
        request_id = str(uuid.uuid4())
        thread_identity = request.thread_ts or request_id
        thread_id = f"{request.channel_id or 'api'}:{thread_identity}"
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "question": request.question.strip(), "request_id": request_id, "user_id": request.user_id,
            "channel_id": request.channel_id, "route": "clarification", "effective_question": "",
            "metadata": [], "generated_sql": None, "columns": [], "rows": [], "truncated": False,
            "sql_valid": False, "sql_unsupported": False, "sql_error": "", "fatal_error": False, "iteration_count": 0,
            "citations": [], "answer": "",
        }
        if self._checkpoint_pool is None:
            # The local SQLite saver shares one DB-API connection; serialize local graph runs.
            with self._local_graph_lock:
                result = self.graph.invoke(initial, config=config)
        else:
            result = self.graph.invoke(initial, config=config)
        citations = [SourceCitation.model_validate(item) for item in result.get("citations", [])]
        return AskResponse(route=result["route"], answer=result.get("answer", ""),
                           sql=result.get("generated_sql") if result["route"] == "sql_agent" else None,
                           columns=result.get("columns", []), rows=result.get("rows", []),
                           retry_count=result.get("iteration_count", 0),
                           citations=citations, request_id=request_id)
