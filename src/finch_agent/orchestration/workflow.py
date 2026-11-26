from __future__ import annotations

import re
import threading
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from finch_agent.agents.rag_agent import RAGAgent
from finch_agent.agents.sql_agent import SQLAgent
from finch_agent.agents.supervisor import SupervisorAgent
from finch_agent.core.llm import GroqLanguageModel, LanguageModel, MockLanguageModel, OpenAILanguageModel
from finch_agent.core.guardrails import InputRejected, validate_final_response, validate_user_input
from finch_agent.memory.checkpointer import Checkpointer, trim_conversation_history
from finch_agent.memory.long_term_store import LongTermStore
from finch_agent.orchestration.state import AgentState, AskRequest, AskResponse, SourceCitation
from finch_agent.tools.sql_engine import ReadOnlyQueryEngine
from finch_agent.tools.vector_store import VectorStore
from finch_agent.config.settings import Settings


class FinchService:
    """Builds and runs the bounded three-agent supervisor workflow."""

    def __init__(self, settings: Settings, *, llm: LanguageModel | None = None, vectors: Any | None = None) -> None:
        self.settings = settings
        if llm is not None:
            self.llm = llm
        elif settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            self.llm = OpenAILanguageModel(settings.openai_api_key.get_secret_value(), settings.openai_model)
        elif settings.llm_provider == "groq":
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
            self.llm = GroqLanguageModel(settings.groq_api_key.get_secret_value(), settings.groq_model, settings.groq_base_url)
        else:
            self.llm = MockLanguageModel()
        self.vectors = vectors if vectors is not None else VectorStore(settings)
        self.database = ReadOnlyQueryEngine(settings)
        self.checkpointer = Checkpointer(settings)
        self.long_term_store = LongTermStore(settings.checkpoint_path)
        self._local_graph_lock = threading.RLock()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node(
            "supervisor",
            SupervisorAgent(self.llm, catalog_path=self.settings.metadata_catalog_path),
        )
        builder.add_node(
            "sql_agent",
            SQLAgent(self.llm, self.vectors, self.database, catalog_path=self.settings.metadata_catalog_path),
        )
        builder.add_node("rag_agent", RAGAgent(self.vectors))
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges("supervisor", lambda state: state.get("next_agent", "FINISH"),
                                      {"sql_agent": "sql_agent", "rag_agent": "rag_agent", "FINISH": END})
        builder.add_edge("sql_agent", "supervisor")
        builder.add_edge("rag_agent", "supervisor")
        return builder.compile(checkpointer=self.checkpointer.saver)

    def ask(self, request: AskRequest) -> AskResponse:
        """Run one guarded workflow turn for the supplied user request."""
        request_id = str(uuid.uuid4())
        thread_key = request.thread_ts or request_id
        thread_id = f"{request.channel_id or 'api'}:{thread_key}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
            "tags": ["supervisor-workflow"],
            "metadata": {"thread_id": thread_id, "step_count": 0, "max_iterations": 6,
                         "workflow": "supervisor-rag-sql"},
        }
        with self._local_graph_lock:
            prior = self.graph.get_state(config).values
            messages = trim_conversation_history(prior.get("messages", []), max_turns=6)
            contextual_input = "\n".join(item["content"] for item in messages[-6:]) + "\n" + request.question
            try:
                validate_user_input(contextual_input)
            except InputRejected as exc:
                return AskResponse(route="refusal", answer=exc.user_message, request_id=request_id)
            user_context = self.long_term_store.get_user_context(request.user_id)
        if self._is_table_count_question(request.question):
            tables = [item["table_name"] for item in self.database.get_schema_context(sample_rows=0)]
            return AskResponse(
                route="sql_agent",
                answer=f"The database contains {len(tables)} tables: {', '.join(tables)}.",
                columns=["table_name"],
                rows=[[name] for name in tables],
                request_id=request_id,
                trajectory=["sql_agent"],
            )
        initial: AgentState = {
            "question": request.question.strip(), "request_id": request_id, "user_id": request.user_id,
            "channel_id": request.channel_id, "route": "clarification", "next_agent": "supervisor",
            "sub_task_instructions": {}, "current_task": request.question.strip(), "last_agent": "",
            "effective_question": "", "metadata": [], "generated_sql": None,
            "sql_results": [], "rag_context": [], "columns": [], "rows": [], "truncated": False,
            "sql_valid": False, "sql_unsupported": False, "sql_error": "", "fatal_error": False,
            "iteration_count": 0, "step_count": 0, "max_iterations": 6, "worker_completed": False,
            "citations": [], "trajectory": [], "execution_trace": [], "sql_executed": False,
            "sql_attempted": False, "sql_guardrail_violations": 0,
            "user_context": user_context, "answer": "",
        }
        with self._local_graph_lock:
            result = self.graph.invoke(initial, config=config)
        citations = [SourceCitation.model_validate(item) for item in result.get("citations", [])]
        route = result.get("route", "clarification")
        answer = validate_final_response(
            result.get("answer", ""), question=request.question, sql_results=result.get("sql_results", []),
            rag_context=result.get("rag_context", []), factual=route in {"sql_agent", "rag_agent", "doc_rag_agent"},
        )
        return AskResponse(route="doc_rag_agent" if route == "rag_agent" else route,
                           answer=answer,
                           sql=result.get("generated_sql") if route == "sql_agent" else None,
                           columns=result.get("columns", []), rows=result.get("rows", []),
                           retry_count=result.get("iteration_count", 0), citations=citations, request_id=request_id,
                           trajectory=result.get("trajectory", []),
                           execution_trace=result.get("execution_trace", []),
                           sql_executed=result.get("sql_executed", False),
                           sql_guardrail_violations=result.get("sql_guardrail_violations", 0))

    def close(self) -> None:
        self.database.engine.dispose()
        self.checkpointer.close()

    @staticmethod
    def _is_table_count_question(question: str) -> bool:
        """Recognize database table-count questions that can be answered from live schema."""
        asks_for_count = re.search(r"\b(how many|number of|count of)\b", question, re.I)
        mentions_tables = re.search(r"\btables?\b", question, re.I)
        return bool(asks_for_count and mentions_tables)
