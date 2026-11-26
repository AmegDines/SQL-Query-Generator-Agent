from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog

from finch_agent.core.guardrails import GuardrailViolation, validate_sql_query
from finch_agent.orchestration.state import AgentState

log = structlog.get_logger(__name__)


def extract_sql_response(response: Any) -> str:
    """Normalize common LLM SQL response formats to one bare SQL statement.

    Only a JSON object with a recognized SQL field or a complete Markdown code
    fence is unwrapped; surrounding prose is deliberately left for validation
    to reject rather than guessed at.
    """
    if not isinstance(response, str):
        raise ValueError("SQL model response must be a string")
    value = response.strip()
    if value.startswith("{"):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = next((payload[key] for key in ("sql", "query", "statement")
                              if isinstance(payload.get(key), str)), None)
            if candidate is None:
                raise ValueError("SQL JSON response must contain a string 'sql' field")
            value = candidate.strip()
    fenced = re.fullmatch(r"```(?:sql)?\s*\n?(.*?)\n?```", value, re.IGNORECASE | re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    # Remove only a final terminator. Internal semicolons remain visible to
    # validate_sql_query so stacked statements are still blocked.
    return value.strip().rstrip(";").strip()


class SQLAgent:
    """Generates, validates, executes, and repairs read-only SQL queries."""

    def __init__(
        self,
        llm: Any,
        vectors: Any,
        database: Any,
        max_retries: int = 2,
        catalog_path: Path | None = None,
    ) -> None:
        self.llm, self.vectors, self.database, self.max_retries = llm, vectors, database, max_retries
        self.catalog_path = catalog_path

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Generate, validate, execute, and repair a grounded read-only query."""
        question = state.get("current_task") or state.get("effective_question", state["question"])
        metadata = self.vectors.search_metadata(question)
        metadata.insert(0, {"database_dialect": self.database.engine.dialect.name,
                            "live_schema": self.database.get_schema_context(),
                            "catalog_schema": self._catalog_schema(),
                            "rag_context": state.get("rag_context", [])})
        error = None
        guardrail_retries = 0
        for attempt in range(min(self.max_retries, 2) + 1):
            try:
                generated = self.llm.generate_sql(question, metadata, error)
                sql = validate_sql_query(extract_sql_response(generated))
                if (sql or "").strip().rstrip(";").lower() == "select 1 where 1=0":
                    return {"generated_sql": sql, "metadata": metadata, "sql_unsupported": True,
                            "answer": "I couldn't match this question to an approved database schema. Please clarify the metric or request access to the relevant source.",
                            "iteration_count": attempt, "worker_completed": True, "last_agent": "sql_agent",
                            "sql_attempted": True,
                            "execution_trace": state.get("execution_trace", []) + ["sql_agent -> supervisor"]}
                tool_result = self.database.execute(sql)
                if tool_result.status == "error":
                    error = tool_result.error_msg or "Database query failed"
                    log.warning("sql_tool_error", request_id=state.get("request_id"),
                                tables=tool_result.tables_used, error=error)
                    if tool_result.guardrail_violation:
                        guardrail_retries += 1
                        if guardrail_retries > 1:
                            break
                    continue
                sql = tool_result.query
                sql_results = tool_result.rows
                columns = tool_result.columns
                rows = [[self._json_value(value) for value in row.values()] for row in sql_results]
                suffix = " (results truncated at the configured row limit)" if tool_result.truncated else ""
                if not rows:
                    summary = "The query completed successfully and returned no matching rows."
                elif len(rows) == 1:
                    summary = "Result: " + "; ".join(f"{col} = {val}" for col, val in zip(columns, rows[0])) + "."
                else:
                    summary = f"Query returned {len(rows)} rows across {len(columns)} columns. See the table below."
                return {"generated_sql": sql, "metadata": metadata, "columns": columns, "rows": rows,
                        "truncated": tool_result.truncated, "sql_results": sql_results,
                        "answer": summary + suffix, "iteration_count": attempt, "worker_completed": True,
                        "last_agent": "sql_agent", "sql_executed": True,
                        "sql_attempted": True,
                        "execution_trace": state.get("execution_trace", []) + ["sql_agent -> supervisor"],
                        "sql_guardrail_violations": state.get("sql_guardrail_violations", 0) + guardrail_retries}
            except Exception as exc:
                error = str(exc)[:500]
                log.warning("sql_attempt_failed", request_id=state.get("request_id"), attempt=attempt + 1,
                            error_type=type(exc).__name__)
                if isinstance(exc, GuardrailViolation):
                    guardrail_retries += 1
                    if guardrail_retries > 1:
                        break
                    error = f"Guardrail rejected the query: {error}. Regenerate one safe SELECT query."
        return {"metadata": metadata, "sql_error": error or "SQL generation failed", "fatal_error": True,
                "answer": "The read-only query could not be completed after three corrections. Please refine the question or contact the data team.",
                "iteration_count": min(self.max_retries, 2), "worker_completed": True, "last_agent": "sql_agent",
                "sql_attempted": True,
                "execution_trace": state.get("execution_trace", []) + ["sql_agent -> supervisor"],
                "sql_guardrail_violations": state.get("sql_guardrail_violations", 0) + guardrail_retries}

    def _catalog_schema(self) -> list[dict[str, Any]]:
        """Load the static table catalog used to ground business terms and columns."""
        path = self.catalog_path or Path("data/raw/catalog.json")
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [item for item in catalog if item.get("table_name")]

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
