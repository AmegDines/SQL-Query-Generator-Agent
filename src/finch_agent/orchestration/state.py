from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    user_id: str = "api-user"
    channel_id: str | None = None
    thread_ts: str | None = None


class RouteDecision(BaseModel):
    destination: str
    reasoning: str = ""


class SourceCitation(BaseModel):
    document: str
    page: int
    excerpt: str
    chunk_id: str = ""
    section: str = ""
    score: float = 0.0


class ToolResult(BaseModel):
    status: Literal["success", "error"]
    query: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    truncated: bool = False
    tables_used: list[str] = Field(default_factory=list)
    error_msg: str | None = None
    guardrail_violation: bool = False


class AskResponse(BaseModel):
    route: str
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    retry_count: int = 0
    citations: list[SourceCitation] = Field(default_factory=list)
    request_id: str
    trajectory: list[str] = Field(default_factory=list)
    execution_trace: list[str] = Field(default_factory=list)
    sql_executed: bool = False
    sql_guardrail_violations: int = 0


class AgentState(TypedDict, total=False):
    question: str
    request_id: str
    user_id: str
    channel_id: str | None
    route: str
    next_agent: str
    last_agent: str
    current_task: str
    sub_task_instructions: dict[str, str]
    history: list[dict[str, str]]
    effective_question: str
    metadata: list[dict[str, Any]]
    generated_sql: str | None
    sql_results: list[dict[str, Any]]
    rag_context: list[str]
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    sql_unsupported: bool
    sql_valid: bool
    sql_error: str
    fatal_error: bool
    iteration_count: int
    step_count: int
    max_iterations: int
    worker_completed: bool
    citations: list[dict[str, Any]]
    trajectory: list[str]
    sql_executed: bool
    sql_attempted: bool
    execution_trace: list[str]
    sql_guardrail_violations: int
    messages: list[dict[str, str]]
    user_context: dict[str, str]
    answer: str
