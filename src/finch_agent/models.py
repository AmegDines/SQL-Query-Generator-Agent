from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    user_id: str = "api-user"
    channel_id: str | None = None
    thread_ts: str | None = None


class RouteDecision(BaseModel):
    destination: Literal["sql_agent", "doc_rag_agent", "clarification"]
    reasoning: str = ""


class SourceCitation(BaseModel):
    document: str
    page: int
    excerpt: str


class AskResponse(BaseModel):
    route: str
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    retry_count: int = 0
    citations: list[SourceCitation] = Field(default_factory=list)
    request_id: str


