from __future__ import annotations

import html
import re
from typing import Any


class GuardrailViolation(ValueError):
    """A deterministic validation rule rejected input or generated SQL."""


class InputRejected(GuardrailViolation):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


_INJECTION = re.compile(
    r"\b(ignore|disregard|override|forget)\b.{0,50}\b(previous|prior|above|system|developer|instructions?)\b"
    r"|\b(reveal|show|print|dump)\b.{0,40}\b(system prompt|hidden prompt|all tables|credentials|secrets)\b"
    r"|\b(jailbreak|developer mode|act as an unrestricted|bypass (?:all )?safety)\b",
    re.IGNORECASE | re.DOTALL,
)
_IN_SCOPE = re.compile(
    r"\b(company|business|operation|finance|financial|revenue|expenses?|cost|budget|invoices?|procurement|"
    r"purchases?|supplier|vendor|inventory|catalog|product|price|pricing|discount|customer|sales|bookings?|"
    r"contracts?|acv|overdue|credit.?risk|hotel|dining|cloud|quarter|department|approval|threshold|license|"
    r"policy|policies|report|10-k|10k|sop|sql|database|table|account|forecast|payment|order)\b",
    re.IGNORECASE,
)
_CLEARLY_OUT_OF_SCOPE = re.compile(
    r"\b(weather|football|soccer|basketball|baseball|movie|movies|celebrity|recipe|cooking|vacation|"
    r"election|president|music|song lyrics|video game)\b", re.I
)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I)
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)")
_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")
_SECRET_ASSIGNMENT = re.compile(
    r"\b(api[_-]?key|access[_-]?token|password|secret|authorization)\b\s*[:=]\s*[^\s,;]+", re.I
)
_INSTRUCTION = re.compile(
    r"(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+instructions?"
    r"|(?:reveal|print|dump)\s+(?:the\s+)?(?:system prompt|credentials|secrets)"
    r"|\b(?:you are now|act as)\s+(?:an?\s+)?(?:system|developer|admin|unrestricted)",
    re.I,
)
_SENSITIVE_IDENTIFIER = re.compile(r"\b(?:password|passwd|ssn|social_security|api_key|access_token|token|salary)\b", re.I)


def validate_user_input(question: str) -> None:
    """Reject unsafe, oversized, empty, or clearly out-of-domain requests before routing."""
    if not isinstance(question, str) or not question.strip():
        raise InputRejected("Please enter a question about company operations or data.")
    if len(question) > 2000:
        raise InputRejected("Please keep your question under 2,000 characters.")
    if _INJECTION.search(question):
        raise InputRejected("I can help with company information, but I can't follow requests to reveal or override system instructions.")
    if not _IN_SCOPE.search(question) and _CLEARLY_OUT_OF_SCOPE.search(question):
        raise InputRejected("I can help with company operations, finance, procurement, catalog, and pricing questions. Please ask about one of those areas.")


def validate_sql_query(sql: str) -> str:
    """Validate one read-only SELECT, reject sensitive identifiers, and bound unbounded rows."""
    from finch_agent.tools.sql_engine import validate_read_only_sql

    if not isinstance(sql, str) or not sql.strip():
        raise GuardrailViolation("SQL is empty")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise GuardrailViolation("SQL comments are not allowed")
    if _SENSITIVE_IDENTIFIER.search(sql):
        raise GuardrailViolation("The query references a sensitive column")
    try:
        normalized = validate_read_only_sql(sql)
    except ValueError as exc:
        raise GuardrailViolation(str(exc)) from exc
    # Existing validator allows only a single optional trailing semicolon.
    if re.search(r";\s*\S", sql):
        raise GuardrailViolation("Only one SQL statement is allowed")
    if not re.search(r"\bLIMIT\b", normalized, re.I):
        normalized = f"{normalized} LIMIT 100"
    return normalized


def sanitize_retrieved_docs(text: str, source: str = "unknown") -> str:
    """Neutralize instruction-like document text and encode the remaining text as passive XML data."""
    safe_source = html.escape(str(source).replace('"', "'"), quote=True)
    cleaned = _INSTRUCTION.sub("[instruction-like text removed]", str(text))
    return f'<retrieved_policy source="{safe_source}">{html.escape(cleaned, quote=False)}</retrieved_policy>'


def redact_sensitive_output(text: str) -> str:
    """Mask common PII and credential forms before text is returned to users."""
    redacted = _SSN.sub("[REDACTED SSN]", str(text))
    redacted = _EMAIL.sub("[REDACTED EMAIL]", redacted)
    redacted = _PHONE.sub("[REDACTED PHONE]", redacted)
    redacted = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
    return _KEY.sub("[REDACTED KEY]", redacted)


def validate_final_response(
    response: str,
    *,
    question: str = "",
    sql_results: list[dict[str, Any]] | None = None,
    rag_context: list[str] | None = None,
    factual: bool = False,
) -> str:
    """Prevent unsupported factual answers and redact sensitive text."""
    if factual and not sql_results and not rag_context:
        response = "I couldn't find verified data or policy rules to answer that."
    return redact_sensitive_output(response)
