from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

from finch_agent.config.settings import Settings

F = TypeVar("F", bound=Callable[..., Any])
log = structlog.get_logger(__name__)


def init_observability(settings: Settings | None = None, *, enabled: bool | None = None) -> bool:
    """Configure LangSmith only when explicitly enabled and an API key is available."""
    if settings is None:
        from finch_agent.config.settings import get_settings

        settings = get_settings()
    key = settings.langchain_api_key or settings.langsmith_api_key
    requested = (settings.langchain_tracing_v2 or settings.langsmith_tracing) if enabled is None else enabled
    active = bool(requested and key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if active else "false"
    os.environ["LANGSMITH_TRACING"] = "true" if active else "false"
    if active:
        os.environ["LANGCHAIN_API_KEY"] = key.get_secret_value()
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project or settings.langsmith_project
        os.environ["LANGSMITH_PROJECT"] = settings.langchain_project or settings.langsmith_project
    else:
        os.environ.pop("LANGCHAIN_API_KEY", None)
        os.environ.pop("LANGSMITH_API_KEY", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGSMITH_PROJECT", None)
    log.info("langsmith_observability_configured", enabled=active)
    return active


def trace_step(*, run_type: str) -> Callable[[F], F]:
    """Lazily trace a tool/retriever call; disabled operation has no LangSmith side effects."""
    def decorate(function: F) -> F:
        try:
            from langsmith import traceable

            traced = traceable(run_type=run_type)(function)
        except ImportError:
            traced = function

        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
                return traced(*args, **kwargs)
            return function(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorate
