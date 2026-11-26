from __future__ import annotations

import asyncio
import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from finch_agent.config.settings import get_settings
from finch_agent.core.limiter import RateLimiter
from finch_agent.core.observability import init_observability
from finch_agent.integrations.slack import register_slack
from finch_agent.orchestration.state import AskRequest, AskResponse
from finch_agent.orchestration.workflow import FinchService

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
structlog.configure(processors=[
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.format_exc_info,
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger(__name__)
limiter = RateLimiter(settings.rate_limit_per_minute)


_socket_handler: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _socket_handler
    init_observability(settings)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=settings.worker_threads))
    try:
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.worker_threads
    except ImportError:
        pass
    app.state.service = FinchService(settings)
    if settings.slack_bot_token and settings.slack_app_token:
        try:
            from slack_bolt.async_app import AsyncApp as SlackAsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        except ImportError as exc:
            raise RuntimeError("Install the slack extra to enable Slack Socket Mode") from exc
        bolt = SlackAsyncApp(
            token=settings.slack_bot_token.get_secret_value(),
            signing_secret=settings.slack_signing_secret.get_secret_value() if settings.slack_signing_secret else None,
        )
        register_slack(bolt, app.state.service)
        _socket_handler = AsyncSocketModeHandler(bolt, settings.slack_app_token.get_secret_value())
        await _socket_handler.start_async()
        log.info("slack_socket_mode_started", model=settings.groq_model if settings.llm_provider == "groq" else settings.openai_model)
    yield
    if _socket_handler:
        await _socket_handler.close_async()
    if hasattr(app.state, "service"):
        app.state.service.close()


app = FastAPI(title="Finch Agent", version="1.0.0", lifespan=lifespan)


if settings.app_env == "production":
    missing = [name for name, present in {
        "API_AUTH_TOKEN": bool(settings.api_auth_token),
        "POSTGRES_REPORTING_DATABASE": settings.database_url.startswith("postgresql+psycopg://"),
        "SLACK_BOT_TOKEN": bool(settings.slack_bot_token),
        "SLACK_APP_TOKEN": bool(settings.slack_app_token),
        "OPENAI_API_KEY": bool(settings.openai_api_key) if settings.llm_provider == "openai" else True,
        "GROQ_API_KEY": bool(settings.groq_api_key) if settings.llm_provider == "groq" else True,
        "LLM_PROVIDER=openai|groq": settings.llm_provider in ("openai", "groq"),
        "CHECKPOINT_DATABASE_URL": bool(settings.checkpoint_database_url),
        "CHROMA_HOST": bool(settings.chroma_host),
        "CHROMA_API_KEY": bool(settings.chroma_api_key),
        "CHROMA_SSL": settings.chroma_ssl,
        "LANGSMITH_API_KEY": bool(settings.langsmith_api_key) if settings.langsmith_tracing else True,
    }.items() if not present]
    if missing:
        raise RuntimeError("Missing production configuration: " + ", ".join(missing))


def authenticate(authorization: str | None = Header(default=None)) -> None:
    if settings.app_env == "production":
        expected = settings.api_auth_token.get_secret_value() if settings.api_auth_token else ""
        supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not expected or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ready" if hasattr(request.app.state, "service") else "starting"},
                        status_code=200 if hasattr(request.app.state, "service") else 503)


@app.post("/slack/events")
async def slack_events(request: Request) -> Any:
    raise HTTPException(status_code=404, detail="Slack is running in Socket Mode — HTTP events endpoint is not used")


@app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(authenticate)])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    if not limiter.allow(payload.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        return request.app.state.service.ask(payload)
    except Exception as exc:
        log.error("request_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="The request could not be completed") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("finch_agent.main:app", host=settings.host, port=settings.port)
