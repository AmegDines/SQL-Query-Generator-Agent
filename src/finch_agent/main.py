from __future__ import annotations

import asyncio
import logging
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .limiter import EventDeduplicator, RateLimiter
from .models import AskRequest, AskResponse
from .service import FinchService

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
event_deduplicator = EventDeduplicator()


slack_handler: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global slack_handler
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=settings.worker_threads))
    try:
        import anyio
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.worker_threads
    except ImportError:
        pass
    app.state.service = FinchService(settings)
    if settings.slack_bot_token and settings.slack_signing_secret:
        try:
            from slack_bolt.async_app import AsyncApp as SlackAsyncApp
            from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
        except ImportError as exc:
            raise RuntimeError("Install the slack extra to enable Slack events") from exc
        bolt = SlackAsyncApp(
            token=settings.slack_bot_token.get_secret_value(),
            signing_secret=settings.slack_signing_secret.get_secret_value(),
            process_before_response=False,
        )
        register_slack(bolt, app.state.service)
        slack_handler = AsyncSlackRequestHandler(bolt)
    yield
    if hasattr(app.state, "service"):
        app.state.service.close()


app = FastAPI(title="Finch Agent", version="1.0.0", lifespan=lifespan)


if settings.app_env == "production":
    missing = [name for name, present in {
        "API_AUTH_TOKEN": bool(settings.api_auth_token),
        "POSTGRES_REPORTING_DATABASE": settings.database_url.startswith("postgresql+psycopg://"),
        "SLACK_BOT_TOKEN": bool(settings.slack_bot_token),
        "SLACK_SIGNING_SECRET": bool(settings.slack_signing_secret),
        "OPENAI_API_KEY": bool(settings.openai_api_key) if settings.llm_provider == "openai" else True,
        "LLM_PROVIDER=openai": settings.llm_provider == "openai",
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
    if slack_handler is None:
        raise HTTPException(status_code=503, detail="Slack integration is not configured")
    return await slack_handler.handle(request)


@app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(authenticate)])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    if not limiter.allow(payload.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    try:
        return request.app.state.service.ask(payload)
    except Exception as exc:
        log.error("request_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="The request could not be completed") from exc


def slack_blocks(response: AskResponse) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": response.answer[:2800]}}]
    if response.columns:
        columns = response.columns[:10]
        table_rows = [[{"type": "raw_text", "text": str(value)[:80]} for value in columns]]
        for row in response.rows[:19]:
            table_rows.append([{"type": "raw_text", "text": str(value)[:80]} for value in row[:10]])
        blocks.append({"type": "table", "rows": table_rows,
                       "column_settings": [{"is_wrapped": True} for _ in columns]})
        if len(response.rows) > 19 or len(response.columns) > 10:
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Slack table display is limited to the first 19 rows and 10 columns; the API response contains up to the configured result limit."}]})
    if response.sql:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"Executed SQL (read-only):\n```{response.sql[:2500]}```"}})
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"Request `{response.request_id[:8]}`"}]})
    if response.citations:
        cite = "\n".join(f"• {c.document}, p. {c.page}" for c in response.citations)
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": cite[:2800]}]})
    return blocks


def register_slack(app_: Any, service: FinchService) -> None:
    async def process_message(event: dict[str, Any], body: dict[str, Any], say: Any, logger: Any, client: Any) -> None:
        if event_deduplicator.is_duplicate(body.get("event_id")):
            return
        if event.get("bot_id") or event.get("subtype"):
            return
        if event.get("type") == "message" and event.get("channel_type") != "im":
            return
        channel = event.get("channel", "")
        if settings.allowed_channels and event.get("channel_type") != "im" and channel not in settings.allowed_channels:
            return
        user = event.get("user", "unknown")
        if not limiter.allow(f"{user}:{channel}"):
            await say(text="You're sending requests too quickly. Please wait a minute.", thread_ts=event.get("thread_ts") or event.get("ts"))
            return
        question = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
        if not question:
            await say(text="What would you like me to look up?", thread_ts=event.get("thread_ts") or event.get("ts"))
            return
        response_thread = event.get("thread_ts") or event.get("ts")
        status_message = await say(text="I’m checking the data and reports now…", thread_ts=response_thread)
        try:
            thread_ts = response_thread
            response = await asyncio.to_thread(
                service.ask, AskRequest(question=question, user_id=user, channel_id=channel, thread_ts=thread_ts)
            )
            await client.chat_update(channel=channel, ts=status_message["ts"],
                                     blocks=slack_blocks(response), text=response.answer)
        except Exception as exc:
            logger.error("slack_request_failed", error_type=type(exc).__name__)
            await client.chat_update(channel=channel, ts=status_message["ts"],
                                     text="I couldn't complete that request. Please try again later.")

    @app_.event("app_mention")
    async def handle_mention(event: dict[str, Any], body: dict[str, Any], say: Any, logger: Any, client: Any) -> None:
        await process_message(event, body, say, logger, client)

    @app_.event("message")
    async def handle_direct_message(event: dict[str, Any], body: dict[str, Any], say: Any, logger: Any, client: Any) -> None:
        await process_message(event, body, say, logger, client)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("finch_agent.main:app", host=settings.host, port=settings.port)
