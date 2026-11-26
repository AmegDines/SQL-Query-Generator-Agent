from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import structlog
from finch_agent.config.settings import get_settings
from finch_agent.core.guardrails import redact_sensitive_output
from finch_agent.core.limiter import EventDeduplicator, RateLimiter
from finch_agent.orchestration.state import AskRequest, AskResponse

if TYPE_CHECKING:
    from finch_agent.orchestration.workflow import FinchService

settings = get_settings()
log = structlog.get_logger(__name__)
limiter = RateLimiter(settings.rate_limit_per_minute)
event_deduplicator = EventDeduplicator()

def slack_blocks(response: AskResponse) -> list[dict[str, Any]]:
    """Format an agent response as Slack Block Kit blocks."""
    blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": redact_sensitive_output(response.answer)[:2800]}}]
    if response.columns:
        columns = response.columns[:10]
        table_rows = [[{"type": "raw_text", "text": redact_sensitive_output(str(value))[:80]} for value in columns]]
        for row in response.rows[:19]:
            table_rows.append([{"type": "raw_text", "text": redact_sensitive_output(str(value))[:80]} for value in row[:10]])
        blocks.append({"type": "table", "rows": table_rows,
                       "column_settings": [{"is_wrapped": True} for _ in columns]})
        if len(response.rows) > 19 or len(response.columns) > 10:
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Slack table display is limited to the first 19 rows and 10 columns; the API response contains up to the configured result limit."}]})
    if response.sql:
        safe_sql = redact_sensitive_output(response.sql)
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"Executed SQL (read-only):\n```{safe_sql[:2500]}```"}})
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"Request `{response.request_id[:8]}`"}]})
    if response.citations:
        cite = "\n".join(f"• {redact_sensitive_output(c.document)}, p. {c.page}" for c in response.citations)
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": cite[:2800]}]})
    return blocks


def register_slack(app_: Any, service: FinchService) -> None:
    """Register Slack mention and direct-message handlers on a Bolt app."""
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
        trace_thread_id = f"{channel}:{response_thread}"
        status_message = await say(text="I'm checking the data and reports now...", thread_ts=response_thread)
        try:
            context_tokens = structlog.contextvars.bind_contextvars(
                thread_id=trace_thread_id, slack_user_id=user, slack_channel_id=channel
            )
            try:
                response = await asyncio.to_thread(
                    service.ask, AskRequest(question=question, user_id=user, channel_id=channel,
                                            thread_ts=response_thread)
                )
            finally:
                structlog.contextvars.reset_contextvars(**context_tokens)
            await client.chat_update(channel=channel, ts=status_message["ts"],
                                     blocks=slack_blocks(response), text=redact_sensitive_output(response.answer)[:2800])
        except Exception as exc:
            log.error("slack_request_failed", error_type=type(exc).__name__, error=str(exc))
            await client.chat_update(channel=channel, ts=status_message["ts"],
                                     text="I couldn't complete that request. Please try again later.")

    @app_.event("app_mention")
    async def handle_mention(event: dict[str, Any], body: dict[str, Any], say: Any, logger: Any, client: Any) -> None:
        await process_message(event, body, say, logger, client)

    @app_.event("message")
    async def handle_direct_message(event: dict[str, Any], body: dict[str, Any], say: Any, logger: Any, client: Any) -> None:
        await process_message(event, body, say, logger, client)

