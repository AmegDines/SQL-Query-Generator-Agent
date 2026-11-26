from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


def trim_conversation_history(messages: list[dict[str, str]], max_turns: int = 6) -> list[dict[str, str]]:
    """Keep the most recent complete user/assistant turns, bounded to max_turns."""
    if max_turns <= 0:
        return []
    return list(messages[-(max_turns * 2):])


class Checkpointer:
    """Owns LangGraph checkpoint resources and their lifecycle."""

    def __init__(self, settings: Any) -> None:
        self.connection: sqlite3.Connection | None = None
        self.pool: Any | None = None
        if settings.checkpoint_database_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self.pool = ConnectionPool(
                conninfo=settings.checkpoint_database_url.get_secret_value(), min_size=1, max_size=50,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}, open=True,
            )
            self.saver = PostgresSaver(self.pool)
        else:
            settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(settings.checkpoint_path, check_same_thread=False, timeout=30)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.saver = SqliteSaver(self.connection)
        self.saver.setup()

    def close(self) -> None:
        if self.pool:
            self.pool.close()
        if self.connection:
            self.connection.close()
