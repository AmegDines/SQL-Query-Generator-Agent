from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class LongTermStore:
    """Small SQLite store for user preferences and verified, user-scoped facts."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL, preference_key TEXT NOT NULL, value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, preference_key))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS verified_facts (
                user_id TEXT NOT NULL, fact_key TEXT NOT NULL, value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, fact_key))""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get_user_context(self, user_id: str) -> dict[str, str]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT preference_key, value_json FROM user_preferences WHERE user_id=? "
                "UNION ALL SELECT fact_key, value_json FROM verified_facts WHERE user_id=?",
                (user_id, user_id),
            ).fetchall()
        return {str(key): str(json.loads(value)) for key, value in rows}

    def save_user_preference(self, user_id: str, key: str, value: Any) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO user_preferences(user_id,preference_key,value_json) VALUES(?,?,?) "
                "ON CONFLICT(user_id,preference_key) DO UPDATE SET value_json=excluded.value_json, "
                "updated_at=CURRENT_TIMESTAMP",
                (user_id, key, json.dumps(value, ensure_ascii=False)),
            )

    def save_verified_fact(self, user_id: str, key: str, value: Any) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO verified_facts(user_id,fact_key,value_json) VALUES(?,?,?) "
                "ON CONFLICT(user_id,fact_key) DO UPDATE SET value_json=excluded.value_json, "
                "updated_at=CURRENT_TIMESTAMP",
                (user_id, key, json.dumps(value, ensure_ascii=False)),
            )
