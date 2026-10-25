import random
import re
import sqlite3
import time
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from .config import Settings

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|GRANT|REVOKE|CALL|COPY|ATTACH|DETACH|VACUUM|PRAGMA|EXEC|EXECUTE|INTO)\b",
    re.IGNORECASE,
)
_DANGEROUS_FUNCTION = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_sleep|pg_advisory_[a-z_]+|pg_notify|"
    r"pg_terminate_backend|pg_cancel_backend|pg_reload_conf|pg_stat_reset|pg_create_restore_point|"
    r"pg_switch_wal|pg_logical_slot_[a-z_]+|nextval|setval|set_config|lo_(create|import|unlink|write)|"
    r"dblink([a-z_]*)|load_extension|readfile|writefile)\s*\(", re.IGNORECASE
)
_COMMENT = re.compile(r"--[^\r\n]*|/\*[\s\S]*?\*/")


def validate_read_only_sql(sql: str) -> str:
    cleaned = _COMMENT.sub(" ", sql).strip()
    cleaned = cleaned.removesuffix(";").strip()
    if not cleaned or ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise ValueError("Only read-only SELECT queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise ValueError("The query contains a prohibited SQL operation")
    if _DANGEROUS_FUNCTION.search(cleaned):
        raise ValueError("The query contains a prohibited database function")
    return cleaned


class ReadOnlyQueryEngine:
    def __init__(self, settings: Settings) -> None:
        if not settings.database_read_only:
            raise ValueError("DATABASE_READ_ONLY must remain enabled")
        url = settings.database_url
        if url.startswith("sqlite:") and "mode=ro" not in url:
            # SQLite connections are opened read-only at the driver layer as a second guard.
            if url.startswith("sqlite:///"):
                file_path = url.removeprefix("sqlite:///").split("?", 1)[0].replace("\\", "/")
                if re.match(r"^[A-Za-z]:/", file_path):
                    file_path = "/" + file_path
                url = f"sqlite:///file:{file_path}?mode=ro&uri=true"
        engine_options: dict[str, Any] = {"pool_pre_ping": True, "future": True}
        if not (url.startswith("sqlite:") and ":memory:" in url):
            engine_options.update(pool_size=settings.database_pool_size,
                                  max_overflow=settings.database_max_overflow)
        self.engine: Engine = create_engine(url, **engine_options)
        self.timeout = settings.query_timeout_seconds
        self.max_rows = settings.query_max_rows

    def execute(self, sql: str) -> tuple[list[str], list[list[Any]], bool]:
        safe_sql = validate_read_only_sql(sql)
        for attempt in range(3):
            try:
                return self._execute_once(safe_sql)
            except OperationalError as exc:
                if attempt == 2 or not self._is_transient(exc):
                    raise
                time.sleep(0.1 * (2 ** attempt) + random.uniform(0, 0.1))
        raise RuntimeError("Database retry loop ended unexpectedly")

    def _execute_once(self, safe_sql: str) -> tuple[list[str], list[list[Any]], bool]:
        with self.engine.connect() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql(f"SET LOCAL statement_timeout = {self.timeout * 1000}")
            elif self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql(f"PRAGMA query_only = ON")
                connection.exec_driver_sql(f"PRAGMA busy_timeout = {self.timeout * 1000}")
            result = connection.execute(text(safe_sql))
            if not result.returns_rows:
                raise ValueError("Query did not return tabular results")
            columns = list(result.keys())
            fetched = [list(row) for row in result.fetchmany(self.max_rows + 1)]
            return columns, fetched[: self.max_rows], len(fetched) > self.max_rows

    @staticmethod
    def _is_transient(exc: OperationalError) -> bool:
        original = exc.orig
        sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
        if sqlstate:
            return sqlstate.startswith("08") or sqlstate in {"40001", "40P01", "57P01", "57P03"}
        error_code = getattr(original, "sqlite_errorcode", None)
        transient_sqlite_errors = {
            sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
        }
        return error_code is not None and (error_code & 0xFF) in transient_sqlite_errors
