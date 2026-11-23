from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    slack_bot_token: SecretStr | None = None
    slack_app_token: SecretStr | None = None          # xapp-… required for Socket Mode
    slack_signing_secret: SecretStr | None = None     # optional in Socket Mode
    slack_allowed_channels: str = ""
    api_auth_token: SecretStr | None = None
    llm_provider: Literal["mock", "openai", "groq"] = "mock"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    database_url: str = "sqlite:///./data/finch.db"
    database_read_only: bool = True
    query_timeout_seconds: int = Field(default=10, ge=1, le=120)
    query_max_rows: int = Field(default=200, ge=1, le=1000)
    database_pool_size: int = Field(default=20, ge=1, le=200)
    database_max_overflow: int = Field(default=30, ge=0, le=200)
    worker_threads: int = Field(default=100, ge=10, le=500)
    metadata_catalog_path: Path = Path("./data/catalog.json")
    reports_directory: Path = Path("./data/reports")
    chroma_path: Path = Path("./data/chroma")
    chroma_host: str | None = None
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    chroma_api_key: SecretStr | None = None
    checkpoint_path: Path = Path("./data/checkpoints.sqlite")
    checkpoint_database_url: SecretStr | None = None
    chroma_top_k: int = Field(default=5, ge=1, le=20)
    report_chunk_size: int = Field(default=1200, ge=200, le=5000)
    report_max_file_mb: int = Field(default=20, ge=1, le=100)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "finch-agent"

    @field_validator(
        "slack_bot_token", "slack_app_token", "slack_signing_secret", "api_auth_token", "openai_api_key",
        "groq_api_key", "chroma_api_key", "checkpoint_database_url", "langsmith_api_key", mode="before",
    )
    @classmethod
    def normalize_empty_secrets(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_read_only")
    @classmethod
    def enforce_read_only(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Database access must always be read-only")
        return value

    @property
    def allowed_channels(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.slack_allowed_channels.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
