import os

from finch_agent.config.settings import Settings
from finch_agent.core.observability import init_observability, trace_step


def test_observability_is_disabled_without_key(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "inherited-key")
    settings = Settings(langchain_tracing_v2=True, langchain_api_key=None)
    assert init_observability(settings) is False
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert "LANGCHAIN_API_KEY" not in os.environ


def test_trace_step_runs_normally_when_tracing_is_disabled(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    @trace_step(run_type="tool")
    def add_one(value):
        return value + 1

    assert add_one(2) == 3
