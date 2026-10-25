import json
import sqlite3

from finch_agent.config import Settings
from finch_agent.models import AskRequest, RouteDecision
from finch_agent.service import FinchService


class EmptyVectors:
    def search_metadata(self, query: str):
        return []

    def search_reports(self, query: str):
        return []


class CorrectingModel:
    def __init__(self):
        self.generated = 0

    def route(self, question: str):
        return RouteDecision(destination="sql_agent")

    def generate_sql(self, question: str, metadata, error=None):
        self.generated += 1
        if self.generated == 1:
            return "SELECT SUM(missing_column) AS gross_bookings FROM finance_bookings"
        return "SELECT SUM(gross_bookings) AS gross_bookings FROM finance_bookings"

    def synthesize_report(self, question: str, passages):
        return ""


def test_sql_agent_corrects_database_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with sqlite3.connect("finch.db") as connection:
        connection.execute("CREATE TABLE finance_bookings (gross_bookings DECIMAL)")
        connection.executemany("INSERT INTO finance_bookings VALUES (?)", [(10,), (15,)])
    (tmp_path / "catalog.json").write_text(json.dumps([]), encoding="utf-8")
    settings = Settings(
        app_env="test", llm_provider="mock", database_url="sqlite:///./finch.db",
        metadata_catalog_path=tmp_path / "catalog.json", reports_directory=tmp_path / "reports",
        checkpoint_path=tmp_path / "checkpoints.sqlite", chroma_path=tmp_path / "chroma",
    )
    service = FinchService(settings, llm=CorrectingModel(), vectors=EmptyVectors())
    response = service.ask(AskRequest(question="total gross bookings", user_id="test-user"))
    service.close()
    assert response.route == "sql_agent"
    assert response.retry_count == 1
    assert response.columns == ["gross_bookings"]
    assert response.rows == [[25]]
