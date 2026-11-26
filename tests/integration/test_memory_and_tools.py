import sqlite3

from finch_agent.agents.rag_agent import RAGAgent
from finch_agent.config.settings import Settings
from finch_agent.memory.checkpointer import trim_conversation_history
from finch_agent.orchestration.state import AskRequest, RouteDecision, SourceCitation
from finch_agent.orchestration.workflow import FinchService
from finch_agent.tools.sql_engine import ReadOnlyQueryEngine
from finch_agent.orchestration.state import ToolResult


class ConversationModel:
    def route(self, question, system_prompt=None):
        if "20 seats" in question.lower():
            return RouteDecision(destination="sql_agent")
        return RouteDecision(destination="rag_agent" if "discount" in question.lower() or "policy" in question.lower() else "sql_agent")

    def synthesize_report(self, question, passages):
        return "Enterprise discount is 10% [enterprise_discount_policy.md]."

    def generate_sql(self, question, metadata, error=None):
        assert "10%" in question
        return "SELECT 20 * (100 - 10) / 100 AS total_price"


class EmptyVectors:
    def search_metadata(self, query):
        return []

    def search_reports(self, query):
        return [SourceCitation(document="enterprise_discount_policy.md", page=1,
                               excerpt="Enterprise discount is 10%.")]


def test_table_count_question_uses_live_schema_without_llm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with sqlite3.connect(tmp_path / "finch.db") as connection:
        for name in ("customers", "invoices", "products", "contracts"):
            connection.execute(f"CREATE TABLE {name} (id INTEGER)")

    class NoRoutingModel:
        def route(self, question, system_prompt=None):
            raise AssertionError("Schema count requests should not require an LLM route call")

    settings = Settings(
        app_env="test", llm_provider="mock", database_url="sqlite:///./finch.db",
        metadata_catalog_path=tmp_path / "catalog.json", reports_directory=tmp_path / "reports",
        checkpoint_path=tmp_path / "checkpoints.sqlite", chroma_path=tmp_path / "chroma",
    )
    (tmp_path / "catalog.json").write_text("[]", encoding="utf-8")
    service = FinchService(settings, llm=NoRoutingModel(), vectors=EmptyVectors())
    response = service.ask(AskRequest(question="How many tables are there in the database?"))

    assert "4 tables" in response.answer
    assert response.rows == [["contracts"], ["customers"], ["invoices"], ["products"]]
    assert response.trajectory == ["sql_agent"]
    service.close()


def test_sql_tool_returns_live_schema_and_structured_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "finch.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE products (sku TEXT, price REAL, password TEXT)")
        connection.execute("INSERT INTO products VALUES ('A-1', 100, 'must-not-leak')")
    settings = Settings(database_url="sqlite:///./finch.db", query_max_rows=10)
    tool = ReadOnlyQueryEngine(settings)
    schema = tool.get_schema_context()
    assert schema[0]["table_name"] == "products"
    assert {item["name"] for item in schema[0]["columns"]} == {"sku", "price"}
    assert schema[0]["sample_rows"] == [{"sku": "A-1", "price": 100.0}]
    result = tool.execute("SELECT sku, price FROM products")
    assert isinstance(result, ToolResult)
    assert result.status == "success"
    assert result.rows == [{"sku": "A-1", "price": 100.0}]
    assert result.tables_used == ["products"]
    tool.engine.dispose()


def test_thread_followup_uses_policy_history_and_resets_working_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "finch.db"
    with sqlite3.connect(database):
        pass
    settings = Settings(app_env="test", llm_provider="mock", database_url="sqlite:///./finch.db",
                        metadata_catalog_path=tmp_path / "catalog.json", reports_directory=tmp_path / "reports",
                        checkpoint_path=tmp_path / "checkpoints.sqlite", chroma_path=tmp_path / "chroma")
    (tmp_path / "catalog.json").write_text("[]", encoding="utf-8")
    service = FinchService(settings, llm=ConversationModel(), vectors=EmptyVectors())
    first = service.ask(AskRequest(question="What is the Enterprise discount?", user_id="u1",
                                   channel_id="C1", thread_ts="T1"))
    assert first.trajectory == ["rag_agent"]
    second = service.ask(AskRequest(question="What about for 20 seats?", user_id="u1",
                                    channel_id="C1", thread_ts="T1"))
    assert second.trajectory == ["sql_agent"]
    assert second.rows == [[18.0]]
    config = {"configurable": {"thread_id": "C1:T1"}}
    state = service.graph.get_state(config).values
    assert state["trajectory"] == ["sql_agent"]
    assert state["sql_results"] == [{"total_price": 18}]
    assert len(state["messages"]) == 4
    service.ask(AskRequest(question="What is the discount policy?", user_id="u1",
                           channel_id="C1", thread_ts="T1"))
    state = service.graph.get_state(config).values
    assert state["sql_results"] == []
    assert state["rag_context"]
    assert len(state["messages"]) == 6
    assert trim_conversation_history(state["messages"], max_turns=1) == state["messages"][-2:]
    service.long_term_store.save_user_preference("u1", "currency", "USD")
    assert service.long_term_store.get_user_context("u1")["currency"] == "USD"
    service.close()


def test_rag_agent_keeps_source_cited_structured_chunks():
    result = RAGAgent(EmptyVectors())({"question": "Enterprise discount policy"})
    assert result["citations"][0]["document"] == "enterprise_discount_policy.md"
    assert "10%" in result["rag_context"][0]
