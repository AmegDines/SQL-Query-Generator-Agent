import pytest

from finch_agent.llm import MockLanguageModel


@pytest.mark.parametrize(("question", "route"), [
    ("What was gross bookings last quarter?", "sql_agent"),
    ("Summarize the 10-K report", "doc_rag_agent"),
    ("Can you help me with this?", "clarification"),
])
def test_mock_supervisor_routes_obvious_intents(question: str, route: str) -> None:
    assert MockLanguageModel().route(question).destination == route
