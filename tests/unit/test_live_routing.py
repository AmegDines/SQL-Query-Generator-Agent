import pytest

from finch_agent.core.llm import (
    GroqLanguageModel,
    OpenAILanguageModel,
    _parse_route_decision,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sql_agent", "sql_agent"),
        ("Route to the SQL agent first, then return to the supervisor.", "sql_agent"),
        ("rag_agent then sql_agent", "rag_agent"),
        ('{"destination": "doc_rag_agent"}', "rag_agent"),
        ("No governed route is clear; clarification", "clarification"),
    ],
)
def test_route_parser_accepts_labels_in_short_explanations(raw: str, expected: str) -> None:
    assert _parse_route_decision(raw).destination == expected


@pytest.mark.parametrize("model_type", [GroqLanguageModel, OpenAILanguageModel])
def test_live_models_add_strict_route_output_instruction(model_type) -> None:
    model = object.__new__(model_type)
    seen: dict[str, str] = {}

    def complete(system: str, question: str) -> str:
        seen["system"] = system
        seen["question"] = question
        return "sql_agent"

    model._complete = complete
    decision = model.route("Which active contracts are in India?")

    assert decision.destination == "sql_agent"
    assert "Return one label and no explanation" in seen["system"]
    assert seen["question"] == "Which active contracts are in India?"
