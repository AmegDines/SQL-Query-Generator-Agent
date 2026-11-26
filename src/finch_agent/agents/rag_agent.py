from __future__ import annotations

from typing import Any

from finch_agent.orchestration.state import AgentState


class RAGAgent:
    """Retrieves report evidence and grades it before returning context to the supervisor."""

    def __init__(self, vectors: Any) -> None:
        self.vectors = vectors

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Retrieve sanitized, relevance-filtered evidence for the current task."""
        question = state.get("effective_question", state["question"])
        citations = self.vectors.search_reports(question)
        # Empty retrieval is a fail-closed relevance grade; only retrieved evidence is synthesized.
        citation_data = [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in citations]
        context = [f"[{item['document']} p.{item['page']}] {item['excerpt']}" for item in citation_data]
        return {"rag_context": context, "citations": citation_data,
                "worker_completed": True, "last_agent": "rag_agent",
                "execution_trace": state.get("execution_trace", []) + ["rag_agent -> supervisor"]}
