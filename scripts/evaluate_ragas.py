from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from finch_agent.config.settings import get_settings
from finch_agent.core.observability import init_observability
from finch_agent.orchestration.state import AskRequest
from finch_agent.orchestration.workflow import FinchService


async def score(question: str, answer: str, contexts: list[str], reference: str,
                model_name: str, api_key: str) -> dict[str, float]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
    except ImportError as exc:
        raise RuntimeError('Install the evaluation extra with `pip install -e ".[evaluation]"`') from exc

    client = AsyncOpenAI(api_key=api_key)
    try:
        evaluator_llm = llm_factory(model_name, client=client)
        evaluator_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)
        faithfulness = await Faithfulness(llm=evaluator_llm).ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        relevancy = await AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings).ascore(
            user_input=question, response=answer
        )
        precision = await ContextPrecision(llm=evaluator_llm).ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        )
        return {"faithfulness": float(faithfulness.value), "answer_relevancy": float(relevancy.value),
                "context_precision": float(precision.value)}
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score golden RAG workflow cases with RAGAS (tracing defaults off).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N RAG cases")
    parser.add_argument("--trace", action="store_true", help="Opt in to LangSmith traces")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    settings = get_settings()
    init_observability(settings, enabled=args.trace)
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        parser.error("Set LLM_PROVIDER=openai and OPENAI_API_KEY before running RAGAS scoring")

    cases_path = settings.metadata_catalog_path.parent / "golden_queries.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rag_cases = [case for case in cases if "rag_agent" in case.get("expected_route", []) and case.get("ground_truth")]
    if args.limit:
        rag_cases = rag_cases[:args.limit]
    service = FinchService(settings)
    results: list[dict[str, Any]] = []
    try:
        for case in rag_cases:
            response = service.ask(AskRequest(question=case["query"], user_id=f"ragas-{case['id']}"))
            contexts = [citation.excerpt for citation in response.citations]
            if not contexts:
                results.append({"id": case["id"], "error": "No relevant report contexts retrieved"})
                continue
            scores = asyncio.run(score(case["query"], response.answer, contexts, case["ground_truth"],
                                       settings.openai_model, settings.openai_api_key.get_secret_value()))
            results.append({"id": case["id"], "scores": scores,
                            "sources": [citation.document for citation in response.citations]})
    finally:
        service.close()
    valid = [item["scores"] for item in results if "scores" in item]
    averages = {metric: sum(item[metric] for item in valid) / len(valid) if valid else 0.0
                for metric in ("faithfulness", "answer_relevancy", "context_precision")}
    print(json.dumps({"evaluated": len(results), "scored": len(valid), "averages": averages, "cases": results}, indent=2))
    return 0 if valid and all(value >= 0.80 for value in averages.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
