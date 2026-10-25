from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from finch_agent.config import get_settings
from finch_agent.models import AskRequest
from finch_agent.service import FinchService


async def score(question: str, answer: str, contexts: list[str], model_name: str, api_key: str) -> dict[str, float]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
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
        return {"faithfulness": float(faithfulness.value), "answer_relevancy": float(relevancy.value)}
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a report answer with RAGAS faithfulness and relevance.")
    parser.add_argument("question", help="Question to evaluate against the configured report corpus")
    args = parser.parse_args()
    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        parser.error("Set LLM_PROVIDER=openai and OPENAI_API_KEY before running RAGAS scoring")
    service = FinchService(settings)
    try:
        response = service.ask(AskRequest(question=args.question, user_id="ragas-evaluation"))
        contexts = [citation.excerpt for citation in response.citations]
        if response.route != "doc_rag_agent" or not contexts:
            raise RuntimeError("The question did not retrieve report passages; no RAGAS score was produced")
        scores = asyncio.run(score(args.question, response.answer, contexts,
                                   settings.openai_model, settings.openai_api_key.get_secret_value()))
        output: dict[str, Any] = {"question": args.question, "scores": scores,
                                  "citations": [citation.model_dump() for citation in response.citations]}
        print(json.dumps(output, indent=2))
        return 0 if scores["faithfulness"] >= 0.85 and scores["answer_relevancy"] >= 0.80 else 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
