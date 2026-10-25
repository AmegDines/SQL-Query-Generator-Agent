from __future__ import annotations

import json
import sys

from finch_agent.config import get_settings
from finch_agent.models import AskRequest
from finch_agent.service import FinchService


def main() -> int:
    settings = get_settings()
    cases = json.loads((settings.metadata_catalog_path.parent / "golden_queries.json").read_text(encoding="utf-8"))
    service = FinchService(settings)
    results = []
    for case in cases:
        response = service.ask(AskRequest(question=case["question"], user_id=f"golden-{case['id']}"))
        errors = []
        if response.route != case["route"]:
            errors.append(f"route={response.route}, expected={case['route']}")
        if case.get("sql_contains"):
            sql = (response.sql or "").lower()
            missing = [part for part in case["sql_contains"] if part.lower() not in sql]
            if missing:
                errors.append(f"SQL missing required tokens: {missing}")
        if case.get("columns") and response.columns != case["columns"]:
            errors.append(f"columns={response.columns}, expected={case['columns']}")
        if case.get("expected_rows") is not None and response.rows != case["expected_rows"]:
            errors.append(f"rows={response.rows}, expected={case['expected_rows']}")
        if case.get("citation_contains"):
            cited_documents = [citation.document for citation in response.citations]
            missing = [item for item in case["citation_contains"]
                       if not any(item in document for document in cited_documents)]
            if missing:
                errors.append(f"missing citation documents: {missing}")
        results.append({"id": case["id"], "passed": not errors, "errors": errors,
                        "retry_count": response.retry_count})
    passed = sum(item["passed"] for item in results)
    sql_cases = [item for item in results if item["id"] == "TC_REG_01"]
    first_pass = sum(item["passed"] and item["retry_count"] == 0 for item in sql_cases)
    print(json.dumps({"passed": passed, "total": len(results), "cases": results,
                      "sql_first_pass_cases": first_pass,
                      "sql_first_pass_accuracy": first_pass / len(sql_cases) if sql_cases else None}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
