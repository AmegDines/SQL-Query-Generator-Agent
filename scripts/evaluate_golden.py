from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from finch_agent.config.settings import get_settings
from finch_agent.core.observability import init_observability
from finch_agent.orchestration.state import AskRequest, RouteDecision
from finch_agent.orchestration.workflow import FinchService


class GoldenCaseModel:
    """Deterministic model fixture: the real workflow/tools run against curated case SQL."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def route(self, question: str, system_prompt: str | None = None) -> RouteDecision:
        route = (self.case.get("expected_route") or ["clarification"])[0]
        return RouteDecision(destination={"BLOCKED": "clarification"}.get(route, route))

    def generate_sql(self, question: str, metadata: list[dict[str, Any]], error: str | None = None) -> str:
        sql = self.case.get("reference_sql")
        if not sql:
            raise RuntimeError("This case has no reference SQL")
        return sql

    def synthesize_report(self, question: str, passages: list[dict[str, Any]]) -> str:
        return self.case.get("ground_truth", "")


def _score_case(case: dict[str, Any], response: Any) -> dict[str, Any]:
    expected_route = case.get("expected_route", [])
    blocked = expected_route == ["BLOCKED"]
    observed_route = ["BLOCKED"] if response.route == "refusal" else response.trajectory
    trajectory = observed_route == expected_route

    sql_keywords = case.get("expected_sql_keywords", [])
    if sql_keywords:
        sql_text = (response.sql or "").lower()
        sql_keywords_ok = all(keyword.lower() in sql_text for keyword in sql_keywords)
        sql_safety = bool(response.sql_executed and response.sql_guardrail_violations == 0)
        sql_ok = sql_keywords_ok and sql_safety
    else:
        sql_ok = True

    cited_sources = [citation.document for citation in response.citations]
    expected_sources = case.get("expected_sources", [])
    citation_ok = all(any(source.casefold() in doc.casefold() for doc in cited_sources)
                      for source in expected_sources)

    answer_lower = response.answer.casefold()
    factual_ok = all(str(value).casefold() in answer_lower
                     for value in case.get("expected_answer_contains", []))
    return {
        "id": case["id"], "query": case["query"], "passed": trajectory and sql_ok and citation_ok and factual_ok,
        "metrics": {"trajectory_accuracy": trajectory, "sql_execution_safety": sql_ok,
                    "citation_accuracy": citation_ok, "answer_factuality": factual_ok},
        "expected_route": expected_route, "observed_route": observed_route,
        "sql": response.sql, "sources": cited_sources, "answer": response.answer,
    }


def _write_report(results: list[dict[str, Any]], total: int, passed: int,
                  json_path: Path, markdown_path: Path, rate: float, threshold: float) -> None:
    report = {"passed": passed, "total": total, "pass_rate": rate, "threshold": threshold,
              "metric_pass_rates": {
                  metric: sum(bool(item["metrics"][metric]) for item in results) / total if total else 0.0
                  for metric in ("trajectory_accuracy", "sql_execution_safety", "citation_accuracy", "answer_factuality")
              }, "cases": results}
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Multi-Agent Golden Evaluation", "", f"Pass rate: **{passed}/{total} ({rate:.1%})**; threshold: **{threshold:.1%}**", "",
             "| Case | Trajectory | SQL safety | Citations | Factuality | Result |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in results:
        m = item["metrics"]
        check = lambda value: "PASS" if value else "FAIL"
        lines.append(f"| {item['id']} | {check(m['trajectory_accuracy'])} | {check(m['sql_execution_safety'])} | {check(m['citation_accuracy'])} | {check(m['answer_factuality'])} | {'PASS' if item['passed'] else 'FAIL'} |")
        if item.get("error"):
            lines.append(f"\nError for `{item['id']}`: {item['error']}\n")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic workflow and trajectory evaluations.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum passing case rate (default: 0.85)")
    parser.add_argument("--trace", action="store_true", help="Opt in to LangSmith traces for this evaluation")
    parser.add_argument("--report-dir", type=Path, default=Path("."), help="Directory for eval_report.json/md")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    settings = get_settings()
    init_observability(settings, enabled=args.trace)
    cases_path = settings.metadata_catalog_path.parent / "golden_queries.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    model = GoldenCaseModel({})
    service = None
    try:
        service = FinchService(settings, llm=model)
        for case in cases:
            try:
                model.case = case
                response = service.ask(AskRequest(question=case["query"], user_id=f"golden-{case['id']}"))
                results.append(_score_case(case, response))
            except Exception as exc:
                results.append({"id": case["id"], "query": case["query"], "passed": False,
                                "metrics": {"trajectory_accuracy": False, "sql_execution_safety": False,
                                            "citation_accuracy": False, "answer_factuality": False},
                                "expected_route": case.get("expected_route", []), "observed_route": [],
                                "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        for case in cases[len(results):]:
            results.append({"id": case["id"], "query": case["query"], "passed": False,
                            "metrics": {"trajectory_accuracy": False, "sql_execution_safety": False,
                                        "citation_accuracy": False, "answer_factuality": False},
                            "expected_route": case.get("expected_route", []), "observed_route": [],
                            "error": f"Workflow setup {type(exc).__name__}: {exc}"})
    finally:
        if service is not None:
            service.close()

    passed = sum(item["passed"] for item in results)
    total = len(results)
    rate = passed / total if total else 0.0
    json_path = args.report_dir / "eval_report.json"
    markdown_path = args.report_dir / "eval_report.md"
    _write_report(results, total, passed, json_path, markdown_path, rate, args.threshold)

    print(f"{'CASE':<14} {'TRAJ':<7} {'SQL':<7} {'CITE':<7} {'FACT':<7} RESULT")
    for item in results:
        m = item["metrics"]
        print(f"{item['id']:<14} {str(m['trajectory_accuracy']):<7} {str(m['sql_execution_safety']):<7} "
              f"{str(m['citation_accuracy']):<7} {str(m['answer_factuality']):<7} "
              f"{'PASS' if item['passed'] else 'FAIL'}")
        if item.get("error"):
            print(f"  {item['error']}")
    print(f"Pass rate: {passed}/{total} ({rate:.1%}); required: {args.threshold:.1%}")
    print(f"Reports: {json_path} and {markdown_path}")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
