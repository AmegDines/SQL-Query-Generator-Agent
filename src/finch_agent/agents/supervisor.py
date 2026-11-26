from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

from finch_agent.core.prompts import load_prompt
from finch_agent.memory.checkpointer import trim_conversation_history
from finch_agent.orchestration.state import AgentState


class SupervisorAgent:
    """Plans bounded worker chains and synthesizes the accumulated evidence."""

    def __init__(
        self,
        llm: Any,
        max_iterations: int = 6,
        catalog_path: Path | None = None,
    ) -> None:
        self.llm = llm
        self.max_iterations = max_iterations
        self.system_prompt = load_prompt("supervisor") + self._catalog_prompt(catalog_path)

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Select the next worker or synthesize a final grounded response."""
        step = state.get("step_count", 0) + 1
        if step > min(state.get("max_iterations", self.max_iterations), self.max_iterations):
            return {"step_count": step, "next_agent": "FINISH", "answer": "I couldn't complete this request within the workflow limit.",
                    "trajectory": state.get("trajectory", []),
                    "execution_trace": state.get("execution_trace", []) + ["supervisor -> FINISH"]}
        if state.get("worker_completed"):
            force_sql_audit = (
                state.get("last_agent") == "rag_agent"
                and not state.get("sql_attempted", False)
                and self._requires_sql_audit(state["question"])
            )
            if state.get("last_agent") == "rag_agent" and (
                force_sql_audit or self._needs_calculation(state["question"])
            ):
                rules = "\n".join(state.get("rag_context", [])) or "No relevant policy text was retrieved."
                if (not force_sql_audit and not state.get("rag_context")
                        and self._needs_policy_lookup(state["question"])):
                    return {"step_count": step, "next_agent": "FINISH", "worker_completed": False,
                            "answer": "I couldn't find the policy rule needed to calculate this reliably.",
                            "trajectory": state.get("trajectory", []),
                            "execution_trace": state.get("execution_trace", []) + ["supervisor -> FINISH"]}
                task = (f"User question: {state['question']}\n\nPolicy rules retrieved by rag_agent:\n{rules}\n\n"
                        "Use these exact rules as constraints. Query the approved SQLite data for the requested records. "
                        "For an audit, return the relevant record identifiers and approver fields. "
                        "If a required rule or input is missing, do not guess; return that limitation.")
                return {"step_count": step, "next_agent": "sql_agent", "current_task": task,
                        "route": "sql_agent",
                        "sub_task_instructions": {"sql_agent": load_prompt("sql_agent")},
                        "worker_completed": False,
                        "trajectory": state.get("trajectory", []) + ["sql_agent"],
                        "execution_trace": state.get("execution_trace", []) + ["supervisor -> sql_agent"]}
            if state.get("last_agent") == "rag_agent":
                answer = self.llm.synthesize_report(state.get("effective_question", state["question"]),
                                                    state.get("citations", [])) if state.get("citations") else "I couldn't find relevant evidence in the configured reports or policies."
            else:
                answer = state.get("answer", "The SQL agent returned no answer.")
            messages = trim_conversation_history(state.get("messages", []) + [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ], max_turns=6)
            return {"step_count": step, "next_agent": "FINISH", "answer": answer,
                    "worker_completed": False, "messages": messages,
                    "trajectory": state.get("trajectory", []),
                    "execution_trace": state.get("execution_trace", []) + ["supervisor -> FINISH"]}

        history = state.get("messages", [])[-12:]
        question = state["question"]
        effective = ("Conversation so far:\n" + "\n".join(f"{x['role']}: {x['content']}" for x in history)
                     + "\n\nCurrent question:\n" + question) if history else question
        if state.get("user_context"):
            effective += f"\n\nSaved user preferences (use only when relevant): {state['user_context']}"
        policy_first = self._needs_policy_lookup(question)
        if policy_first:
            target = "rag_agent"
        elif self._requires_sql_audit(question):
            target = "sql_agent"
        else:
            decision = self.llm.route(effective, system_prompt=self.system_prompt)
            target = "rag_agent" if decision.destination in {"rag_agent", "doc_rag_agent"} else decision.destination
        if self._is_policy_followup(question, history):
            target = "sql_agent"
        if target not in {"sql_agent", "rag_agent"}:
            answer = "Could you clarify whether you need a database result or an answer from a report or policy?"
            messages = trim_conversation_history(history + [
                {"role": "user", "content": question}, {"role": "assistant", "content": answer}
            ], max_turns=6)
            return {"step_count": step, "next_agent": "FINISH", "route": "clarification",
                    "effective_question": effective,
                    "answer": answer, "messages": messages,
                    "trajectory": state.get("trajectory", []),
                    "execution_trace": state.get("execution_trace", []) + ["supervisor -> FINISH"]}
        task = effective
        return {"step_count": step, "next_agent": target, "route": target, "current_task": task,
                "effective_question": effective,
                "sub_task_instructions": {target: load_prompt("sql_agent" if target == "sql_agent" else "rag_agent")},
                "trajectory": state.get("trajectory", []) + [target],
                "execution_trace": state.get("execution_trace", []) + [f"supervisor -> {target}"]}

    @staticmethod
    def _catalog_prompt(catalog_path: Path | None) -> str:
        """Render catalog table names, columns, and aliases into the routing prompt."""
        if catalog_path is None or not catalog_path.is_file():
            return "\n\nSQLite catalog context: niiha_contracts, niiha_invoices, niiha_expenses contain governed business records."
        try:
            entries = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
        lines = ["\n\nSQLite catalog context (these are queryable business records):"]
        for item in entries:
            table = item.get("table_name")
            if not table:
                continue
            columns = ", ".join(str(column) for column in item.get("columns", []))
            aliases = ", ".join(str(alias) for alias in item.get("aliases", []))
            lines.append(f"- {table}: columns [{columns}]; aliases [{aliases}].")
        return "\n".join(lines)

    @staticmethod
    def _requires_sql_audit(question: str) -> bool:
        """Detect record-oriented questions that require a SQL check after policy retrieval."""
        record_terms = (
            r"\b(contract|contracts|cnt-\w+|deal|deals|invoice|invoices|expense|expenses|client|"
            r"department|revenue|receivable|receivables|hotel|dining|SaaS|GPU|budget|cash[- ]flow)\b"
        )
        audit_terms = r"\b(audit|audits|audit(?:ing)?|leakage|unauthorized|active|approved|approval)\b"
        return bool(re.search(record_terms, question, re.I) or re.search(audit_terms, question, re.I))

    @staticmethod
    def _needs_policy_lookup(question: str) -> bool:
        return bool(re.search(
            r"\b(discount|discounted|pricing tier|price after|rebate|markup|policy|sop|"
            r"eligib(?:le|ility)|according to (?:the )?(?:policy|rules)|credit risk|"
            r"license expansions?|hotel|client dining|T&E|travel|procurement|budget caps?|"
            r"quarterly budget|finops|cost-benefit|SaaS|cloud GPU|cash[- ]flow|Net-(?:60|90)|"
            r"annual report|year[- ]end|strategic targets?|leakage rate|receivables?)\b",
            question,
            re.IGNORECASE,
        ))

    @staticmethod
    def _needs_calculation(question: str) -> bool:
        return bool(re.search(r"\b(calculate|compute|how much|how many|total|amount|price|cost|sum|average|count|compare|violate|flag\w*|list|identify|exceed|require|requires)\b", question, re.I))

    @staticmethod
    def _is_policy_followup(question: str, messages: list[dict[str, str]]) -> bool:
        followup = re.search(r"\b(what about|apply that|and for|how about|for \d+ seats)\b", question, re.I)
        if not followup or len(messages) < 2:
            return False
        previous = " ".join(item["content"] for item in messages[-4:])
        return bool(re.search(r"\b(discount|pricing|price|policy)\b", previous, re.I)
                    and re.search(r"\b\d+\s*%", previous))
