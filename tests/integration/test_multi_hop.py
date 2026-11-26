from finch_agent.agents.sql_agent import SQLAgent
from finch_agent.agents.supervisor import SupervisorAgent
from finch_agent.orchestration.state import RouteDecision
from finch_agent.orchestration.state import ToolResult


class PlanningModel:
    def route(self, question, system_prompt=None):
        return RouteDecision(destination="sql_agent")

    def synthesize_report(self, question, passages):
        return "synthesized"

    def generate_sql(self, question, metadata, error=None):
        self.sql_question = question
        return "SELECT 90 AS discounted_price"


class Vectors:
    def search_metadata(self, question):
        return []


class Dialect:
    name = "sqlite"


class Database:
    dialect = Dialect()


class Engine:
    engine = Database()

    def get_schema_context(self):
        return [{"table_name": "products", "columns": [{"name": "price", "type": "INTEGER"}],
                 "sample_rows": [{"price": 100}]}]

    def execute(self, sql):
        return ToolResult(status="success", query=sql, columns=["discounted_price"],
                          rows=[{"discounted_price": 90}], row_count=1)


def test_policy_calculation_is_rag_then_sql_with_context():
    model = PlanningModel()
    supervisor = SupervisorAgent(model)
    state = {
        "question": "What is the price after discount?",
        "effective_question": "What is the price after discount?",
        "step_count": 0,
        "max_iterations": 6,
        "worker_completed": False,
    }
    first = supervisor(state)
    assert first["next_agent"] == "rag_agent"

    after_rag = {
        **state,
        **first,
        "worker_completed": True,
        "last_agent": "rag_agent",
        "rag_context": ["[discount_policy.md p.2] Apply a 10% discount."],
        "citations": [{"document": "discount_policy.md", "page": 2,
                       "excerpt": "Apply a 10% discount."}],
    }
    second = supervisor(after_rag)
    assert second["next_agent"] == "sql_agent"
    assert "10% discount" in second["current_task"]

    after_sql = {**after_rag, **second, "worker_completed": False}
    result = SQLAgent(model, Vectors(), Engine())(after_sql)
    assert "10% discount" in model.sql_question
    assert result["sql_results"] == [{"discounted_price": 90}]
    sql_context = result["metadata"][0]
    assert "10% discount" in sql_context["rag_context"][0]
    contracts = next(item for item in sql_context["catalog_schema"]
                     if item["table_name"] == "niiha_contracts")
    assert "discount_pct DECIMAL(5, 2)" in contracts["columns"]

    final = supervisor({**after_sql, **result})
    assert final["next_agent"] == "FINISH"


def test_discount_audit_forces_sql_after_rag_even_without_calculation_words(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '[{"table_name":"niiha_contracts","columns":["contract_id", "discount_pct"],'
        '"aliases":["deals", "discount leakage"]}]',
        encoding="utf-8",
    )
    model = PlanningModel()
    supervisor = SupervisorAgent(model, catalog_path=catalog_path)
    question = (
        "Audit all active contracts for unauthorized discount leakage and tell me who approved any violations."
    )
    state = {
        "question": question,
        "effective_question": question,
        "step_count": 0,
        "max_iterations": 6,
        "worker_completed": False,
        "sql_attempted": False,
        "execution_trace": [],
    }

    first = supervisor(state)
    assert first["next_agent"] == "rag_agent"
    after_rag = {
        **state,
        **first,
        "worker_completed": True,
        "last_agent": "rag_agent",
        "rag_context": ["Tier 1 cap is 10%; Tier 2 cap is 15%; Tier 3 cap is 20%."],
        "execution_trace": first["execution_trace"] + ["rag_agent -> supervisor"],
    }
    second = supervisor(after_rag)
    assert second["next_agent"] == "sql_agent"
    assert "Tier 1 cap is 10%" in second["current_task"]
    assert "niiha_contracts" in supervisor.system_prompt
    assert "discount leakage" in supervisor.system_prompt

    after_sql = {
        **after_rag,
        **second,
        "worker_completed": True,
        "last_agent": "sql_agent",
        "sql_attempted": True,
        "answer": "Query returned 1 rows.",
        "execution_trace": second["execution_trace"] + ["sql_agent -> supervisor"],
    }
    final = supervisor(after_sql)
    assert final["next_agent"] == "FINISH"
    assert final["execution_trace"][-1] == "supervisor -> FINISH"


def test_all_eight_business_audit_prompts_route_through_rag_then_sql():
    questions = [
        "Audit all active contracts for unauthorized discount leakage and tell me who approved any violations.",
        "Which clients should be restricted from license expansions due to High Credit Risk, and what are their active contract ACVs?",
        "Flag all hotel and client dining expenses in niiha_expenses that violated our 2026 T&E policy caps.",
        "Check which departments exceeded their FY 2026 quarterly budget caps in any quarter.",
        "Are there any recurring SaaS or cloud GPU expenses that require a FinOps Cost-Benefit Review?",
        "Compare our current total overdue receivables (> 60 days) in the database against the $620,000 year-end 2025 figure from the annual report.",
        "Calculate our current Q1/Q2 cash-flow weight penalty for all active contracts on Net-60 or Net-90 payment terms.",
        "Check our EMEA and APAC contracts to see if our current discount leakage rate is below the FY 2026 strategic target.",
    ]

    class NoRouteModel:
        def route(self, question, system_prompt=None):
            raise AssertionError("These policy-dependent audits must route deterministically.")

    supervisor = SupervisorAgent(NoRouteModel())
    for question in questions:
        state = {
            "question": question,
            "step_count": 0,
            "max_iterations": 6,
            "worker_completed": False,
            "sql_attempted": False,
        }
        rag_step = supervisor(state)
        assert rag_step["next_agent"] == "rag_agent", question
        after_rag = {
            **state,
            **rag_step,
            "worker_completed": True,
            "last_agent": "rag_agent",
            "rag_context": ["Retrieved policy or benchmark context."],
        }
        sql_step = supervisor(after_rag)
        assert sql_step["next_agent"] == "sql_agent", question
