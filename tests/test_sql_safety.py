import pytest

from finch_agent.sql_engine import validate_read_only_sql


@pytest.mark.parametrize("statement", [
    "DELETE FROM finance_bookings",
    "WITH target AS (SELECT 1) UPDATE finance_bookings SET gross_bookings = 0",
    "SELECT 1; DROP TABLE finance_bookings",
    "SELECT load_extension('evil')",
    "SELECT nextval('financial_sequence')",
    "SELECT pg_notify('channel', 'message')",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT 1 INTO report_copy",
])
def test_rejects_write_or_dangerous_statements(statement: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(statement)


def test_accepts_single_read_only_select() -> None:
    assert validate_read_only_sql("SELECT COUNT(*) FROM finance_bookings;") == "SELECT COUNT(*) FROM finance_bookings"
