from finch_agent.integrations.slack import slack_blocks
from finch_agent.orchestration.state import AskResponse


def test_sql_results_render_as_block_kit_table():
    response = AskResponse(route="sql_agent", answer="Query returned 1 row(s).", columns=["total"],
                           rows=[[42]], sql="SELECT 42", request_id="request-id")
    blocks = slack_blocks(response)
    table = next(block for block in blocks if block["type"] == "table")
    assert table["rows"][0][0] == {"type": "raw_text", "text": "total"}
    assert table["rows"][1][0] == {"type": "raw_text", "text": "42"}
    assert any("SELECT 42" in block.get("text", {}).get("text", "") for block in blocks)
