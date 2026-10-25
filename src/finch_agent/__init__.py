"""Governed conversational financial data assistant."""

import os

# Checkpoint payloads may contain untrusted conversation data; restrict msgpack types.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

__version__ = "1.0.0"
