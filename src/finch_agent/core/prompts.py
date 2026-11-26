from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=3)
def load_prompt(name: str) -> str:
    """Load one packaged agent prompt by its stable prompt name."""
    if name not in {"supervisor", "sql_agent", "rag_agent"}:
        raise ValueError(f"Unknown prompt: {name}")
    return files("finch_agent").joinpath("config", "prompts", f"{name}.md").read_text(encoding="utf-8")
