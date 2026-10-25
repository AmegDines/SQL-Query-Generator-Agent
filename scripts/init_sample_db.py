from pathlib import Path
import sqlite3


root = Path(__file__).resolve().parents[1]
database = root / "data" / "finch.db"
schema = root / "data" / "sample_data.sql"
database.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(database) as connection:
    connection.executescript(schema.read_text(encoding="utf-8"))
print(f"Initialized demonstration database at {database}")
