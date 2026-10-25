# Finch Agent

Finch is a Slack-first financial data assistant implemented against the supplied PRD, HLD, and LLD. A LangGraph supervisor routes questions into specialized SQL and report workers. ChromaDB stores and retrieves schema descriptions, business aliases, curated SQL examples, and cited report chunks. SQL execution runs through a strictly read-only SQLAlchemy connection.

## Implemented capabilities

- Slack Bolt Events API at `/slack/events`; request signature verification, direct messages and explicit channel mentions, thread replies, channel allowlists, per-user rate limits, and bounded conversation history.
- FastAPI `POST /v1/ask`, liveness/readiness routes, production bearer-token authentication, and structured response lineage.
- LangGraph supervisor, SQL writer/executor with up to three correction retries, report retriever/synthesizer, clarification path, and persisted thread state.
- Chroma collections `schema_catalog`, `business_aliases`, `golden_sql_queries`, and `report_pages`; startup sync is idempotent and indexes `.txt`, `.md`, and text-based `.pdf` pages.
- SQL statement validation plus read-only SQLite URI / PostgreSQL transaction, query timeout, result row cap, and execution SQL in responses.
- LangSmith tracing configuration, golden-query evaluation data and runner, unit and recovery tests, Docker Compose, and Slack app manifest.

## Local setup

Requires Python 3.12 or later and Docker Compose for the Chroma server.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
Copy-Item .env.example .env
python scripts/init_sample_db.py
docker compose up -d chroma
```

Configure `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`, and `DATABASE_URL` for live answers. The `mock` provider can route basic questions but intentionally cannot generate SQL or synthesize report answers. Set Slack credentials to enable Slack. Add approved `.txt`, `.md`, and `.pdf` sources under `data/reports/`; Finch indexes them at startup.

The sample schema and records are illustrative demonstration data, not company financial data. Replace `data/catalog.json` with the approved production schema and aliases before asking company questions. Add human-reviewed question/SQL pairs to `data/golden_queries.json`.

```powershell
uvicorn finch_agent.main:app --reload
```

## Production deployment

Set `APP_ENV=production`, `DATABASE_URL` using the psycopg driver, `API_AUTH_TOKEN`, Slack credentials, an OpenAI key, a remote `CHROMA_HOST` plus `CHROMA_API_KEY`, `CHROMA_SSL=true`, and `CHECKPOINT_DATABASE_URL`. Production startup requires PostgreSQL for reporting and LangGraph checkpoint state, a model provider capable of generation, and server-backed TLS Chroma. Keep these services on private networks and restrict `/v1/ask` ingress. Use separate credentials: the reporting database account is SELECT-only; the checkpoint account can write only to its dedicated checkpoint database/schema. Do not reuse the read-only reporting credential for checkpoints.

For local Docker Compose, the Chroma service is private to the Compose network. The app and Chroma persist their data under `data/` and the `chroma-data` volume. For production use a managed/server Chroma deployment and PostgreSQL checkpointer with backups and retention configured. SQLite checkpoint storage is development-only and does not meet the 50-session production target.

## Configuration

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL for the approved reporting database |
| `DATABASE_READ_ONLY` | Must be `true`; startup validation rejects `false` |
| `QUERY_TIMEOUT_SECONDS`, `QUERY_MAX_ROWS` | SQL execution limits |
| `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW`, `WORKER_THREADS` | Database and request concurrency settings |
| `METADATA_CATALOG_PATH` | JSON catalog with table, columns, descriptions, and aliases |
| `REPORTS_DIRECTORY`, `REPORT_CHUNK_SIZE` | Report corpus and page chunking settings |
| `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`, `CHROMA_API_KEY` | Remote Chroma configuration; blank host selects local persistent Chroma |
| `CHECKPOINT_DATABASE_URL` | PostgreSQL connection for persistent production graph state; blank selects local SQLite |
| `SLACK_ALLOWED_CHANNELS` | Optional comma-separated Slack channel IDs |
| `API_AUTH_TOKEN` | Required production bearer token for `/v1/ask` |
| `RATE_LIMIT_PER_MINUTE` | Per-user rolling-minute request limit |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | LangSmith trace export configuration |

Use `postgresql+psycopg://...` for PostgreSQL reporting databases. Install optional extras with `pip install -e ".[postgres]"` when deploying PostgreSQL support without the full extra set.

## Slack setup

Create a Slack app from `slack-manifest.json`, set the public HTTPS request URL, install the app, and configure its bot token and signing secret. The app listens for direct messages and explicit mentions; it replies in the source thread. Add the intended channel IDs to `SLACK_ALLOWED_CHANNELS` where an allowlist is required.

## API and responses

`POST /v1/ask` accepts `{ "question": "...", "user_id": "...", "channel_id": "...", "thread_ts": "..." }`. In production send `Authorization: Bearer <API_AUTH_TOKEN>`. SQL responses include executed SQL, columns, bounded rows, and retry count. Report responses include citations with document and page number. Health routes are `/health/live` and `/health/ready`.

## Reindexing and evaluation

Chroma syncs changed and removed source records when the service starts. To run the golden evaluation against the configured reporting database and LLM:

```powershell
python scripts/evaluate_golden.py
```

Run an optional RAGAS faithfulness and answer-relevance score for a report question after installing the evaluator extra with `python -m pip install -e ".[evaluation]"`:

```powershell
python scripts/evaluate_ragas.py "What does the demonstration data access policy say?"
```

Run automated regression tests with:

```powershell
python -m pytest
```

The data file under `data/golden_queries.json` contains baseline routing, SQL-result, and citation expectations. Expand it with reviewed representative questions for the actual organization; the target first-pass SQL execution accuracy is at least 92%. The supplied demo has only one SQL benchmark, so it is an executable example, not evidence that the production target has been met.

## Operational notes

Chroma's embedded persistent client is for local development; production uses server-backed Chroma. Scanned PDFs need OCR before indexing. The in-process rate limiter is shared by workers only within one process; deploy a gateway or shared limiter when using multiple replicas. Checkpoints contain conversation data, so enforce database access controls, backups, and retention at the PostgreSQL layer. SQL safeguards are defense in depth; database permissions remain the security boundary. The supplied folder does not contain real database schemas, report content, or credentials, so production configuration and corpus onboarding remain deployment tasks.
