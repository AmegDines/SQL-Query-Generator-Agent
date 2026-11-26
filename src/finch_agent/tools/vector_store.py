from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb
import structlog

from finch_agent.config.settings import Settings
from finch_agent.core.observability import trace_step
from finch_agent.core.guardrails import sanitize_retrieved_docs

log = structlog.get_logger(__name__)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class VectorStore:
    """Chroma-backed schema, alias, golden-query, and report passage search."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.chroma_host:
            headers = None
            if settings.chroma_api_key:
                headers = {"X-Chroma-Token": settings.chroma_api_key.get_secret_value()}
            kwargs: dict[str, Any] = {
                "host": settings.chroma_host,
                "port": settings.chroma_port,
                "ssl": settings.chroma_ssl,
            }
            if headers:
                kwargs["headers"] = headers
            if settings.chroma_tenant:
                kwargs["tenant"] = settings.chroma_tenant
            if settings.chroma_database:
                kwargs["database"] = settings.chroma_database
            self.client = chromadb.HttpClient(**kwargs)
        else:
            settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(settings.chroma_path))

        self.schema = self.client.get_or_create_collection("schema_catalog")
        self.aliases = self.client.get_or_create_collection("business_aliases")
        self.golden = self.client.get_or_create_collection("golden_sql_queries")
        self.reports = self.client.get_or_create_collection("report_pages")
        self.sync_catalog()
        self.sync_reports()

    @staticmethod
    def _replace_source(collection: Any, source: str, records: list[tuple[str, str, dict[str, Any]]]) -> None:
        existing = collection.get(where={"source": source}, include=["metadatas", "documents"])
        current_ids = existing.get("ids", [])
        old_documents = dict(zip(current_ids, existing.get("documents", [])))
        current_source_ids = {record[0] for record in records}
        stale = [record_id for record_id in current_ids if record_id not in current_source_ids]
        if stale:
            collection.delete(ids=stale)
        changed = [record for record in records if old_documents.get(record[0]) != record[1]]
        if changed:
            collection.upsert(
                ids=[record[0] for record in changed],
                documents=[record[1] for record in changed],
                metadatas=[record[2] for record in changed],
            )

    def sync_catalog(self) -> None:
        catalog_path = self.settings.metadata_catalog_path
        entries = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
        schema_records: list[tuple[str, str, dict[str, Any]]] = []
        alias_records: list[tuple[str, str, dict[str, Any]]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            table_name = str(entry.get("table_name", f"table_{index}"))
            description = str(entry.get("description", ""))
            columns = entry.get("columns", [])
            if isinstance(columns, list):
                col_text = ", ".join(str(column) for column in columns)
            else:
                col_text = str(columns)
            schema_text = f"Table: {table_name}. Domain: {entry.get('domain', '')}. Description: {description}. Columns: {col_text}."
            schema_meta = {
                "source": "catalog", "table_name": table_name,
                "domain": str(entry.get("domain", "unknown")),
                "access_level": str(entry.get("access_level", "read_only")),
            }
            schema_records.append((_stable_id("schema", table_name), schema_text, schema_meta))
            aliases = entry.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias = str(alias).strip()
                    if alias:
                        alias_text = f"Business alias: {alias}. Target table: {table_name}. Description: {description}. Columns: {col_text}."
                        alias_records.append((_stable_id("alias", table_name, alias.lower()), alias_text,
                                             {"source": "catalog", "table_name": table_name, "alias": alias}))
        self._replace_source(self.schema, "catalog", schema_records)
        self._replace_source(self.aliases, "catalog", alias_records)

        golden_path = self.settings.metadata_catalog_path.parent / "golden_queries.json"
        golden_entries = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.exists() else []
        golden_records = []
        for index, item in enumerate(golden_entries):
            question = item.get("question", item.get("query")) if isinstance(item, dict) else None
            sql = item.get("sql", item.get("reference_sql")) if isinstance(item, dict) else None
            if not isinstance(item, dict) or not question or not sql:
                continue
            record_id = str(item.get("id") or f"golden-{index}")
            record_text = f"Question: {question}\nSQL: {sql}"
            golden_records.append((_stable_id("golden", record_id), record_text,
                                  {"source": "golden", "golden_id": record_id,
                                   "sql": str(sql), "author": str(item.get("author", "curated"))}))
        self._replace_source(self.golden, "golden", golden_records)

    def _report_chunks(self, path: Path) -> list[tuple[int, str]]:
        if path.suffix.lower() in {".txt", ".md"}:
            pages = path.read_text(encoding="utf-8", errors="replace").split("\f")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install the documents extra to index PDF reports") from exc
            pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        else:
            return []
        output = []
        chunk_size = self.settings.report_chunk_size
        for page, content in enumerate(pages, 1):
            text = re.sub(r"\s+", " ", content).strip()
            for offset in range(0, len(text), chunk_size - 150):
                chunk = text[offset:offset + chunk_size]
                if chunk:
                    output.append((page, chunk))
        return output

    def sync_reports(self) -> None:
        directory = self.settings.reports_directory
        records: list[tuple[str, str, dict[str, Any]]] = []
        max_bytes = self.settings.report_max_file_mb * 1024 * 1024
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                    continue
                if path.stat().st_size > max_bytes:
                    log.warning("report_skipped_size_limit", report=path.name)
                    continue
                relative = path.relative_to(directory).as_posix()
                try:
                    for page, chunk in self._report_chunks(path):
                        record_id = _stable_id("report", relative, str(page), chunk)
                        records.append((record_id, chunk, {
                            "source": "report", "document": relative, "page": page,
                            "section": self._section_name(chunk, path.stem),
                        }))
                except (OSError, ValueError, RuntimeError):
                    log.exception("report_index_failed", report=relative)
        self._replace_source(self.reports, "report", records)

    @trace_step(run_type="retriever")
    def search_metadata(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        count = limit or self.settings.chroma_top_k
        found: dict[tuple[str, str], dict[str, Any]] = {}
        for collection in (self.schema, self.aliases, self.golden):
            if collection.count() == 0:
                continue
            results = collection.query(query_texts=[query], n_results=min(count, collection.count()),
                                       include=["documents", "metadatas", "distances"])
            for document, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                value = dict(metadata or {})
                value.update({"content": document, "distance": float(distance)})
                key = (str(value.get("table_name", "")), str(value.get("alias", value.get("golden_id", ""))))
                found[key] = value
        return sorted(found.values(), key=lambda item: item["distance"])[:count * 2]

    @trace_step(run_type="retriever")
    def search_reports(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Return relevant report chunks with source and section citations."""
        count = limit or self.settings.chroma_top_k
        if self.reports.count() == 0:
            return []
        results = self.reports.query(query_texts=[query], n_results=min(count, self.reports.count()),
                                     include=["documents", "metadatas", "distances"])
        citations: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            metadata = metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
            if score < self.settings.min_relevance_score:
                continue
            source = str(metadata.get("document", "unknown"))
            content = sanitize_retrieved_docs(str(document), source)
            citations.append({"chunk_id": str(chunk_id), "source_file": source,
                              "section": str(metadata.get("section", Path(source).stem)),
                              "score": score, "content": content, "document": source,
                              "page": int(metadata.get("page", 1)), "excerpt": content})
        return citations

    @staticmethod
    def _section_name(chunk: str, fallback: str) -> str:
        heading = re.search(r"^#{1,6}\s+(.+)$", chunk, re.M)
        return heading.group(1).strip() if heading else fallback
