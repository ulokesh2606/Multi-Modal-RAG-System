import json
import time
import uuid
import logging
import asyncio
import os
from pathlib import Path
from langchain_core.documents import Document

from config import RAW_DIR, REPORT_DIR, init_storage
from registry import (
    load_registry, save_registry, file_hash, chunk_hash,
    register_document, register_chunk, is_chunk_duplicate
)
from partitioning import partition_document
from chunking import chunk_by_title, chunk_by_sliding_window, extract_original, validate_chunk
from summarization import Summarizer
from reporting import start_report, finalize_report
from vectorstore import VectorStore

logger = logging.getLogger("ingestion.pipeline")

# ── Optional Redis for SSE progress events ─────────────────────────
_redis_sync = None
try:
    import redis as _redis_lib
    _rurl = (
        f"redis://:{os.getenv('REDIS_PASSWORD', '')}@"
        f"{os.getenv('REDIS_HOST', 'localhost')}:"
        f"{os.getenv('REDIS_PORT', '6379')}"
    )
    _redis_sync = _redis_lib.Redis.from_url(_rurl, decode_responses=True, socket_timeout=2)
    _redis_sync.ping()
    logger.info("Pipeline Redis connected — SSE events will be emitted.")
except Exception as e:
    logger.info(f"Pipeline Redis not available — SSE disabled. ({e})")
    _redis_sync = None


def emit(stage: str, payload: dict, job_id: str = None):
    logger.info(f"[{stage.upper()}] {json.dumps(payload)}")
    if _redis_sync and job_id:
        try:
            evt = json.dumps({"stage": stage, **payload})
            _redis_sync.rpush(f"ingest_progress:{job_id}", evt)
            _redis_sync.expire(f"ingest_progress:{job_id}", 600)
        except Exception:
            pass


class RAGIngestionPipeline:
    def __init__(self):
        init_storage()
        self.summarizer = Summarizer()

    def ingest(self, file_path: str, tenant_id: str = "default",
               original_filename: str = None) -> dict:
        return asyncio.run(self.ingest_async(file_path, tenant_id, original_filename))

    async def ingest_async(self, file_path: str, tenant_id: str = "default",
                           original_filename: str = None) -> dict:
        start_time = time.time()
        job_id     = str(uuid.uuid4())[:12]
        registry   = load_registry()

        doc_hash = file_hash(file_path)

        # FIX: use original_filename if provided so registry shows real name
        display_name = original_filename or Path(file_path).name
        doc_meta     = register_document(registry, doc_hash, display_name)

        if not doc_meta:
            emit("dedup", {"message": "Already ingested", "file": display_name}, job_id)
            return {"status": "duplicate", "file": display_name, "job_id": job_id}

        document_id = doc_meta["document_id"]
        report      = start_report(document_id, display_name)
        emit("upload", {"status": "complete", "file": display_name}, job_id)

        # ── Partition ──────────────────────────────────────────────
        try:
            elements, stats = partition_document(file_path)
        except Exception as e:
            logger.error(f"Partitioning failed: {e}", exc_info=True)
            emit("error", {"reason": str(e)}, job_id)
            return {"status": "error", "reason": str(e)}

        tables_found = stats.get("Table", 0)
        images_found = stats.get("Image", 0) + stats.get("Figure", 0)

        report["stats"]["atomic_elements"]  = len(elements)
        report["stats"]["tables_detected"]  = tables_found
        report["stats"]["images_detected"]  = images_found
        emit("partition", {
            "total_elements": len(elements),
            "tables":         tables_found,
            "images":         images_found,
            "categories":     stats,
        }, job_id)

        # ── Chunk ──────────────────────────────────────────────────
        # Flat text formats (txt, md, csv) have no Title elements — partition_text()
        # returns only NarrativeText, so chunk_by_title() produces one giant blob.
        # Use a sliding-window paragraph/sentence splitter for these formats instead.
        _flat_text_exts = {".txt", ".md", ".csv", ".xlsx"}
        _file_ext = os.path.splitext(file_path)[1].lower()

        if _file_ext in _flat_text_exts:
            # Reassemble full text from partitioned elements and re-chunk sensibly
            full_text = "\n\n".join(str(el) for el in elements if str(el).strip())
            raw_chunks = chunk_by_sliding_window(full_text)
            # Wrap in a list-of-list shape so the rest of the loop works unchanged:
            # each "chunk" here is already an extract_original() dict, so we pass
            # it through directly and skip the extract_original() call below.
            chunks = raw_chunks          # list[dict]
            _preextracted = True
        else:
            chunks = chunk_by_title(elements)
            _preextracted = False

        total_chunks = len(chunks)
        report["stats"]["chunks_created"] = total_chunks
        emit("chunk", {"total_chunks": total_chunks}, job_id)

        # ── Summarise + Embed ──────────────────────────────────────
        docs         = []
        summary_done = 0
        tables_embedded = 0
        images_embedded = 0

        for idx, chunk in enumerate(chunks, start=1):
            # For flat-text files chunk_by_sliding_window() already returns
            # extract_original()-compatible dicts; skip the extraction step.
            if _preextracted:
                original = chunk
            else:
                original = extract_original(chunk)
            if not validate_chunk(original):
                continue

            c_hash = chunk_hash(original["text"])
            if is_chunk_duplicate(registry, c_hash):
                report["stats"]["chunks_deduped"] += 1
                continue

            try:
                summary = await self.summarizer.summarize_async(original)
            except Exception as e:
                logger.warning(f"Summarization failed chunk {idx}: {e}")
                summary = original["text"]

            summary_done += 1
            if summary_done % 3 == 0:
                emit("summarise", {"current": summary_done, "total": total_chunks}, job_id)

            chunk_id = str(uuid.uuid4())
            raw_path = RAW_DIR / f"{chunk_id}.json"

            # ── FIX: use raw element counts for accurate meta flags ──
            # Original code: bool(original["tables_html"]) — False if HTML missing
            # Fixed code:    table_count > 0 — True whenever a Table element existed
            has_tables = original.get("table_count", 0) > 0
            has_images = original.get("image_count", 0) > 0

            if has_tables: tables_embedded += 1
            if has_images: images_embedded += 1

            chunk_payload = {
                "chunk_id":    chunk_id,
                "document_id": document_id,
                "tenant_id":   tenant_id,
                "original":    original,
                "summary":     summary,
                "meta": {
                    "chunk_index": idx,
                    "has_tables":  has_tables,
                    "has_images":  has_images,
                    # Store counts for diagnostics
                    "table_count": original.get("table_count", 0),
                    "image_count": original.get("image_count", 0),
                    "tables_with_html":   len(original.get("tables_html", [])),
                    "images_with_base64": len(original.get("images_base64", [])),
                }
            }
            raw_path.write_text(json.dumps(chunk_payload, indent=2))

            docs.append(Document(
                page_content=summary,
                metadata={
                    "document_id": document_id,
                    "chunk_id":    chunk_id,
                    "tenant_id":   tenant_id,
                    "raw_path":    str(raw_path)
                }
            ))
            register_chunk(registry, c_hash, {
                "chunk_id":    chunk_id,
                "document_id": document_id,
                "tenant_id":   tenant_id,
                "raw_path":    str(raw_path)
            })
            report["stats"]["chunks_embedded"] = summary_done

        emit("embed", {"count": summary_done}, job_id)

        vs = VectorStore(tenant_id=tenant_id)
        vs.add(docs)
        emit("vectorstore", {
            "count":           summary_done,
            "collection":      f"tenant_{tenant_id}",
            "tables_embedded": tables_embedded,
            "images_embedded": images_embedded,
        }, job_id)

        save_registry(registry)
        finalize_report(report, time.time() - start_time)
        (REPORT_DIR / f"{document_id}.json").write_text(json.dumps(report, indent=2))

        final = {
            "status":          "success",
            "job_id":          job_id,
            "filename":        display_name,
            "tables_embedded": tables_embedded,
            "images_embedded": images_embedded,
            **report["stats"]
        }
        emit("done", final, job_id)

        logger.info(
            f"Ingestion complete: {summary_done} chunks | "
            f"{tables_embedded} table chunks | "
            f"{images_embedded} image chunks | "
            f"{time.time() - start_time:.1f}s"
        )
        return final
