import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from config import REGISTRY_PATH

logger = logging.getLogger("ingestion.registry")


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_registry() -> dict:
    try:
        return json.loads(Path(REGISTRY_PATH).read_text())
    except FileNotFoundError:
        return {"documents": {}, "chunks": {}}


def save_registry(registry: dict):
    Path(REGISTRY_PATH).write_text(json.dumps(registry, indent=2))


def register_document(registry: dict, doc_hash: str, filename: str):
    if doc_hash in registry["documents"]:
        logger.info(f"Already registered: {filename}")
        return None

    document_id = hashlib.sha256(doc_hash.encode()).hexdigest()
    registry["documents"][doc_hash] = {
        "document_id": document_id,
        "file": filename,
        "version": 1,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }
    logger.info(f"Registered: {filename} → {document_id[:12]}...")
    return registry["documents"][doc_hash]


def is_chunk_duplicate(registry: dict, c_hash: str) -> bool:
    return c_hash in registry["chunks"]


def register_chunk(registry: dict, c_hash: str, meta: dict):
    registry["chunks"][c_hash] = meta


