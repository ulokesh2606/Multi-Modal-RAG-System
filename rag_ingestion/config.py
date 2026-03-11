import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o")

BASE_DIR      = Path(__file__).parent.resolve()
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "db" / "chroma")))
RAW_DIR       = Path(os.getenv("RAW_DIR",        str(BASE_DIR / "storage" / "raw_chunks")))
REPORT_DIR    = Path(os.getenv("REPORT_DIR",     str(BASE_DIR / "storage" / "ingestion_reports")))
REGISTRY_PATH = Path(os.getenv("REGISTRY_PATH",  str(BASE_DIR / "storage" / "document_registry.json")))

MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", 100))
MIN_ALPHA_RATIO  = float(os.getenv("MIN_ALPHA_RATIO", 0.5))

# ── Sliding-window chunking for flat text files (txt, md, csv) ────────────────
# Plain text has no Title elements so chunk_by_title() produces one giant chunk.
# Instead we split on paragraph/sentence boundaries within a token budget.
# TXT_CHUNK_SIZE   : target chunk size in characters (~400 chars ≈ 100 tokens)
# TXT_CHUNK_OVERLAP: overlap between consecutive chunks so context isn't lost
#                    at boundaries (typically 15-20% of chunk size)
TXT_CHUNK_SIZE    = int(os.getenv("TXT_CHUNK_SIZE",    "1200"))
TXT_CHUNK_OVERLAP = int(os.getenv("TXT_CHUNK_OVERLAP", "200"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def init_storage():
    for d in [VECTOR_DB_DIR, RAW_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text('{"documents": {}, "chunks": {}}')

