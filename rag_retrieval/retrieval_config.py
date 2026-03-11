"""
retrieval_config.py — Updated thresholds for sigmoid-normalized scoring

CHANGES:
  CONFIDENCE_HIGH_TOP_SCORE:    5.0 → 3.0   (unchanged — raw logit space)
  CONFIDENCE_MEDIUM_TOP_SCORE:  0.8          (unchanged)
  ESCALATION_THRESHOLD:         0.25         (NEW — sigmoid space, in answer_generation.py)

The raw logit thresholds are still used for the confidence LABEL (high/medium/low)
in analyze_scores(). The sigmoid normalization is a separate layer added on top.
This means you don't need to change retrieval_config.py values — existing behaviour
for confidence labelling is preserved. Only the escalation decision uses sigmoid.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL   = os.getenv("GEN_MODEL", "gpt-4o")

_THIS_DIR      = Path(__file__).parent.resolve()
VECTOR_DB_DIR  = Path(os.getenv("VECTOR_DB_DIR", str(_THIS_DIR.parent / "rag_ingestion" / "db" / "chroma")))
RAW_DIR        = Path(os.getenv("RAW_DIR",        str(_THIS_DIR.parent / "rag_ingestion" / "storage" / "raw_chunks")))

# ── Confidence label thresholds (raw cross-encoder logit space) ───────────────
# These are used by analyze_scores() to assign "high" / "medium" / "low" labels.
# ms-marco-MiniLM-L-6-v2 typical range: -6 to +8
CONFIDENCE_HIGH_TOP_SCORE   = float(os.getenv("CONFIDENCE_HIGH_TOP_SCORE",   "3.0"))
CONFIDENCE_HIGH_AVG_SCORE   = float(os.getenv("CONFIDENCE_HIGH_AVG_SCORE",   "1.5"))
CONFIDENCE_HIGH_MAX_SPREAD  = float(os.getenv("CONFIDENCE_HIGH_MAX_SPREAD",  "3.0"))
CONFIDENCE_MEDIUM_TOP_SCORE = float(os.getenv("CONFIDENCE_MEDIUM_TOP_SCORE", "0.8"))

# ── Retrieval sizes ───────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "40"))
RERANKER_TOP_N  = int(os.getenv("RERANKER_TOP_N", "5"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rag.retrieval")
