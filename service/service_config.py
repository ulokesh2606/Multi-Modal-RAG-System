import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RETRIEVAL_VERSION = "v1.0"
INGESTION_VERSION = "v1.0"

EXACT_CACHE_TTL       = int(os.getenv("EXACT_CACHE_TTL", 86400))
SEMANTIC_CACHE_TTL    = int(os.getenv("SEMANTIC_CACHE_TTL", 172800))
SEMANTIC_SIM_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", 0.75))

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o")
