import hashlib
from service_config import RETRIEVAL_VERSION, INGESTION_VERSION


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def exact_cache_key(tenant_id: str, normalized_query: str) -> str:
    base = f"{tenant_id}:{normalized_query}:{RETRIEVAL_VERSION}:{INGESTION_VERSION}"
    return f"rag:exact:{hash_text(base)}"


def semantic_cache_prefix(tenant_id: str) -> str:
    return f"rag:semantic:{tenant_id}:"
