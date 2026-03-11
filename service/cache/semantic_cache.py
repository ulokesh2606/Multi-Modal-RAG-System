import json
import logging
import numpy as np
from cache.redis_client import redis_client
from cache.keys import semantic_cache_prefix
from service_config import SEMANTIC_SIM_THRESHOLD, SEMANTIC_CACHE_TTL

logger = logging.getLogger("service.semantic_cache")


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def find_semantic_hit(tenant_id: str, query_embedding: list):
    prefix = semantic_cache_prefix(tenant_id)
    for key in redis_client.scan_iter(f"{prefix}*", count=100):
        try:
            raw = redis_client.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            stored_emb = entry.pop("embedding", None)
            if stored_emb is None:
                continue
            score = cosine(query_embedding, stored_emb)
            if score >= SEMANTIC_SIM_THRESHOLD:
                entry["semantic_score"] = round(score, 3)
                return entry
        except Exception as e:
            logger.warning(f"Semantic cache error {key}: {e}")
    return None


def store_semantic(tenant_id: str, key_hash: str, payload: dict):
    redis_client.setex(
        f"{semantic_cache_prefix(tenant_id)}{key_hash}",
        SEMANTIC_CACHE_TTL,
        json.dumps(payload)
    )
