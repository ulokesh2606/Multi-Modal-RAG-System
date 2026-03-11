import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

_WSL_ROOT = Path(__file__).parent.parent.parent  # ~/wsl/
sys.path.insert(0, str(_WSL_ROOT / "rag_retrieval"))

from query_normalization import normalize_query
from cache.keys import exact_cache_key, hash_text
from cache.exact_cache import get_exact, set_exact
from cache.semantic_cache import find_semantic_hit, store_semantic
from security.tenant_validation import validate_tenant
from service_config import EMBED_MODEL

logger = logging.getLogger("service.rag_service")

_embedder     = None
_rag_pipeline = None


def get_pipeline():
    """Expose pipeline for cache invalidation after ingestion."""
    return _rag_pipeline


async def init_service():
    global _embedder, _rag_pipeline
    from retrieval_pipeline import RAGRetrievalPipeline
    _embedder     = OpenAIEmbeddings(model=EMBED_MODEL)
    _rag_pipeline = RAGRetrievalPipeline()
    logger.info("RAG service initialized.")


async def close_service():
    logger.info("RAG service shut down.")


def _extract_current_question(query: str) -> str:

    if "[Current question]" in query:
        return query.split("[Current question]")[-1].strip()
    return query.strip()


async def handle_query(tenant_id: str, query: str) -> dict:
    tenant_id = validate_tenant(tenant_id)

    current_question = _extract_current_question(query)
    normalized       = normalize_query(current_question)

    # 1. Exact cache
    exact_key = exact_cache_key(tenant_id, normalized)
    cached = get_exact(exact_key)
    if cached:
        logger.info(f"Exact cache HIT: {normalized!r}")
        cached["cache"] = "exact"
        return cached

    # 2. Semantic cache — embed the CURRENT QUESTION ONLY for accurate similarity
    q_emb = _embedder.embed_query(normalized)
    semantic_hit = find_semantic_hit(tenant_id, q_emb)
    if semantic_hit:
        logger.info(f"Semantic cache HIT: {normalized!r}")
        semantic_hit["cache"] = "semantic"
        return semantic_hit

    # 3. Full RAG pipeline — passes FULL query so LLM has history context
    result = await _rag_pipeline.run_async(query, tenant_id=tenant_id)

    payload = {
        **result,
        "tenant_id":        tenant_id,
        "normalized_query": normalized,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "cache":            "miss"
    }

    # FIX: Never cache "not available" or escalation answers.
    # If confidence is too low and the answer is a fallback/not-found response,
    # caching it would permanently poison the cache — every future identical
    # question would get the bad answer even after the knowledge base is updated.
    # Only cache when: (a) escalate=False AND (b) answer is non-empty and not a
    # known not-found placeholder.
    _NOT_FOUND_MARKERS = (
        "not available in the provided",
        "could not find an answer",
        "no relevant information",
        "information is not available",
        "i don't have enough information",
        "please contact support",
        "i was unable to retrieve",
    )
    answer_text = (result.get("answer") or "").lower().strip()
    should_escalate = result.get("escalate", False)
    is_not_found = (
        not answer_text
        or any(marker in answer_text for marker in _NOT_FOUND_MARKERS)
    )

    if should_escalate or is_not_found:
        logger.info(
            f"RAG complete (NOT cached): conf={result.get('confidence')} "
            f"escalate={should_escalate} is_not_found={is_not_found} "
            f"query={normalized!r}"
        )
    else:
        set_exact(exact_key, payload)
        store_semantic(tenant_id, hash_text(normalized), {**payload, "embedding": q_emb})
        logger.info(
            f"RAG complete (cached): conf={result.get('confidence')} "
            f"query={normalized!r}"
        )

    return payload
