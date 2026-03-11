import os
import json
import logging
import time
from typing import List

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger("rag.hybrid_retriever")


class HybridRetriever:
    def __init__(self, vector_db: Chroma):
        self.vector_db = vector_db
        self._load_bm25()

    def _load_bm25(self):
        data = self.vector_db.get(include=["documents", "metadatas"])
        chroma_docs = data.get("documents") or []
        metadatas   = data.get("metadatas") or []

        if not chroma_docs:
            self.bm25      = None
            self.bm25_docs = []
            self.bm25_meta = []
            self.bm25_texts = []
            return

        # Build token lists from ORIGINAL text (not summary) for accurate BM25 scoring
        token_lists = []
        full_texts  = []

        for chroma_text, meta in zip(chroma_docs, metadatas):
            raw_path = meta.get("raw_path")
            original_text = ""

            if raw_path and os.path.exists(raw_path):
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    # Use original text for BM25 — preserves exact terms like
                    # "$7.5 billion", "GitHub", "Bill Gates", etc.
                    original_text = raw.get("original", {}).get("text", "")
                    if not original_text.strip():
                        # Fall back to summary if original is empty
                        original_text = raw.get("summary", "")
                except Exception:
                    pass

            # Final fallback: use the ChromaDB summary text
            bm25_text = original_text.strip() if original_text.strip() else chroma_text

            token_lists.append(bm25_text.lower().split())
            full_texts.append(bm25_text)

        self.bm25_docs  = token_lists
        self.bm25_meta  = metadatas
        self.bm25_texts = full_texts
        self.bm25       = BM25Okapi(self.bm25_docs)
        logger.info(f"BM25 index built with {len(chroma_docs)} documents (using original text).")

    def reload(self):
        """Hot-reload BM25 after new ingestion. No restart needed."""
        logger.info("Reloading BM25 index...")
        self._load_bm25()

    def _collection_has_docs(self) -> bool:
        
        try:
            count = self.vector_db._collection.count()
            if count == 0:
                return False
            # Quick probe: try a tiny similarity search to confirm HNSW is live
            self.vector_db.similarity_search("test", k=1)
            return True
        except Exception:
            return False

    def _wait_for_hnsw(self, max_wait: float = 10.0) -> bool:
        
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self._collection_has_docs():
                return True
            logger.info("HNSW index not ready yet, waiting 1s...")
            time.sleep(1.0)
        logger.warning("HNSW index did not become ready in time — falling back to BM25 only.")
        return False

    def retrieve(self, query: str, k: int = 40) -> List[Document]:
        
        vector_docs = []

        if self._collection_has_docs():
            try:
                vector_docs = self.vector_db.max_marginal_relevance_search(
                    query, k=k, fetch_k=3 * k, lambda_mult=0.5
                )
            except Exception as e:
                logger.warning(
                    f"Vector search failed (returning empty — BM25 will cover): {e}"
                )
                vector_docs = []

        enriched_vector_docs = []
        for doc in vector_docs:
            raw_path = doc.metadata.get("raw_path")
            original_text = ""
            if raw_path and os.path.exists(raw_path):
                try:
                    with open(raw_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    original_text = raw.get("original", {}).get("text", "").strip()
                except Exception:
                    pass
            # Fall back to summary (page_content) if original is missing
            if original_text:
                doc = Document(page_content=original_text, metadata=doc.metadata)
            enriched_vector_docs.append(doc)
        vector_docs = enriched_vector_docs

        bm25_docs = []
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k]
            for idx, score in ranked:
                if score <= 0:
                    # Skip chunks with zero/negative BM25 score — they share no
                    # query terms and would only add noise for the cross-encoder.
                    continue
                meta = self.bm25_meta[idx]
                # Use the pre-loaded original text (consistent with BM25 scoring)
                text = self.bm25_texts[idx]
                if text.strip():
                    bm25_docs.append(Document(page_content=text, metadata=meta))

        return vector_docs + bm25_docs
