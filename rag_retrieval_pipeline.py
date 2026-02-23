"""
rag_retrieval_pipeline.py

Production-grade Hybrid RAG Retrieval Pipeline
- Multi-query expansion
- Vector search (MMR + cosine)
- Keyword search (BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Strong grounded prompting
"""

import os
import json
import logging
from typing import List, Dict
from collections import defaultdict

from dotenv import load_dotenv

# ---------------- LangChain ----------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------- Retrieval ----------------
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ======================================================
# ENV + LOGGING
# ======================================================

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o"

VECTOR_DB_DIR = "db/chroma"
RAW_CHUNK_DIR = "storage/raw_chunks"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("RAG-RETRIEVAL")

# ======================================================
# RETRIEVAL PIPELINE
# ======================================================

class RAGRetrievalPipeline:
    def __init__(self):
        self.embedder = OpenAIEmbeddings(model=EMBED_MODEL)
        self.llm = ChatOpenAI(model=GEN_MODEL, temperature=0)

        # ---- Vector DB (cosine already set at ingestion) ----
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embedder
        )

        # ---- Cross Encoder (loaded once, cached locally) ----
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

        self._load_bm25_index()

    # --------------------------------------------------
    # BM25 INDEX
    # --------------------------------------------------

    def _load_bm25_index(self):
        logger.info("Loading BM25 index from vector DB...")

        data = self.vector_db.get(include=["documents", "metadatas"])

        self.bm25_docs = []
        self.bm25_meta = []

        for text, meta in zip(data["documents"], data["metadatas"]):
            self.bm25_docs.append(text.split())
            self.bm25_meta.append(meta)

        self.bm25 = BM25Okapi(self.bm25_docs)

        logger.info(f"BM25 ready | total_docs={len(self.bm25_docs)}")

    # --------------------------------------------------
    # MULTI QUERY GENERATION
    # --------------------------------------------------

    def generate_queries(self, user_query: str) -> List[str]:
        prompt = f"""
You are generating search queries for a RAG system.

Original question:
"{user_query}"

Generate exactly 3 alternative queries that:
- Use different technical phrasing
- Expand acronyms if any
- Emphasize definitions, mechanisms, or steps

Return one query per line.
Do not explain.
"""

        response = self.llm.invoke(prompt).content.strip()
        variations = [q.strip() for q in response.split("\n") if q.strip()]

        queries = [user_query] + variations
        logger.info(f"Generated {len(queries)} queries")

        return queries

    # --------------------------------------------------
    # HYBRID RETRIEVAL (MMR + BM25)
    # --------------------------------------------------

    def hybrid_retrieve(self, query: str, k: int = 40) -> List[Document]:
        # ---- VECTOR SEARCH (MMR + COSINE) ----
        vector_docs = self.vector_db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=3 * k,
            lambda_mult=0.5
        )

        # ---- KEYWORD SEARCH (BM25) ----
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        bm25_docs = []
        for idx, _ in ranked:
            meta = self.bm25_meta[idx]
            raw_path = meta.get("raw_path")

            if not raw_path or not os.path.exists(raw_path):
                continue

            with open(raw_path, "r") as f:
                raw = json.load(f)

            bm25_docs.append(
                Document(
                    page_content=raw["text"],
                    metadata=meta
                )
            )

        return vector_docs + bm25_docs

    # --------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------

    def rrf(self, results: Dict[str, List[Document]], top_k: int = 20):
        scores = defaultdict(float)

        for docs in results.values():
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata["chunk_id"]
                scores[doc_id] += 1 / (rank + 60)

        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_docs = []
        seen = set()

        for doc_id, _ in ranked_ids:
            for docs in results.values():
                for doc in docs:
                    if doc.metadata["chunk_id"] == doc_id and doc_id not in seen:
                        final_docs.append(doc)
                        seen.add(doc_id)
                        break
            if len(final_docs) >= top_k:
                break

        return final_docs

    # --------------------------------------------------
    # CROSS ENCODER RERANK
    # --------------------------------------------------

    def rerank(self, query: str, docs: List[Document], top_n: int = 5):
        if not docs:
            return []

        pairs = [(query, d.page_content) for d in docs]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_n]]

    # --------------------------------------------------
    # ANSWER GENERATION
    # --------------------------------------------------

    def answer(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return "The information is not available in the provided documents."

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You must answer ONLY using the context below.

Rules:
- Do not use outside knowledge
- Do not hallucinate
- If the answer is not present, say:
  "The information is not available in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        return self.llm.invoke(prompt).content.strip()

    # --------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------

    def run(self, query: str) -> str:
        queries = self.generate_queries(query)

        all_results = {}
        for q in queries:
            all_results[q] = self.hybrid_retrieve(q)

        fused = self.rrf(all_results)
        reranked = self.rerank(query, fused)

        return self.answer(query, reranked)


# ======================================================
# CLI TEST
# ======================================================

if __name__ == "__main__":
    rag = RAGRetrievalPipeline()

    print("\n--- TEST: IN-DOMAIN QUESTION ---\n")
    print(
        rag.run("Explain the attention mechanism in transformers")
    )

    print("\n--- TEST: OUT-OF-DOMAIN QUESTION ---\n")
    print(
        rag.run("What is the capital of France?")
    )
