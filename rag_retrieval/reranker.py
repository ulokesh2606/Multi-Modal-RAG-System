from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from retrieval_config import RERANKER_TOP_N


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def rerank(self, query: str, docs: List[Document], top_n: int = None):
        top_n = top_n or RERANKER_TOP_N
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [{"doc": d, "score": float(s)} for d, s in ranked[:top_n]]

