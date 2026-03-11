from collections import defaultdict
from typing import Dict, List
from langchain_core.documents import Document


def reciprocal_rank_fusion(results: Dict[str, List[Document]], top_k: int = 20) -> List[Document]:
    scores: dict = defaultdict(float)
    doc_lookup: dict = {}

    for docs in results.values():
        for rank, doc in enumerate(docs):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1.0 / (rank + 60)
            if cid not in doc_lookup:
                doc_lookup[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[cid] for cid, _ in ranked[:top_k] if cid in doc_lookup]

