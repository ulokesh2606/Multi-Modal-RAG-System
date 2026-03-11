import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from retrieval_config import GEN_MODEL, EMBED_MODEL, VECTOR_DB_DIR
from validation import validate_query
from query_expansion import QueryExpander
from vector_retrieval import HybridRetriever
from fusion import reciprocal_rank_fusion
from reranker import CrossEncoderReranker
from answer_generation import AnswerGenerator

load_dotenv()
logger = logging.getLogger("rag.pipeline")

_retriever_cache: dict = {}


def _get_retriever(tenant_id: str, embedder: OpenAIEmbeddings) -> "HybridRetriever":
    if tenant_id not in _retriever_cache:
        vector_db = Chroma(
            collection_name=f"tenant_{tenant_id}",
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embedder
        )
        _retriever_cache[tenant_id] = HybridRetriever(vector_db)
        logger.info(f"Built retriever for tenant '{tenant_id}'")
    return _retriever_cache[tenant_id]


class RAGRetrievalPipeline:
    def __init__(self):
        # max_tokens=300 hard-caps the answer at ~250 words (3-4 sentences)
        # and significantly reduces TPM usage on answer generation calls
        self.llm        = ChatOpenAI(model=GEN_MODEL, temperature=0, max_tokens=300)
        self.embedder   = OpenAIEmbeddings(model=EMBED_MODEL)
        self.expander   = QueryExpander(self.llm)
        self.reranker   = CrossEncoderReranker()
        self.answer_gen = AnswerGenerator(self.llm)

    def reload_bm25(self, tenant_id: str = None):
        if tenant_id and tenant_id in _retriever_cache:
            _retriever_cache[tenant_id]._load_bm25()
            logger.info(f"BM25 reloaded for tenant '{tenant_id}'")
        elif tenant_id is None:
            for tid, retriever in _retriever_cache.items():
                retriever._load_bm25()
                logger.info(f"BM25 reloaded for tenant '{tid}'")

    def invalidate_tenant_cache(self, tenant_id: str):
        if tenant_id in _retriever_cache:
            del _retriever_cache[tenant_id]
            logger.info(f"Retriever cache invalidated for tenant '{tenant_id}'")

    def run(self, query: str, tenant_id: str = "default") -> dict:
        return asyncio.run(self.run_async(query, tenant_id))

    async def run_async(self, query: str, tenant_id: str = "default") -> dict:
        v = validate_query(query)
        if not v["valid"]:
            return {"error": v["reason"], "confidence": "low"}

        retrieval_query = query
        if "[Current question]" in query:
            parts = query.split("[Current question]")
            retrieval_query = parts[-1].strip()
            logger.debug(f"Stripped history from query. Retrieval query: {retrieval_query!r}")

        queries = self.expander.generate(retrieval_query)
        logger.info(f"[{tenant_id}] Expanded to {len(queries)} query variants.")

        retriever = _get_retriever(tenant_id, self.embedder)

        if not retriever._collection_has_docs():
            if retriever.bm25 is not None:
                logger.info(f"[{tenant_id}] HNSW not ready — waiting up to 10s...")
                retriever._wait_for_hnsw(max_wait=10.0)
            else:
                logger.warning(f"[{tenant_id}] Collection is empty — no documents ingested.")
                return {
                    "answer":     "No documents have been ingested yet. Please upload documents first.",
                    "confidence": "low",
                    "cache":      "miss",
                    "sources":    [],
                }

        loop = asyncio.get_event_loop()
        try:
            results_list = await asyncio.gather(*[
                loop.run_in_executor(None, retriever.retrieve, q)
                for q in queries
            ])
        except Exception as e:
            logger.error(f"[{tenant_id}] Retrieval failed: {e}", exc_info=True)
            return {
                "answer":     "I encountered an error searching the knowledge base. Please try again.",
                "confidence": "low",
                "cache":      "miss",
                "sources":    [],
            }

        results  = dict(zip(queries, results_list))
        fused    = reciprocal_rank_fusion(results)
        reranked = self.reranker.rerank(retrieval_query, fused)

        # Pass all reranked docs — answer_generation.py caps to top 3 internally
        return self.answer_gen.generate(query, reranked)
