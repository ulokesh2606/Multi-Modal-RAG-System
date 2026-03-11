from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import VECTOR_DB_DIR, EMBED_MODEL

load_dotenv()


class VectorStore:
    def __init__(self, tenant_id: str = "default"):
        # Each tenant gets their own ChromaDB collection.
        # This is the critical isolation boundary for multi-tenancy.
        # Collection names: tenant_default, tenant_acme, tenant_contoso, etc.
        collection_name = f"tenant_{tenant_id}"
        self.db = Chroma(
            collection_name=collection_name,
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=OpenAIEmbeddings(model=EMBED_MODEL)
        )

    def add(self, documents):
        if documents:
            self.db.add_documents(documents)
