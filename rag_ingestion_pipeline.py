"""
rag_ingestion_pipeline.py

Production-grade, UI-observable RAG ingestion pipeline
with explicit file-type routing and correct image/table summarisation.
"""

import os
import uuid
import json
import shutil
import logging
from collections import defaultdict
from typing import List

from dotenv import load_dotenv

# ================== UNSTRUCTURED ==================
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.html import partition_html

# ================== LANGCHAIN ==================
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# =================================================
# ENV
# =================================================

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
SUMMARY_MODEL = "gpt-4o"

BASE_DIR = os.getcwd()
VECTOR_DB_DIR = f"{BASE_DIR}/db/chroma"
RAW_DIR = f"{BASE_DIR}/storage/raw_chunks"
REPORT_DIR = f"{BASE_DIR}/storage/ingestion_reports"
LOG_DIR = f"{BASE_DIR}/logs"

for d in [VECTOR_DB_DIR, RAW_DIR, REPORT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# =================================================
# LOGGING
# =================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/ingestion.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RAG-INGESTION")

# =================================================
# FILE TYPE DETECTION
# =================================================

def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext == ".pptx":
        return "pptx"
    if ext == ".txt":
        return "txt"
    if ext in [".html", ".htm"]:
        return "html"
    return "unsupported"

# =================================================
# PARTITION ROUTER
# =================================================

def partition_document(file_path: str):
    file_type = detect_file_type(file_path)
    logger.info(f"FILE ROUTING | {file_path} → {file_type}")

    if file_type == "pdf":
        return partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            hi_res_model_name="yolox"
        )
    if file_type == "docx":
        return partition_docx(filename=file_path)
    if file_type == "pptx":
        return partition_pptx(filename=file_path)
    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return partition_text(text=f.read())
    if file_type == "html":
        return partition_html(filename=file_path)

    raise ValueError(f"Unsupported file type: {file_path}")

# =================================================
# INGESTION PIPELINE
# =================================================

class RAGIngestionPipeline:
    def __init__(self, reset_vector_db=False):
        if reset_vector_db and os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        self.embedder = OpenAIEmbeddings(model=EMBED_MODEL)
        self.llm = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)

        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embedder
        )

    # ---------------------------------------------
    # QUEUE INGESTION
    # ---------------------------------------------

    def ingest_queue(self, files: List[str]):
        logger.info(f"QUEUE STARTED | total_files={len(files)}")
        for idx, file in enumerate(files, start=1):
            print(f"\n📄 Document {idx}/{len(files)} queued → {file}")
            self.ingest_single(file, idx, len(files))

    # ---------------------------------------------
    # SINGLE FILE INGESTION
    # ---------------------------------------------

    def ingest_single(self, file_path: str, qpos: int, qtotal: int):
        doc_id = str(uuid.uuid4())

        report = {
            "document_id": doc_id,
            "file_name": os.path.basename(file_path),
            "queue": {"position": qpos, "total": qtotal},
            "storage": {},
            "partitioning": {},
            "chunking": {},
            "summary": {},
            "vector_db": {},
            "chunks": []
        }

        print(f"⬆️ Uploading: {file_path}")

        # ---------------- STORAGE ----------------
        if not os.path.exists(file_path):
            report["storage"] = {"status": "failed", "error": "File not found"}
            return self._finalize(report)

        report["storage"] = {"status": "stored"}
        print("✅ Document stored")

        # ---------------- PARTITION ----------------
        print("🔍 Partitioning document...")
        elements = partition_document(file_path)

        stats = defaultdict(int)
        for el in elements:
            stats[el.category] += 1

        report["partitioning"] = {
            "status": "completed",
            "elements": dict(stats),
            "total": len(elements)
        }

        print("📊 Partition stats:", dict(stats))

        # ---------------- CHUNKING ----------------
        chunks = self._chunk_by_title(elements)
        total_chunks = len(chunks)
        report["chunking"]["total_chunks"] = total_chunks

        print(f"✂️ Chunking started ({total_chunks} chunks)")

        docs = []
        image_chunks = 0
        table_chunks = 0

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = str(uuid.uuid4())
            original = self._extract_original(chunk)

            has_images = bool(original["images_base64"])
            has_tables = bool(original["tables_html"])
            is_multimodal = has_images or has_tables

            if has_images:
                image_chunks += 1
                logger.info(f"IMAGE CHUNK | chunk_id={chunk_id}")

            if has_tables:
                table_chunks += 1

            summary = (
                self._summarize(self._build_summary_input(original))
                if is_multimodal
                else original["text"]
            )

            raw_path = f"{RAW_DIR}/{chunk_id}.json"
            with open(raw_path, "w") as f:
                json.dump(original, f, indent=2)

            docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "raw_path": raw_path,
                        "chunk_type": "multimodal" if is_multimodal else "text"
                    }
                )
            )

            report["chunks"].append({
                "chunk_id": chunk_id,
                "type": "multimodal" if is_multimodal else "text",
                "raw_path": raw_path,
                "summary": summary
            })

            if i % 10 == 0 or i == total_chunks:
                print(f"⏳ Chunk progress: {i}/{total_chunks}")

        # ---------------- SUMMARY STATS ----------------
        report["summary"] = {
            "image_chunks": image_chunks,
            "table_chunks": table_chunks,
            "text_chunks": total_chunks - (image_chunks + table_chunks),
            "status": "completed"
        }

        print(f"🧠 Summarised {image_chunks} image chunks")
        print(f"🧠 Summarised {table_chunks} table chunks")

        # ---------------- VECTOR DB ----------------
        print("📦 Writing embeddings to vector DB...")
        self.vector_db.add_documents(docs)

        report["vector_db"] = {
            "status": "completed",
            "embedded_documents": len(docs)
        }

        print("✅ Vector DB updated")

        return self._finalize(report)

    # ---------------------------------------------
    # HELPERS
    # ---------------------------------------------

    def _chunk_by_title(self, elements):
        chunks, current = [], []
        for el in elements:
            if el.category == "Title" and current:
                chunks.append(current)
                current = []
            current.append(el)
        if current:
            chunks.append(current)
        return chunks

    def _extract_original(self, elements):
        text, tables, images = [], [], []

        for el in elements:
            # TABLES
            if el.category == "Table" and hasattr(el.metadata, "text_as_html"):
                tables.append(el.metadata.text_as_html)

            # IMAGES (robust detection)
            if el.category in ("Image", "Figure"):
                if hasattr(el.metadata, "image_base64") and el.metadata.image_base64:
                    images.append(el.metadata.image_base64)
                elif hasattr(el.metadata, "image_path") and el.metadata.image_path:
                    images.append(el.metadata.image_path)

            # TEXT
            if el.category not in ("Image", "Figure"):
                text.append(str(el))

        return {
            "text": "\n".join(text).strip(),
            "tables_html": tables,
            "images_base64": images
        }

    def _build_summary_input(self, original: dict) -> str:
        parts = []

        if original["text"]:
            parts.append(f"TEXT CONTEXT:\n{original['text']}")

        if original["tables_html"]:
            parts.append(f"TABLE CONTEXT:\nContains {len(original['tables_html'])} table(s).")

        if original["images_base64"]:
            parts.append(f"IMAGE CONTEXT:\nContains {len(original['images_base64'])} figure(s).")

        return "\n\n".join(parts)

    def _summarize(self, content: str) -> str:
        prompt = f"""
You are summarizing a document chunk for a RAG system.

The chunk may include text, tables, and/or images.

Your task:
- Explain what this chunk represents
- Explain what information the image/table conveys
- Mention what kind of user questions this chunk can answer

Rules:
- Do NOT hallucinate unseen image details
- Use only the provided context
- Be concise and retrieval-oriented

CONTENT:
{content}
"""
        return self.llm.invoke(prompt).content.strip()

    def _finalize(self, report):
        path = f"{REPORT_DIR}/{report['document_id']}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📄 Report saved → {path}")
        logger.info(f"INGEST COMPLETE | {path}")
        return report

# =================================================
# CLI ENTRY
# =================================================

if __name__ == "__main__":
    pipeline = RAGIngestionPipeline(reset_vector_db=False)

    pipeline.ingest_queue([
        "test_docs/attention-is-all-you-need.pdf"
    ])
