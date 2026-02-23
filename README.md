# 🚀 Multi-Modal RAG System

A **production-grade, safety-first Retrieval-Augmented Generation (RAG) system**
designed to handle **real-world multimodal documents** containing **text, tables,
and images**, using **hybrid retrieval** and **grounded answer generation**.

This project prioritizes **correctness, traceability, and extensibility** over
toy demos.

---

## ✨ Key Features

### 🔹 Ingestion Pipeline
- File-type aware routing (`PDF`, `DOCX`, `PPTX`, `TXT`, `HTML`)
- High-resolution PDF parsing
  - Table structure inference
  - Image extraction
- Title-based semantic chunking
- Automatic multimodal chunk detection
- LLM-based summarization for image/table chunks
- Raw chunk persistence for auditability
- Vector database storage with rich metadata
- Detailed ingestion reports and logs

### 🔹 Retrieval Pipeline
- Multi-query expansion using LLM
- Hybrid retrieval strategy:
  - Dense vector search (Cosine + MMR)
  - Sparse keyword search (BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Strict grounded answer generation
- Safe fallback for out-of-domain queries

---

## 🏗 Ingestion Pipeline – Architecture Diagram

The ingestion pipeline is designed to robustly process **multimodal documents**
while preserving full traceability and auditability.

![RAG Ingestion Architecture](https://github.com/ulokesh2606/Multi-Modal-RAG-System/blob/main/images/Ingestion.png)

## ⚙️ Ingestion Flow (Detailed)

1. **File Detection**
   - Routes documents based on extension

2. **Partitioning**
   - Uses Unstructured for document parsing
   - Extracts text, tables (HTML), and images

3. **Chunking**
   - Title-based segmentation for semantic integrity

4. **Multimodal Handling**
   - Detects tables/images per chunk
   - Summarizes multimodal chunks using LLM
   - Text-only chunks bypass summarization

5. **Persistence**
   - Raw chunks saved as JSON
   - Metadata includes document ID, chunk ID, and chunk type

6. **Embedding & Storage**
   - Embeddings stored in Chroma vector database
   - Full ingestion report generated per document

---
![RAG Ingestion Architecture](https://github.com/ulokesh2606/Multi-Modal-RAG-System/blob/main/images/Retrievalpng)
## 🔍 Retrieval Flow (Detailed)

1. **Query Expansion**
   - Generates multiple semantically diverse queries

2. **Hybrid Retrieval**
   - Vector search with MMR for diversity
   - BM25 keyword search for lexical recall

3. **Fusion**
   - Reciprocal Rank Fusion merges results across queries

4. **Reranking**
   - Cross-encoder scores query–chunk relevance

5. **Answer Generation**
   - LLM answers strictly from retrieved context
   - Explicit fallback for missing information

---

## 🛡️ Safety & Correctness (Current)

- ❌ No hallucinated answers
- ❌ No external knowledge usage
- ✅ Strict context grounding
- ✅ Deterministic generation (temperature = 0)
- ✅ Raw content preserved for auditability

---

## Currently working

### Query Engine
- User ask the questions in their preferred language 
- Language Translation using Sarvam AI
- Normalize the User query and get the intent detection

### Safety Policies & Guardrails 
- Query intent classification
- Domain allow/deny lists
- Prompt-level safety enforcement
- Policy-driven response shaping

### Document & Chunk Validation 
- Schema validation for extracted chunks
- Low-signal / empty chunk filtering
- Duplicate chunk detection
- Semantic density checks

### Answer Validation 
- Answer-to-context entailment verification
- Citation enforcement
- Secondary verification pass

## Future Scope

### Confidence Scoring
- Retrieval confidence based on rank distribution
- Cross-encoder score aggregation
- Answer-level confidence estimation

### Agentic Extensions
- Planner–Retriever–Verifier agents
- Adaptive retrieval depth
- Dynamic reranking strategies
- Developing the agent to work using the grounded source of truths
---

## How to Run

### Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 
```bash
python ingestion_pipeline.py

python retrieval_pipeline.py
```
