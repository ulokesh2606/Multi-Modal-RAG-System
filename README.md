# Multi-Modal RAG System

A **production-style Retrieval Augmented Generation (RAG) system** designed for scalable document ingestion, advanced retrieval pipelines, and API-based querying.

This project demonstrates how to build an **end-to-end intelligent retrieval system** capable of processing large document collections and generating grounded responses using LLMs.

---

# Overview

The system is divided into three main modules:

- **rag_ingestion** → Handles document processing and indexing
- **rag_retrieval** → Implements the retrieval and ranking pipeline
- **service** → Provides API services, caching, and security

---

# System Architecture


User Query
│
▼
API Service Layer
│
▼
Query Processing
│
▼
Retrieval Pipeline
│
▼
Vector Database
│
▼
Answer Generation
│
▼
Response to User


---

# End-to-End User Flow

When a user sends a query, the system processes it through multiple stages.


User Query
│
▼
API Endpoint
│
▼
Query Normalization
│
▼
Query Expansion
│
▼
Hybrid Retrieval
│
▼
Reciprocal Rank Fusion
│
▼
Reranker
│
▼
Context Assembly
│
▼
LLM Answer Generation
│
▼
Final Response


---

# Document Ingestion Pipeline

The ingestion pipeline prepares documents before they are stored in the vector database.


Raw Documents
│
▼
Document Partitioning
│
▼
Chunking
│
▼
Chunk Validation
│
▼
Summarization
│
▼
Embedding Generation
│
▼
Vector Database Storage


### Key Steps

#### **Partitioning**

Splits documents into structured sections such as paragraphs, headings, and tables.

#### **Chunking**

Breaks documents into smaller segments suitable for embedding.

#### **Validation**

Ensures chunks meet quality and size constraints.

#### **Summarization**

Generates condensed representations of document sections.

#### **Embedding Generation**

Transforms chunks into vector embeddings using embedding models.

#### **Vector Storage**

Stores embeddings in a vector database for semantic retrieval.

---

# Retrieval Pipeline

The retrieval system identifies the most relevant documents for a query.


User Query
│
▼
Query Expansion
│
▼
Vector Search
│
▼
Keyword Search
│
▼
Hybrid Retrieval
│
▼
Reciprocal Rank Fusion
│
▼
Reranking
│
▼
Top Context Selection
│
▼
Answer Generation


---

# Core Retrieval Techniques

### **Query Expansion**

Generates alternative queries to improve recall.

### **Hybrid Retrieval**

Combines:

- Vector search
- Keyword search

### **Reciprocal Rank Fusion (RRF)**

Merges results from multiple retrieval methods to improve ranking.

### **Reranking**

Ranks retrieved results based on semantic relevance.

---

# Service Layer

The **service module** exposes the RAG system through APIs and manages system-level features.

### Responsibilities

- Query normalization
- Semantic caching
- Redis integration
- Security and authentication
- Tenant validation

---

# Project Structure


Multi-Modal-RAG-System
│
├── rag_ingestion
│ ├── chunking.py
│ ├── partitioning.py
│ ├── summarization.py
│ ├── vectorstore.py
│ └── pipeline.py
│
├── rag_retrieval
│ ├── query_expansion.py
│ ├── vector_retrieval.py
│ ├── reranker.py
│ ├── fusion.py
│ └── retrieval_pipeline.py
│
├── service
│ ├── app.py
│ ├── query_normalization.py
│ ├── service_config.py
│ │
│ ├── cache
│ │ ├── redis_client.py
│ │ ├── semantic_cache.py
│ │ └── exact_cache.py
│ │
│ ├── security
│ │ ├── auth.py
│ │ └── tenant_validation.py
│ │
│ └── services
│ └── rag_service.py
│
└── requirements.txt


---

# Installation

### Clone the repository


git clone https://github.com/ulokesh2606/Multi-Modal-RAG-System.git

cd Multi-Modal-RAG-System


### Install dependencies


pip install -r requirements.txt


---

# Running the Service


python service/app.py


The API service will start and be ready to process queries.

---

# Future Improvements

- Evaluation pipeline for retrieval quality
- Streaming LLM responses
- Docker deployment
- Observability and monitoring
- Multi-modal retrieval extensions

---

# Key Highlights

**Advanced Retrieval Techniques**

- Query Expansion
- Hybrid Retrieval
- Reciprocal Rank Fusion
- Reranking

**Production Features**

- API-based architecture
- Semantic caching
- Modular pipeline design
- Security layer

---

# License

This project is open-source and available under the MIT License.
