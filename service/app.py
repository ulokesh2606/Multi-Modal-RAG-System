import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services.rag_service import handle_query, init_service, close_service, get_pipeline
from security.auth import verify_api_key

logger = logging.getLogger("service.app")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Service starting...")
    await init_service()
    yield
    await close_service()


app = FastAPI(title="RAG Service API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{rid}] Unhandled: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "request_id": rid}
    )


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=4, max_length=64,
                           pattern=r'^[a-zA-Z0-9_-]+$')
    query: str = Field(..., min_length=5, max_length=4000)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rag/query", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def query(payload: QueryRequest, request: Request):
    try:
        return await handle_query(tenant_id=payload.tenant_id, query=payload.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─── Document Stats Endpoint ───────────────────────────────────────────────────

@app.get("/documents", dependencies=[Depends(verify_api_key)])
async def get_documents(tenant_id: str = "default"):
    """Return document metadata for a tenant's ChromaDB collection.

    BUG FIX: Registry path was using parent.parent.parent which resolves to
    /home/lokes/ — but rag_ingestion lives at /home/lokes/wsl/rag_ingestion/.
    
    Path breakdown (app.py is at /home/lokes/wsl/service/app.py):
      __file__              = /home/lokes/wsl/service/app.py
      parent                = /home/lokes/wsl/service/
      parent.parent         = /home/lokes/wsl/          ← correct wsl root
      parent.parent.parent  = /home/lokes/              ← WRONG (was used before)

    Using parent.parent gives: /home/lokes/wsl/rag_ingestion/storage/document_registry.json ✓
    Using parent.parent.parent gave: /home/lokes/rag_ingestion/... ✗ (doesn't exist)
    This caused the registry to never load → filenames always showed as hash slices.
    """
    from security.tenant_validation import validate_tenant
    from retrieval_config import VECTOR_DB_DIR, EMBED_MODEL
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    import os
    import json
    from pathlib import Path as _Path

    try:
        tenant_id = validate_tenant(tenant_id)

        # ── FIX: use parent.parent (wsl root), not parent.parent.parent (home dir) ──
        # app.py: /home/lokes/wsl/service/app.py
        # parent.parent = /home/lokes/wsl/  ← rag_ingestion lives here
        _registry_path = _Path(os.getenv(
            "REGISTRY_PATH",
            str(_Path(__file__).parent.parent / "rag_ingestion" / "storage" / "document_registry.json")
        ))

        _doc_id_to_meta: dict = {}
        try:
            registry = json.loads(_registry_path.read_text())
            for _entry in registry.get("documents", {}).values():
                _did = _entry.get("document_id")
                if _did:
                    _doc_id_to_meta[_did] = {
                        "filename":    _entry.get("file", ""),
                        "ingested_at": _entry.get("ingested_at", ""),
                    }
            logger.info(f"Registry loaded: {len(_doc_id_to_meta)} documents from {_registry_path}")
        except Exception as _e:
            logger.warning(f"Could not load registry: {_e} (path={_registry_path})")

        # ── Query ChromaDB ─────────────────────────────────────────────────
        db = Chroma(
            collection_name=f"tenant_{tenant_id}",
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=OpenAIEmbeddings(model=EMBED_MODEL)
        )
        data = db.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []

        # Group chunks by document_id, resolve filename from registry
        docs: dict = {}
        for m in metadatas:
            did = m.get("document_id", "unknown")
            if did not in docs:
                reg_meta = _doc_id_to_meta.get(did, {})
                filename = reg_meta.get("filename", "")
                # Only fall back to hash slice if filename is genuinely missing
                if not filename or (filename.startswith("tmp") and len(filename) < 20):
                    filename = did[:16] + "…"
                docs[did] = {
                    "document_id": did,
                    "chunk_count": 0,
                    "tenant_id":   tenant_id,
                    "filename":    filename,
                    "ingested_at": reg_meta.get("ingested_at", ""),
                }
            docs[did]["chunk_count"] += 1

        return {
            "tenant_id":       tenant_id,
            "total_chunks":    len(metadatas),
            "total_documents": len(docs),
            "documents":       list(docs.values()),
        }
    except Exception as e:
        logger.error(f"get_documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Ingestion Endpoint ───────────────────────────────────────────────
import subprocess
import tempfile
import os
import json
from pathlib import Path as FPath
from fastapi import UploadFile, File, Form

ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.pptx', '.xlsx', '.csv', '.md'}
WSL_INGEST_DIR = FPath(__file__).parent.parent / "rag_ingestion"


@app.post("/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_file(
    file: UploadFile = File(...),
    tenant_id: str = Form(default="default")
):
    ext = FPath(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")

    # FIX: normalise tenant_id at ingestion time to match what validate_tenant()
    # does at query time (lowercases the id). This ensures ChromaDB collection
    # name is consistent between ingestion and retrieval.
    tenant_id = tenant_id.lower().strip()

    logger.info(f"Ingesting: {file.filename} ({len(content)} bytes) tenant={tenant_id}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python", "cli.py", tmp_path, tenant_id, file.filename],
            cwd=str(WSL_INGEST_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Ingestion failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {result.stderr[-300:]}"
            )

        last_line = result.stdout.strip().split('\n')[-1]
        try:
            stats = json.loads(last_line)
        except Exception:
            stats = {"status": "success"}

        pipeline = get_pipeline()
        if pipeline:
            pipeline.invalidate_tenant_cache(tenant_id)
            logger.info(f"Retriever cache invalidated for tenant '{tenant_id}' after ingestion")

        return {
            "status":          "success",
            "filename":        file.filename,
            "tenant_id":       tenant_id,
            "chunks_embedded": stats.get("chunks_embedded", 0),
            "chunks_created":  stats.get("chunks_created", 0),
        }
    finally:
        os.unlink(tmp_path)
