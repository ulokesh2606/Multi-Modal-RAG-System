import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

VALID_API_KEYS = set(filter(None, os.getenv("RAG_API_KEYS", "dev-key-123").split(",")))


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
