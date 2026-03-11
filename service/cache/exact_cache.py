import json
from cache.redis_client import redis_client
from service_config import EXACT_CACHE_TTL


def get_exact(key: str):
    val = redis_client.get(key)
    return json.loads(val) if val else None


def set_exact(key: str, payload: dict):
    redis_client.setex(key, EXACT_CACHE_TTL, json.dumps(payload))

