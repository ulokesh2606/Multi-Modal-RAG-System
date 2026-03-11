import os
from dotenv import load_dotenv
import redis

load_dotenv()

_pool = redis.ConnectionPool(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD") or None,
    decode_responses=True,
    max_connections=20
)

redis_client = redis.Redis(connection_pool=_pool)
