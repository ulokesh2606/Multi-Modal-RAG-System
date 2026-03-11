INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all instructions",
    "you are chatgpt", "you are now", "system prompt",
    "act as", "bypass", "jailbreak", "forget instructions",
    "disregard", "new persona", "pretend you are",
    "your new instructions", "override", "admin mode",
]

MAX_QUERY_LENGTH = 4000
MIN_QUERY_LENGTH = 5


def validate_query(query: str) -> dict:
    q = query.strip()
    if len(q) < MIN_QUERY_LENGTH:
        return {"valid": False, "reason": "query_too_short"}
    if len(q) > MAX_QUERY_LENGTH:
        return {"valid": False, "reason": "query_too_long"}
    q_lower = q.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in q_lower:
            return {"valid": False, "reason": "prompt_injection_detected"}
    return {"valid": True}

