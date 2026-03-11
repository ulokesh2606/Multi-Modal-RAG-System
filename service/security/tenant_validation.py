import re

TENANT_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{4,64}$')


def validate_tenant(tenant_id: str) -> str:
    if not tenant_id:
        raise ValueError("tenant_id_missing")
    if not TENANT_PATTERN.match(tenant_id):
        raise ValueError("invalid_tenant_id")
    return tenant_id.lower()
