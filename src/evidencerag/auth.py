import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from evidencerag.config import get_settings

bearer = HTTPBearer(auto_error=False)


def authenticate(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    if credentials is not None:
        for owner_id, secret in get_settings().api_keys.items():
            key = secret.get_secret_value()
            if key and secrets.compare_digest(credentials.credentials.encode(), key.encode()):
                return owner_id
    raise HTTPException(
        status_code=401,
        detail="Нужен действующий ключ клиента",
        headers={"WWW-Authenticate": "Bearer"},
    )


OwnerId = Annotated[str, Depends(authenticate)]
