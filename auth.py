import os
from functools import lru_cache

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

CLERK_JWKS_URL = os.getenv(
    "CLERK_JWKS_URL",
    "https://optimum-monarch-22.clerk.accounts.dev/.well-known/jwks.json",
)
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "")

bearer = HTTPBearer()


@lru_cache(maxsize=1)
def _get_jwks() -> dict:
    resp = httpx.get(CLERK_JWKS_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _verify_token(token: str) -> dict:
    jwks = _get_jwks()
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")

    key = next(
        (k for k in jwks.get("keys", []) if k.get("kid") == kid),
        None,
    )
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token key")

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return payload


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    return _verify_token(creds.credentials)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not ADMIN_USER_ID or user.get("sub") != ADMIN_USER_ID:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user
