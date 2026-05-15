"""Google ID 토큰 검증 + 자체 JWT 발급."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
from jose import JWTError, jwt

_GOOGLE_TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"
_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
_JWT_SECRET       = os.getenv("JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM    = "HS256"
_JWT_EXPIRE_DAYS  = 30


async def verify_google_token(id_token: str) -> dict:
    """Google ID 토큰을 검증하고 {sub, email, name, picture} 반환."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(_GOOGLE_TOKENINFO, params={"id_token": id_token})
    if resp.status_code != 200:
        raise ValueError("invalid google token")
    info = resp.json()
    if _GOOGLE_CLIENT_ID and info.get("aud") != _GOOGLE_CLIENT_ID:
        raise ValueError("token audience mismatch")
    return {
        "google_id":  info["sub"],
        "email":      info["email"],
        "username":   info.get("name", info["email"].split("@")[0]),
        "avatar_url": info.get("picture"),
    }


def create_jwt(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=_JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_jwt(token: str) -> str:
    """user_id(str) 반환. 실패 시 ValueError."""
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        return payload["sub"]
    except JWTError as e:
        raise ValueError(f"invalid token: {e}")
