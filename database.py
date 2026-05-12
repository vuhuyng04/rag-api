import os
from datetime import date

from supabase import create_client, Client

FREE_QUESTION_LIMIT = 10
FREE_DOC_LIMIT = 1


def get_client() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL và SUPABASE_SECRET_KEY chưa được set")
    return create_client(url, key)


def get_or_create_user(clerk_user_id: str) -> dict:
    db = get_client()
    res = db.table("users").select("*").eq("clerk_user_id", clerk_user_id).execute()
    if res.data:
        return res.data[0]
    insert = db.table("users").insert({"clerk_user_id": clerk_user_id}).execute()
    return insert.data[0]


def get_user_plan(clerk_user_id: str) -> str:
    user = get_or_create_user(clerk_user_id)
    return user.get("plan", "free")


def count_user_documents(clerk_user_id: str) -> int:
    db = get_client()
    res = db.table("documents").select("id", count="exact").eq("user_id", clerk_user_id).execute()
    return res.count or 0


def add_document(clerk_user_id: str, filename: str, chunk_count: int) -> dict:
    db = get_client()
    res = db.table("documents").insert({
        "user_id": clerk_user_id,
        "filename": filename,
        "chunk_count": chunk_count,
    }).execute()
    return res.data[0]


def get_user_documents(clerk_user_id: str) -> list[dict]:
    db = get_client()
    res = (
        db.table("documents")
        .select("*")
        .eq("user_id", clerk_user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def delete_document(doc_id: str, clerk_user_id: str) -> bool:
    db = get_client()
    res = (
        db.table("documents")
        .delete()
        .eq("id", doc_id)
        .eq("user_id", clerk_user_id)
        .execute()
    )
    return bool(res.data)


def get_daily_usage(clerk_user_id: str) -> int:
    db = get_client()
    today = date.today().isoformat()
    res = (
        db.table("usage")
        .select("question_count")
        .eq("user_id", clerk_user_id)
        .eq("date", today)
        .execute()
    )
    if res.data:
        return res.data[0]["question_count"]
    return 0


def increment_daily_usage(clerk_user_id: str) -> int:
    db = get_client()
    today = date.today().isoformat()

    # Upsert to avoid race condition on first insert
    db.table("usage").upsert(
        {"user_id": clerk_user_id, "date": today, "question_count": 0},
        on_conflict="user_id,date",
        ignore_duplicates=True,
    ).execute()

    current = get_daily_usage(clerk_user_id)
    new_count = current + 1
    db.table("usage").update({"question_count": new_count}).eq(
        "user_id", clerk_user_id
    ).eq("date", today).execute()
    return new_count
