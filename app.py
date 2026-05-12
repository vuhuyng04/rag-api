import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from auth import get_current_user
from database import (
    FREE_DOC_LIMIT,
    FREE_QUESTION_LIMIT,
    add_document,
    count_user_documents,
    delete_document,
    get_daily_usage,
    get_or_create_user,
    get_user_documents,
    get_user_plan,
    increment_daily_usage,
)
from main import BATCH_SIZE, COLLECTION_NAME, build_client, build_vector_store, filter_existing_chunks, split_documents
from qdrant_client.models import PayloadSchemaType
from rag_service import run_rag

from langchain_community.document_loaders import PyMuPDFLoader

app = FastAPI(title="DocuChat API", version="1.0.0")


@app.on_event("startup")
def create_qdrant_indexes() -> None:
    try:
        client = build_client()
        if client.collection_exists(COLLECTION_NAME):
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    except Exception:
        pass

allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/me")
def get_me(user: dict = Depends(get_current_user)) -> dict:
    user_id = user["sub"]
    db_user = get_or_create_user(user_id)
    plan = db_user.get("plan", "free")
    doc_count = count_user_documents(user_id)
    daily_questions = get_daily_usage(user_id)

    return {
        "user_id": user_id,
        "plan": plan,
        "doc_count": doc_count,
        "doc_limit": None if plan == "pro" else FREE_DOC_LIMIT,
        "daily_questions": daily_questions,
        "question_limit": None if plan == "pro" else FREE_QUESTION_LIMIT,
    }


@app.get("/docs")
def list_docs(user: dict = Depends(get_current_user)) -> dict:
    docs = get_user_documents(user["sub"])
    return {"documents": docs}


@app.post("/docs/upload")
async def upload_pdf(
    file: UploadFile,
    user: dict = Depends(get_current_user),
) -> dict:
    user_id = user["sub"]

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

    plan = get_user_plan(user_id)
    if plan == "free":
        doc_count = count_user_documents(user_id)
        if doc_count >= FREE_DOC_LIMIT:
            raise HTTPException(
                status_code=403,
                detail=f"Gói Free chỉ được upload {FREE_DOC_LIMIT} PDF. Nâng cấp Pro để upload thêm.",
            )

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        docs = PyMuPDFLoader(str(tmp_path)).load()
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["file_path"] = file.filename
            doc.metadata["user_id"] = user_id

        chunks, ids = split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Không trích xuất được nội dung từ PDF")

        client = build_client()
        chunks, ids = filter_existing_chunks(client, chunks, ids)

        if not chunks:
            raise HTTPException(status_code=409, detail="PDF này đã được index rồi")

        vector_store = build_vector_store()
        vector_store.add_documents(chunks, ids=ids, batch_size=BATCH_SIZE)

        add_document(user_id, file.filename, len(chunks))

        return {"message": f"Đã index {file.filename} thành công", "chunks": len(chunks)}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/docs/{doc_id}")
def remove_doc(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    deleted = delete_document(doc_id, user["sub"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    return {"message": "Đã xóa tài liệu"}


@app.post("/chat/ask")
def ask(body: AskRequest, user: dict = Depends(get_current_user)) -> dict:
    user_id = user["sub"]

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    plan = get_user_plan(user_id)
    if plan == "free":
        daily = get_daily_usage(user_id)
        if daily >= FREE_QUESTION_LIMIT:
            raise HTTPException(
                status_code=403,
                detail=f"Đã dùng hết {FREE_QUESTION_LIMIT} câu hỏi hôm nay. Nâng cấp Pro để hỏi không giới hạn.",
            )

    increment_daily_usage(user_id)
    answer = run_rag(body.question, user_id=user_id)
    return {"answer": answer}
