import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from auth import get_current_user, require_admin
from main import (
    build_client,
    build_vector_store,
    chunk_id,
    filter_existing_chunks,
    split_documents,
    normalize_source,
    COLLECTION_NAME,
    BATCH_SIZE,
)
from rag_service import run_rag

from langchain_community.document_loaders import PyMuPDFLoader

app = FastAPI(title="RAG API", version="1.0.0")

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


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, _: dict = Depends(get_current_user)) -> AskResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = run_rag(body.question)
    return AskResponse(answer=answer)


@app.post("/upload")
async def upload_pdf(
    file: UploadFile,
    _: dict = Depends(require_admin),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        docs = PyMuPDFLoader(str(tmp_path)).load()
        for doc in docs:
            doc.metadata["source"] = file.filename
            doc.metadata["file_path"] = file.filename

        chunks, ids = split_documents(docs)
        if not chunks:
            return {"message": "No content extracted from PDF", "chunks": 0}

        client = build_client()
        chunks, ids = filter_existing_chunks(client, chunks, ids)

        if not chunks:
            return {"message": f"{file.filename} already indexed", "chunks": 0}

        vector_store = build_vector_store()
        vector_store.add_documents(chunks, ids=ids, batch_size=BATCH_SIZE)

        return {"message": f"Indexed {file.filename} successfully", "chunks": len(chunks)}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents")
def list_documents(_: dict = Depends(get_current_user)) -> dict:
    client = build_client()
    if not client.collection_exists(COLLECTION_NAME):
        return {"documents": []}

    sources: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata") or {}
            source = metadata.get("source") or metadata.get("file_path")
            if source:
                sources.add(normalize_source(source))
        if offset is None:
            break

    return {"documents": sorted(sources)}
