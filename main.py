import hashlib
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient


DOCS_DIR = Path("docs")
COLLECTION_NAME = "rag_docs"
BATCH_SIZE = 32
TIMEOUT = 120


def build_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False,
        timeout=TIMEOUT,
    )


def build_vector_store() -> QdrantVectorStore:
    dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return QdrantVectorStore.construct_instance(
        embedding=dense_embeddings,
        client_options={
            "url": os.getenv("QDRANT_URL"),
            "api_key": os.getenv("QDRANT_API_KEY"),
            "prefer_grpc": False,
            "timeout": TIMEOUT,
        },
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.DENSE,
        force_recreate=False,
    )


def normalize_source(source: str | Path) -> str:
    return Path(source).as_posix()


def list_pdf_files() -> list[Path]:
    return sorted(DOCS_DIR.glob("**/*.pdf"))


def get_indexed_sources(client: QdrantClient) -> set[str]:
    if not client.collection_exists(COLLECTION_NAME):
        return set()

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
    return sources


def load_new_documents(pdf_files: list[Path], indexed_sources: set[str]) -> list[Document]:
    docs: list[Document] = []
    skipped = 0

    for pdf_file in pdf_files:
        source = normalize_source(pdf_file)
        if source in indexed_sources:
            skipped += 1
            continue

        file_docs = PyMuPDFLoader(str(pdf_file)).load()
        for doc in file_docs:
            doc.metadata["source"] = source
            doc.metadata["file_path"] = source
        docs.extend(file_docs)

    print(f"Found {len(pdf_files)} PDF files. Skipped {skipped} already indexed files.")
    print(f"Loaded {len(docs)} new pages.")
    return docs


def chunk_id(chunk: Document) -> str:
    source = normalize_source(chunk.metadata.get("source", ""))
    page = str(chunk.metadata.get("page", ""))
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
    raw_id = f"{source}|{page}|{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def split_documents(docs: list[Document]) -> tuple[list[Document], list[str]]:
    if not docs:
        return [], []

    chunking_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    splitter = SemanticChunker(
        chunking_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = splitter.split_documents(docs)
    ids = [chunk_id(chunk) for chunk in chunks]

    for chunk, point_id in zip(chunks, ids, strict=True):
        chunk.id = point_id
        chunk.metadata["chunk_id"] = point_id

    print(f"Split {len(docs)} new pages into {len(chunks)} chunks.")
    return chunks, ids


def filter_existing_chunks(
    client: QdrantClient,
    chunks: list[Document],
    ids: list[str],
) -> tuple[list[Document], list[str]]:
    if not chunks or not client.collection_exists(COLLECTION_NAME):
        return chunks, ids

    existing_ids: set[str] = set()
    for start in range(0, len(ids), 256):
        batch = ids[start : start + 256]
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids.update(str(point.id) for point in points)

    new_pairs = [
        (chunk, point_id)
        for chunk, point_id in zip(chunks, ids, strict=True)
        if point_id not in existing_ids
    ]
    print(f"Skipped {len(chunks) - len(new_pairs)} already indexed chunks.")

    if not new_pairs:
        return [], []
    new_chunks, new_ids = zip(*new_pairs, strict=True)
    return list(new_chunks), list(new_ids)


def ingest_new_documents() -> QdrantVectorStore:
    load_dotenv()

    client = build_client()
    indexed_sources = get_indexed_sources(client)
    docs = load_new_documents(list_pdf_files(), indexed_sources)
    chunks, ids = split_documents(docs)
    chunks, ids = filter_existing_chunks(client, chunks, ids)

    vector_store = build_vector_store()
    if not chunks:
        print("No new chunks to embed.")
        return vector_store

    vector_store.add_documents(chunks, ids=ids, batch_size=BATCH_SIZE)
    print(f"Embedded and saved {len(chunks)} new chunks into '{COLLECTION_NAME}'.")
    return vector_store


def main() -> None:
    ingest_new_documents()


if __name__ == "__main__":
    main()
