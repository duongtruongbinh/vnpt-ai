"""Knowledge base ingestion utilities for Qdrant vector store."""

import json
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

from src.config import DATA_DIR, settings
from src.utils.common import normalize_text
from src.utils.doc_parsers import load_document
from src.utils.embeddings import get_embeddings
from src.utils.logging import log_pipeline

SUPPORTED_EXTENSIONS = {".json", ".pdf", ".docx", ".txt"}

_qdrant_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create persistent Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        db_path = settings.vector_db_path_resolved
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(db_path))
    return _qdrant_client


def get_vector_store() -> QdrantVectorStore:
    """Get the global vector store instance (Lazy load)."""
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        embeddings = get_embeddings()
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
    return _vector_store


def _initialize_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    force_recreate: bool = False,
) -> None:
    """Initialize Qdrant collection, creating if needed."""
    collection_exists = client.collection_exists(collection_name)

    if collection_exists and force_recreate:
        client.delete_collection(collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _process_crawled_json(json_path: Path) -> tuple[list[str], list[dict]]:
    """Process crawled JSON file, normalize content, and return (chunks, metadatas)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        content = normalize_text(doc.get("content", ""))
        if not content:
            continue

        keywords_raw = doc.get("keywords")
        keywords_str = ""
        if isinstance(keywords_raw, list):
            keywords_str = ",".join([str(k) for k in keywords_raw if k])
        elif isinstance(keywords_raw, str):
            keywords_str = keywords_raw

        base_metadata = {
            "source_url": doc.get("url", ""),
            "title": normalize_text(doc.get("title", "")),
            "summary": normalize_text(doc.get("summary", "")),
            "topic": data.get("topic", ""),
            "keywords": keywords_str,
            "domain": data.get("domain", ""),
            "source_file": str(json_path),
        }

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            all_chunks.append(chunk)
            all_metadatas.append(chunk_metadata)

    return all_chunks, all_metadatas


def _scan_data_files(base_dir: Path) -> list[Path]:
    """Recursively scan directory for supported files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(base_dir.rglob(f"*{ext}"))
    return sorted(files)


def ingest_all_data(
    base_dir: Path | None = None,
    force: bool = False,
) -> QdrantVectorStore:
    """Ingest all data from crawled JSON and documents into Qdrant.

    Recursively scans base_dir for JSON, PDF, DOCX, and TXT files.

    Args:
        base_dir: Directory to scan (default: DATA_DIR)
        force: If True, wipe collection and re-ingest everything

    Returns:
        QdrantVectorStore instance
    """
    global _vector_store

    base_dir = base_dir or DATA_DIR 
    embeddings = get_embeddings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection

    collection_exists = client.collection_exists(collection_name)

    if collection_exists and not force:
        log_pipeline(f"Loading existing vector store: {settings.vector_db_path_resolved}")
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return _vector_store

    if force and collection_exists:
        log_pipeline(f"Force re-ingesting: deleting collection '{collection_name}'")

    files = _scan_data_files(base_dir)
    if not files:
        log_pipeline(f"No supported files found in {base_dir}")
        sample_embedding = embeddings.embed_query("test")
        _initialize_collection(client, collection_name, len(sample_embedding), force_recreate=force)
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return _vector_store

    log_pipeline(f"Found {len(files)} files to ingest from {base_dir}")

    sample_embedding = embeddings.embed_query("test")
    _initialize_collection(client, collection_name, len(sample_embedding), force_recreate=force)

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    total_chunks = 0
    total_docs = 0
    failed_files = 0

    with tqdm(total=len(files), desc="Processing Files", unit="file", position=0) as pbar:
        for file_path in files:
            try:
                pbar.set_postfix_str(f"Current: {file_path.name}")
                
                chunks_to_add = []
                metadatas_to_add = []

                if file_path.suffix.lower() == ".json":
                    chunks, metadatas = _process_crawled_json(file_path)
                    if chunks:
                        chunks_to_add = chunks
                        metadatas_to_add = metadatas
                    else:
                        tqdm.write(f"        [Warning] {file_path.name}: No content found")
                else:
                    text, metadata = load_document(file_path)
                    if text and metadata:
                        chunks = splitter.split_text(text)
                        metadatas = []
                        for i, chunk in enumerate(chunks):
                            chunk_meta = metadata.copy()
                            chunk_meta["chunk_index"] = i
                            chunk_meta["total_chunks"] = len(chunks)
                            metadatas.append(chunk_meta)
                        
                        chunks_to_add = chunks
                        metadatas_to_add = metadatas

                if chunks_to_add:
                    _vector_store.add_texts(
                        chunks_to_add, 
                        metadatas=metadatas_to_add,
                        batch_size=len(chunks_to_add) 
                    )
                    total_chunks += len(chunks_to_add)
                    total_docs += 1
                    tqdm.write(f"        [Ingest] {file_path.name}: {len(chunks_to_add)} chunks")

            except Exception as e:
                tqdm.write(f"        [Error] {file_path.name}: {e}")
                failed_files += 1
            finally:
                pbar.update(1)

    log_pipeline(f"Ingestion complete: {total_docs} files, {total_chunks} chunks")
    if failed_files > 0:
        log_pipeline(f"Failed files: {failed_files}")
    log_pipeline(f"Collection: '{collection_name}'")

    return _vector_store


def ingest_files(
    file_paths: list[Path],
    collection_name: str | None = None,
    append: bool = False,
) -> int:
    """Ingest specific files into Qdrant.

    Args:
        file_paths: List of file paths to ingest
        collection_name: Optional collection name (default from settings)
        append: If True, append to existing collection; otherwise recreate

    Returns:
        Number of chunks ingested
    """
    collection_name = collection_name or settings.qdrant_collection
    embeddings = get_embeddings()
    client = get_qdrant_client()

    sample_embedding = embeddings.embed_query("test")
    _initialize_collection(client, collection_name, len(sample_embedding), force_recreate=not append)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    total_chunks = 0

    with tqdm(total=len(file_paths), desc="Ingesting Files", unit="file") as pbar:
        for file_path in file_paths:
            try:
                pbar.set_postfix_str(f"Current: {file_path.name}")
                chunks_to_add = []
                metadatas_to_add = []

                if file_path.suffix.lower() == ".json":
                    chunks, metadatas = _process_crawled_json(file_path)
                    if chunks:
                        chunks_to_add = chunks
                        metadatas_to_add = metadatas
                else:
                    text, metadata = load_document(file_path)
                    if text and metadata:
                        chunks = splitter.split_text(text)
                        metadatas = []
                        for i, chunk in enumerate(chunks):
                            chunk_meta = metadata.copy()
                            chunk_meta["chunk_index"] = i
                            chunk_meta["total_chunks"] = len(chunks)
                            metadatas.append(chunk_meta)
                        chunks_to_add = chunks
                        metadatas_to_add = metadatas

                if chunks_to_add:
                    vector_store.add_texts(
                        chunks_to_add, 
                        metadatas=metadatas_to_add,
                        batch_size=len(chunks_to_add)
                    )
                    total_chunks += len(chunks_to_add)
                    tqdm.write(f"[Ingest] {file_path.name}: {len(chunks_to_add)} chunks")

            except Exception as e:
                tqdm.write(f"[Error] {file_path.name}: {e}")
                continue
            finally:
                pbar.update(1)

    print(f"[Done] Total: {total_chunks} chunks in '{collection_name}'")
    return total_chunks
