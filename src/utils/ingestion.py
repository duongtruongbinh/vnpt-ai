"""Knowledge base ingestion utilities for Qdrant vector store."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

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
INGESTION_BATCH_SIZE = 100
MAX_WORKERS = 4

JUNK_PATTERNS = [
    r"đăng nhập", r"đăng ký", r"quên mật khẩu", r"chia sẻ qua email", 
    r"bản quyền thuộc", r"liên hệ quảng cáo", r"về đầu trang", 
    r"xem thêm", r"bình luận", r"báo xấu", r"trang chủ", 
    r"facebook", r"twitter", r"linkedin", r"zalo", 
    r"kết nối với chúng tôi", r"thông tin tòa soạn",
    r"wikipedia", r"bách khoa toàn thư", r"sửa đổi", r"biểu quyết",
]

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


def _is_junk_text(text: str) -> bool:
    """Check if text is junk (nav, footer, ads)."""
    if len(text.split()) < 5:
        return True
    text_lower = text.lower()
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _prepend_title_to_chunk(chunk_text: str, title: str) -> str:
    """Prepend title to chunk content for better context in embeddings."""
    if title and title.strip():
        return f"Title: {title.strip()}\nContent: {chunk_text}"
    return chunk_text


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


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with standard settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def _process_crawled_json(json_path: Path) -> tuple[list[str], list[dict]]:
    """Process crawled JSON file, normalize content, and return (chunks, metadatas)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    if not documents:
        return [], []

    splitter = _get_text_splitter()
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        content = normalize_text(doc.get("content", ""))
        if not content:
            continue

        title = normalize_text(doc.get("title", ""))
        keywords_raw = doc.get("keywords")
        keywords_str = ""
        if isinstance(keywords_raw, list):
            keywords_str = ",".join([str(k) for k in keywords_raw if k])
        elif isinstance(keywords_raw, str):
            keywords_str = keywords_raw

        base_metadata = {
            "source_url": doc.get("url", ""),
            "title": title,
            "summary": normalize_text(doc.get("summary", "")),
            "topic": data.get("topic", ""),
            "keywords": keywords_str,
            "domain": data.get("domain", ""),
            "source_file": str(json_path),
        }

        raw_chunks = splitter.split_text(content)
        total_raw_chunks = len(raw_chunks)
        
        for i, chunk in enumerate(raw_chunks):
            if _is_junk_text(chunk):
                continue
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = total_raw_chunks
            chunk_with_title = _prepend_title_to_chunk(chunk, title)
            all_chunks.append(chunk_with_title)
            all_metadatas.append(chunk_metadata)

    return all_chunks, all_metadatas


def _scan_data_files(base_dir: Path) -> list[Path]:
    """Recursively scan directory for supported files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(base_dir.rglob(f"*{ext}"))
    return sorted(files)


def _extract_chunks_from_file(
    file_path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[str], list[dict], str | None]:
    """Extract chunks and metadata from a single file.

    Returns:
        Tuple of (chunks, metadatas, error_message)
    """
    try:
        if file_path.suffix.lower() == ".json":
            chunks, metas = _process_crawled_json(file_path)
            return chunks, metas, None

        text, metadata = load_document(file_path)
        if not text or not metadata:
            return [], [], None

        title = metadata.get("title", "")
        chunks = splitter.split_text(text)
        processed_chunks = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_with_title = _prepend_title_to_chunk(chunk, title)
            processed_chunks.append(chunk_with_title)
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            metadatas.append(chunk_meta)

        return processed_chunks, metadatas, None
    except Exception as e:
        return [], [], str(e)


def _parse_files_parallel(
    files: list[Path],
    max_workers: int = MAX_WORKERS,
) -> tuple[list[str], list[dict], int, int, int]:
    """Parse all files in parallel using ThreadPoolExecutor.

    Returns:
        Tuple of (all_chunks, all_metadatas, total_docs, failed_files, skipped_files)
    """
    splitter = _get_text_splitter()
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    total_docs = 0
    failed_files = 0
    skipped_files = 0

    log_pipeline(f"Parsing {len(files)} files with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_extract_chunks_from_file, f, splitter): f 
            for f in files
        }
        
        with tqdm(total=len(files), desc="Parsing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                pbar.set_postfix_str(f"{file_path.name}")
                
                chunks, metas, error = future.result()
                
                if error:
                    tqdm.write(f"        [Error] {file_path.name}: {error}")
                    failed_files += 1
                elif chunks:
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metas)
                    total_docs += 1
                else:
                    skipped_files += 1
                
                pbar.update(1)

    log_pipeline(f"Parsed {total_docs} files -> {len(all_chunks)} chunks")
    if failed_files > 0:
        log_pipeline(f"Failed: {failed_files} files")
    if skipped_files > 0:
        log_pipeline(f"Skipped (empty): {skipped_files} files")

    return all_chunks, all_metadatas, total_docs, failed_files, skipped_files


def _ingest_chunks_batched(
    chunks: list[str],
    metadatas: list[dict],
    vector_store: QdrantVectorStore,
    batch_size: int = INGESTION_BATCH_SIZE,
) -> int:
    """Ingest chunks into vector store in large batches.

    Returns:
        Number of chunks ingested
    """
    if not chunks:
        return 0

    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size
    
    log_pipeline(f"Ingesting {total} chunks in {num_batches} batches (batch_size={batch_size})...")

    ingested = 0
    with tqdm(total=num_batches, desc="Ingesting batches", unit="batch") as pbar:
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            
            vector_store.add_texts(
                batch_chunks,
                metadatas=batch_metas,
                batch_size=len(batch_chunks),
            )
            ingested += len(batch_chunks)
            pbar.set_postfix_str(f"{ingested}/{total} chunks")
            pbar.update(1)

    return ingested


def _process_and_index_documents(
    files: list[Path],
    vector_store: QdrantVectorStore,
    batch_size: int = INGESTION_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
) -> tuple[int, int, int]:
    """Process files in parallel and ingest in batched mode.

    Returns:
        Tuple of (total_chunks, total_docs, failed_files)
    """
    all_chunks, all_metadatas, total_docs, failed_files, _ = _parse_files_parallel(
        files, max_workers=max_workers
    )

    if not all_chunks:
        return 0, 0, failed_files

    total_chunks = _ingest_chunks_batched(
        all_chunks, all_metadatas, vector_store, batch_size=batch_size
    )

    return total_chunks, total_docs, failed_files


def ingest_all_data(
    base_dir: Path | None = None,
    force: bool = False,
    batch_size: int = INGESTION_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
) -> QdrantVectorStore:
    """Ingest all data from crawled JSON and documents into Qdrant.

    Uses parallel file parsing and batched ingestion for speed.

    Args:
        base_dir: Directory to scan (default: DATA_DIR)
        force: If True, wipe collection and re-ingest everything
        batch_size: Number of chunks per batch for vector store insertion
        max_workers: Number of parallel workers for file parsing

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
    sample_embedding = embeddings.embed_query("test")
    _initialize_collection(client, collection_name, len(sample_embedding), force_recreate=force)

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    if not files:
        log_pipeline(f"No supported files found in {base_dir}")
        return _vector_store

    log_pipeline(f"Found {len(files)} files to ingest from {base_dir}")

    total_chunks, total_docs, failed_files = _process_and_index_documents(
        files, _vector_store, batch_size=batch_size, max_workers=max_workers
    )

    log_pipeline(f"Ingestion complete: {total_docs} files, {total_chunks} chunks")
    if failed_files > 0:
        log_pipeline(f"Failed files: {failed_files}")
    log_pipeline(f"Collection: '{collection_name}'")

    return _vector_store


def ingest_files(
    file_paths: list[Path],
    collection_name: str | None = None,
    append: bool = False,
    batch_size: int = INGESTION_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
) -> int:
    """Ingest specific files into Qdrant with parallel parsing and batched insertion.

    Args:
        file_paths: List of file paths to ingest
        collection_name: Optional collection name (default from settings)
        append: If True, append to existing collection; otherwise recreate
        batch_size: Number of chunks per batch for vector store insertion
        max_workers: Number of parallel workers for file parsing

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

    total_chunks, _, _ = _process_and_index_documents(
        file_paths, vector_store, batch_size=batch_size, max_workers=max_workers
    )

    log_pipeline(f"Total: {total_chunks} chunks in '{collection_name}'")
    return total_chunks