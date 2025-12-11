"""Knowledge base ingestion utilities for Qdrant vector store."""

import json
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
    """Kiểm tra xem đoạn văn bản có phải là rác (nav, footer, ads) không."""
    if len(text.split()) < 5:  # Loại bỏ câu quá ngắn
        return True
    
    text_lower = text.lower()
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


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

        raw_chunks = splitter.split_text(content)
        total_raw_chunks = len(raw_chunks)
        
        for i, chunk in enumerate(raw_chunks):
            if _is_junk_text(chunk):
                continue

            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = total_raw_chunks
            all_chunks.append(chunk)
            all_metadatas.append(chunk_metadata)

    return all_chunks, all_metadatas


def _scan_data_files(base_dir: Path) -> list[Path]:
    """Recursively scan directory for supported files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(base_dir.rglob(f"*{ext}"))
    return sorted(files)


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter with standard settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def _process_and_index_documents(
    files: list[Path],
    vector_store: QdrantVectorStore,
    desc: str = "Processing Files",
) -> tuple[int, int, int]:
    """Process files and add to vector store.

    Args:
        files: List of file paths to process
        vector_store: QdrantVectorStore instance
        desc: Progress bar description

    Returns:
        Tuple of (total_chunks, total_docs, failed_files)
    """
    splitter = _get_text_splitter()
    total_chunks = 0
    total_docs = 0
    failed_files = 0

    with tqdm(total=len(files), desc=desc, unit="file", position=0) as pbar:
        for file_path in files:
            try:
                pbar.set_postfix_str(f"Current: {file_path.name}")
                chunks_to_add, metadatas_to_add = _extract_chunks_from_file(file_path, splitter)

                if chunks_to_add:
                    vector_store.add_texts(
                        chunks_to_add,
                        metadatas=metadatas_to_add,
                        batch_size=len(chunks_to_add),
                    )
                    total_chunks += len(chunks_to_add)
                    total_docs += 1
                    tqdm.write(f"        [Ingest] {file_path.name}: {len(chunks_to_add)} chunks")
                else:
                    tqdm.write(f"        [Warning] {file_path.name}: No content found")

            except Exception as e:
                tqdm.write(f"        [Error] {file_path.name}: {e}")
                failed_files += 1
            finally:
                pbar.update(1)

    return total_chunks, total_docs, failed_files


def _extract_chunks_from_file(
    file_path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[str], list[dict]]:
    """Extract chunks and metadata from a single file.

    Args:
        file_path: Path to the file
        splitter: Text splitter instance

    Returns:
        Tuple of (chunks, metadatas)
    """
    if file_path.suffix.lower() == ".json":
        return _process_crawled_json(file_path)

    text, metadata = load_document(file_path)
    if not text or not metadata:
        return [], []

    chunks = splitter.split_text(text)
    metadatas = []
    for i, _ in enumerate(chunks):
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        chunk_meta["total_chunks"] = len(chunks)
        metadatas.append(chunk_meta)

    return chunks, metadatas


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

    total_chunks, total_docs, failed_files = _process_and_index_documents(files, _vector_store)

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

    total_chunks, _, _ = _process_and_index_documents(
        file_paths,
        vector_store,
        desc="Ingesting Files",
    )

    log_pipeline(f"Total: {total_chunks} chunks in '{collection_name}'")
    return total_chunks
