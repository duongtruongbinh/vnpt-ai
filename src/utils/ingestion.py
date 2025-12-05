"""Knowledge base ingestion utilities for Qdrant vector store."""

import json
import re
import unicodedata
from pathlib import Path

import httpx
import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import DATA_INPUT_DIR, settings
from src.utils.logging import log_pipeline, print_log

SUPPORTED_EXTENSIONS = {".json", ".pdf", ".docx", ".txt"}

_embeddings: Embeddings | None = None
_qdrant_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


class VNPTEmbeddings(Embeddings):
    """LangChain-compatible wrapper for VNPT Embedding API."""

    def __init__(
        self,
        endpoint: str,
        authorization: str,
        token_id: str,
        token_key: str,
        model_name: str = "vnptai_hackathon_embedding",
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.authorization = authorization
        self.token_id = token_id
        self.token_key = token_key
        self.model_name = model_name
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.authorization,
            "Token-id": self.token_id,
            "Token-key": self.token_key,
            "Content-Type": "application/json",
        }

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call VNPT API to get embeddings."""
        payload = {"model": self.model_name, "input": texts}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return [item["embedding"] for item in data["data"]]

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"VNPT Embedding API error ({e.response.status_code}): {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"VNPT Embedding API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected VNPT Embedding API response: {e}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._embed([text])[0]


def get_device() -> str:
    """Detect optimal device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_text(text: str) -> str:
    """Normalize text for ingestion: clean whitespace, unicode, and formatting."""
    if not text:
        return ""
    # Unicode NFKC normalization (composing characters)
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Normalize whitespace: replace multiple spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize newlines: max 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def get_embeddings() -> Embeddings:
    """Get or create embeddings model singleton (VNPT API or local HuggingFace)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if settings.use_vnpt_api:
        if not settings.vnpt_embedding_authorization:
            raise ValueError("VNPT_EMBEDDING_AUTHORIZATION is required when USE_VNPT_API=True")
        _embeddings = VNPTEmbeddings(
            endpoint=settings.vnpt_embedding_endpoint,
            authorization=settings.vnpt_embedding_authorization,
            token_id=settings.vnpt_embedding_token_id,
            token_key=settings.vnpt_embedding_token_key,
        )
        log_pipeline(f"VNPT Embedding API initialized: {settings.vnpt_embedding_endpoint}")
    else:
        device = get_device()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        log_pipeline(f"HuggingFace Embedding loaded: {settings.embedding_model}")

    return _embeddings


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


def _load_pdf(file_path: Path) -> str:
    """Load text from PDF file."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF files. Install with: pip install pypdf")

    reader = pypdf.PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def _load_docx(file_path: Path) -> str:
    """Load text from DOCX file."""
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")

    doc = docx.Document(str(file_path))
    return "\n".join([para.text for para in doc.paragraphs])


def _load_txt(file_path: Path) -> str:
    """Load text from TXT file."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _load_document(file_path: Path) -> tuple[str | None, dict | None]:
    """Load document (PDF, DOCX, TXT), normalize text, and return (text, metadata).

    Returns (None, None) for unsupported or failed files.
    """
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            text = _load_pdf(file_path)
        elif ext == ".docx":
            text = _load_docx(file_path)
        elif ext == ".txt":
            text = _load_txt(file_path)
        else:
            return None, None

        text = normalize_text(text)
        if not text:
            return None, None

        metadata = {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_type": ext[1:],
        }
        return text, metadata

    except Exception as e:
        print_log(f"        [Error] Failed to load {file_path.name}: {e}")
        return None, None


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
        base_dir: Directory to scan (default: DATA_INPUT_DIR)
        force: If True, wipe collection and re-ingest everything

    Returns:
        QdrantVectorStore instance
    """
    global _vector_store

    base_dir = base_dir or DATA_INPUT_DIR
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

    # Initialize collection
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

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".json":
                chunks, metadatas = _process_crawled_json(file_path)
                if chunks:
                    _vector_store.add_texts(chunks, metadatas=metadatas)
                    total_chunks += len(chunks)
                    total_docs += 1
                    print_log(f"        [Ingest] {file_path.name}: {len(chunks)} chunks")
                else:
                    print_log(f"        [Warning] {file_path.name}: No content found")
            else:
                text, metadata = _load_document(file_path)
                if text and metadata:
                    chunks = splitter.split_text(text)
                    metadatas = []
                    for i, chunk in enumerate(chunks):
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = i
                        chunk_meta["total_chunks"] = len(chunks)
                        metadatas.append(chunk_meta)

                    _vector_store.add_texts(chunks, metadatas=metadatas)
                    total_chunks += len(chunks)
                    total_docs += 1
                    print_log(f"        [Ingest] {file_path.name}: {len(chunks)} chunks")

        except Exception as e:
            print_log(f"        [Error] {file_path.name}: {e}")
            failed_files += 1
            continue

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

    for file_path in file_paths:
        try:
            if file_path.suffix.lower() == ".json":
                chunks, metadatas = _process_crawled_json(file_path)
                if chunks:
                    vector_store.add_texts(chunks, metadatas=metadatas)
                    total_chunks += len(chunks)
                    print(f"[Ingest] {file_path.name}: {len(chunks)} chunks")
            else:
                text, metadata = _load_document(file_path)
                if text and metadata:
                    chunks = splitter.split_text(text)
                    metadatas = []
                    for i, chunk in enumerate(chunks):
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = i
                        chunk_meta["total_chunks"] = len(chunks)
                        metadatas.append(chunk_meta)

                    vector_store.add_texts(chunks, metadatas=metadatas)
                    total_chunks += len(chunks)
                    print(f"[Ingest] {file_path.name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"[Error] {file_path.name}: {e}")
            continue

    print(f"[Done] Total: {total_chunks} chunks in '{collection_name}'")
    return total_chunks
