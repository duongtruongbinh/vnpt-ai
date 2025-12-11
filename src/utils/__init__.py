"""Utility functions for the RAG pipeline."""

from src.utils.checkpointing import (
    append_log_entry,
    consolidate_log_file,
    generate_csv_from_log,
    is_rate_limit_error,
    load_log_entries,
    load_processed_qids,
)
from src.utils.common import normalize_text, remove_diacritics, sort_qids
from src.utils.ingestion import (
    get_embeddings,
    get_qdrant_client,
    get_vector_store,
    ingest_all_data,
    ingest_files,
)
from src.utils.llm import get_large_model, get_small_model
from src.utils.web_crawler import WebCrawler, crawl_website, save_crawled_data

__all__ = [
    # Checkpointing
    "load_processed_qids",
    "load_log_entries",
    "append_log_entry",
    "consolidate_log_file",
    "generate_csv_from_log",
    "is_rate_limit_error",
    "sort_qids",
    # Ingestion
    "get_embeddings",
    "get_qdrant_client",
    "get_vector_store",
    "ingest_all_data",
    "ingest_files",
    # LLM
    "get_small_model",
    "get_large_model",
    # Text utilities
    "normalize_text",
    "remove_diacritics",
    # Web crawler
    "WebCrawler",
    "crawl_website",
    "save_crawled_data",
]
