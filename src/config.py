"""Configuration settings for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_INPUT_DIR = Path(os.getenv("DATA_INPUT_DIR", PROJECT_ROOT / "data"))
DATA_OUTPUT_DIR = Path(os.getenv("DATA_OUTPUT_DIR", PROJECT_ROOT / "data"))
BATCH_SIZE = 1

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    llm_model_small: str = Field(
        default="/mnt/dataset1/pretrained_fm/Qwen_Qwen3-4B-Instruct-2507",
        alias="LLM_MODEL_SMALL",
    )
    llm_model_large: str = Field(
        default="/mnt/dataset1/pretrained_fm/Qwen_Qwen3-4B-Instruct-2507",
        alias="LLM_MODEL_LARGE",
    )
    embedding_model: str = Field(
        default="bkai-foundation-models/vietnamese-bi-encoder",
        alias="EMBEDDING_MODEL",
    )
    qdrant_collection: str = Field(
        default="vnpt_knowledge_base",
        alias="QDRANT_COLLECTION",
    )
    vector_db_path: str = Field(
        default="",
        alias="VECTOR_DB_PATH",
    )
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k_retrieval: int = 3

    @property
    def vector_db_path_resolved(self) -> Path:
        """Resolve vector database path, defaulting to DATA_OUTPUT_DIR/qdrant_storage."""
        if self.vector_db_path:
            return Path(self.vector_db_path)
        return DATA_OUTPUT_DIR / "qdrant_storage"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

