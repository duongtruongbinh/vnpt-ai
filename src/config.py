import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_INPUT_DIR = Path(os.getenv("DATA_INPUT_DIR", PROJECT_ROOT / "test_data"))
DATA_OUTPUT_DIR = Path(os.getenv("DATA_OUTPUT_DIR", PROJECT_ROOT / "output"))
DATA_CRAWLED_DIR = Path(os.getenv("DATA_CRAWLED_DIR", DATA_DIR / "crawled"))
BATCH_SIZE = 1


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    use_vnpt_api: bool = Field(
        default=False,
        alias="USE_VNPT_API",
        description="If True, use VNPT API; otherwise use local HuggingFace models",
    )

    vnpt_large_authorization: str = Field(
        default="",
        alias="VNPT_LARGE_AUTHORIZATION",
        description="Full Authorization header value (including 'Bearer ') for large model",
    )
    vnpt_large_token_id: str = Field(
        default="",
        alias="VNPT_LARGE_TOKEN_ID",
    )
    vnpt_large_token_key: str = Field(
        default="",
        alias="VNPT_LARGE_TOKEN_KEY",
    )

    # VNPT API - Small Model credentials
    vnpt_small_authorization: str = Field(
        default="",
        alias="VNPT_SMALL_AUTHORIZATION",
        description="Full Authorization header value (including 'Bearer ') for small model",
    )
    vnpt_small_token_id: str = Field(
        default="",
        alias="VNPT_SMALL_TOKEN_ID",
    )
    vnpt_small_token_key: str = Field(
        default="",
        alias="VNPT_SMALL_TOKEN_KEY",
    )

    # VNPT API - Embedding Model credentials
    vnpt_embedding_authorization: str = Field(
        default="",
        alias="VNPT_EMBEDDING_AUTHORIZATION",
        description="Full Authorization header value (including 'Bearer ') for embedding model",
    )
    vnpt_embedding_token_id: str = Field(
        default="",
        alias="VNPT_EMBEDDING_TOKEN_ID",
    )
    vnpt_embedding_token_key: str = Field(
        default="",
        alias="VNPT_EMBEDDING_TOKEN_KEY",
    )

    # VNPT API endpoints
    vnpt_small_endpoint: str = Field(
        default="https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small",
        alias="VNPT_SMALL_ENDPOINT",
    )
    vnpt_large_endpoint: str = Field(
        default="https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large",
        alias="VNPT_LARGE_ENDPOINT",
    )
    vnpt_embedding_endpoint: str = Field(
        default="https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding",
        alias="VNPT_EMBEDDING_ENDPOINT",
    )

    # Local HuggingFace models
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

    # Vector database
    qdrant_collection: str = Field(
        default="vnpt_knowledge_base",
        alias="QDRANT_COLLECTION",
    )
    vector_db_path: str = Field(
        default="",
        alias="VECTOR_DB_PATH",
    )

    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k_retrieval: int = 3

    @property
    def vector_db_path_resolved(self) -> Path:
        """Resolve vector database path, defaulting to DATA_DIR/qdrant_storage."""
        if self.vector_db_path:
            return Path(self.vector_db_path)
        return DATA_DIR / "qdrant_storage"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
