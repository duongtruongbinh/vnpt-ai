"""Embedding models and utilities for vector generation."""

import httpx
import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.config import settings
from src.utils.logging import log_pipeline


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
        """Embed a list of documents with accurate progress bar."""
        if not texts:
            return []
        
        batch_size = 32 
        all_embeddings = []
        
        with tqdm(total=len(texts), desc="Embedding API", unit="chunk", leave=False) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = self._embed(batch)
                all_embeddings.extend(embeddings)
                pbar.update(len(batch))
            
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


_embeddings: Embeddings | None = None
