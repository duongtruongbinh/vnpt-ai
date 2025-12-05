"""LLM utility functions for loading HuggingFace models with optimized caching."""

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from src.config import settings

_model_cache: dict[str, ChatHuggingFace] = {}

def _load_model(model_path: str, model_type: str) -> ChatHuggingFace:
    """Internal helper to load a HuggingFace model safely."""
    
    if model_path in _model_cache:
        return _model_cache[model_path]

    llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "do_sample": False,
            "return_full_text": False,
        },
        model_kwargs={
            "trust_remote_code": True,
            "device_map": "auto",
        }
    )
    
    llm = ChatHuggingFace(llm=llm_pipeline)
    
    _model_cache[model_path] = llm
    print(f"[Model] {model_type} loaded successfully from {model_path}")
    
    return llm


def get_small_model() -> ChatHuggingFace:
    """Get or create small HuggingFace LLM singleton."""
    return _load_model(settings.llm_model_small, "Small")


def get_large_model() -> ChatHuggingFace:
    """Get or create large HuggingFace LLM singleton."""
    return _load_model(settings.llm_model_large, "Large")