"""LLM utility functions for hybrid model selection (Local HuggingFace vs VNPT API)."""

from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from src.config import settings

_model_cache: dict[str, BaseChatModel] = {}


class VNPTChatModel(BaseChatModel):
    """LangChain-compatible wrapper for VNPT API."""

    endpoint: str
    model_name: str
    authorization: str  
    token_id: str
    token_key: str
    timeout: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "vnpt-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "endpoint": self.endpoint}

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.authorization,
            "Token-id": self.token_id,
            "Token-key": self.token_key,
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        """Convert LangChain messages to VNPT API format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                converted.append({"role": "user", "content": str(msg.content)})
        return converted

    def _handle_api_response(self, data: dict) -> tuple[str, str | None, dict]:
        """Process API response data and handle safety refusals gracefully."""
        # 1. Handle Errors & Safety Refusals
        if "error" in data:
            error_msg = data.get("error", {})
            error_text = error_msg.get("message", str(error_msg)) if isinstance(error_msg, dict) else str(error_msg)
            
            # Catch VNPT Content Safety Filter
            # Nếu gặp lỗi thuần phong mỹ tục, coi như model trả về "toxic" để Router xử lý
            if "thuần phong mỹ tục" in error_text or "không thể trả lời" in error_text:
                return "toxic", "stop", {}
                
            raise RuntimeError(f"VNPT API returned error: {error_text}")

        # 2. Extract Content
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            # Handle variable response formats
            content = choice.get("message", {}).get("content") or choice.get("text") or choice.get("content")
            if content is None:
                 raise RuntimeError(f"Unexpected format in choices[0]: {list(choice.keys())}")
            finish_reason = choice.get("finish_reason")
            
        elif "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            item = data["data"][0]
            content = item.get("content") or item.get("text", "")
            finish_reason = item.get("finish_reason")
            
        elif "content" in data:
            content = data["content"]
            finish_reason = data.get("finish_reason")
            
        else:
            raise RuntimeError(f"Unexpected VNPT API response keys: {list(data.keys())}")

        return content, finish_reason, data.get("usage", {})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from VNPT API."""
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if stop:
            payload["stop"] = stop

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=self._get_headers(),
                    json=payload,
                )
                # Note: VNPT API might return 200 OK even for safety errors, 
                # or 400/500. We parse JSON first to check 'error' key.
                if response.status_code >= 400:
                    try:
                        data = response.json()
                    except:
                        response.raise_for_status() # Raise raw HTTP error if no JSON
                else:
                    data = response.json()

            content, finish_reason, usage = self._handle_api_response(data)

            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={"finish_reason": finish_reason, "usage": usage},
            )
            return ChatResult(
                generations=[generation],
                llm_output={"model_name": self.model_name, "usage": usage},
            )

        except httpx.RequestError as e:
            raise RuntimeError(f"VNPT API request failed: {e}") from e
        except Exception as e:
            # Re-raise RuntimeErrors from _handle_api_response, wrap others
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"VNPT API processing failed: {e}") from e

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response from VNPT API."""
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if stop:
            payload["stop"] = stop

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint,
                    headers=self._get_headers(),
                    json=payload,
                )
                if response.status_code >= 400:
                    try:
                        data = response.json()
                    except:
                        response.raise_for_status()
                else:
                    data = response.json()

            content, finish_reason, usage = self._handle_api_response(data)

            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={"finish_reason": finish_reason, "usage": usage},
            )
            return ChatResult(
                generations=[generation],
                llm_output={"model_name": self.model_name, "usage": usage},
            )

        except httpx.RequestError as e:
            raise RuntimeError(f"VNPT API request failed: {e}") from e
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"VNPT API processing failed: {e}") from e


def _load_huggingface_model(model_path: str, model_type: str) -> ChatHuggingFace:
    """Load a local HuggingFace model with caching."""
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
        },
    )

    llm = ChatHuggingFace(llm=llm_pipeline)
    _model_cache[model_path] = llm
    print(f"[Model] {model_type} loaded from {model_path}")
    return llm


def _get_vnpt_model(model_type: str) -> VNPTChatModel:
    """Get VNPT API model wrapper with per-model credentials."""
    cache_key = f"vnpt_{model_type}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if model_type == "small":
        authorization = settings.vnpt_small_authorization
        token_id = settings.vnpt_small_token_id
        token_key = settings.vnpt_small_token_key
        endpoint = settings.vnpt_small_endpoint
        model_name = "vnptai_hackathon_small"
    elif model_type == "large":
        authorization = settings.vnpt_large_authorization
        token_id = settings.vnpt_large_token_id
        token_key = settings.vnpt_large_token_key
        endpoint = settings.vnpt_large_endpoint
        model_name = "vnptai_hackathon_large"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not authorization:
        raise ValueError(f"VNPT_{model_type.upper()}_AUTHORIZATION is required")

    model = VNPTChatModel(
        endpoint=endpoint,
        model_name=model_name,
        authorization=authorization,
        token_id=token_id,
        token_key=token_key,
    )

    _model_cache[cache_key] = model
    print(f"[Model] VNPT {model_type} initialized: {endpoint}")
    return model


def get_small_model() -> BaseChatModel:
    """Get or create small LLM (VNPT API or local HuggingFace)."""
    if settings.use_vnpt_api:
        return _get_vnpt_model("small")
    return _load_huggingface_model(settings.llm_model_small, "Small")


def get_large_model() -> BaseChatModel:
    """Get or create large LLM (VNPT API or local HuggingFace)."""
    if settings.use_vnpt_api:
        return _get_vnpt_model("large")
    return _load_huggingface_model(settings.llm_model_large, "Large")