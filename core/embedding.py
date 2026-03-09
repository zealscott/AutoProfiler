from typing import Any, List, Optional

import litellm
from llama_index.core.embeddings import BaseEmbedding


class LiteLLMEmbedding(BaseEmbedding):
    """LlamaIndex-compatible embedding class using litellm.embedding().

    Allows any LiteLLM-supported embedding provider (OpenAI, Gemini, Cohere, etc.)
    to be used for RAG with a unified interface.
    """

    model_name: str = "openai/text-embedding-3-small"
    api_key: Optional[str] = None

    def __init__(self, model_name: str = "openai/text-embedding-3-small", api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)

    class Config:
        arbitrary_types_allowed = True

    def _get_text_embedding(self, text: str) -> List[float]:
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = litellm.embedding(model=self.model_name, input=[text], **kwargs)
        return response.data[0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
