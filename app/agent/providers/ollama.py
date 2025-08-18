import logging
import ollama

from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

from .exceptions import GenerationError
from .base import ExtractionProvider, EmbeddingProvider


logger = logging.getLogger(__name__)


class OllamaExtractionProvider(ExtractionProvider):
    def __init__(
        self, model_name: str, host: Optional[str] = None, opts: Dict[str, Any] = None
    ):
        logger.info("Running model with HugingFace ........")

        if opts is None:
            opts = {}
        self.opts = opts
        self.model = model_name
        self._client = ollama.Client(host=host) if host else ollama.Client()

        installed_ollama_models = [
            model_class.model for model_class in self._client.list().models
        ]
        if model_name not in installed_ollama_models:
            raise GenerationError("Model has not installed !!!")
    
    def _preprocess_data(self, resume_data: bytes, prompt: Optional[str]=None):
        pass

    def _generate_sync(self, prompt: str, options: Dict[str, Any]) -> str:
        """
        Generate a response from the model.
        """
        try:
            response = self._client.generate(
                prompt=prompt,
                model=self.model,
                options=options,
            )
            return response["response"].strip()
        except Exception as e:
            raise GenerationError(f"Ollama - Error generating response: {e}") from e

    async def __call__(self, resume_data: bytes, prompt: Optional[str]=None) -> str:
        return await run_in_threadpool(self._generate_sync, resume_data, prompt)


# class OllamaEmbeddingProvider(EmbeddingProvider):
#     def __init__(
#         self,
#         embedding_model: str = settings.EMBEDDING_MODEL,
#         host: Optional[str] = None,
#     ):
#         self._model = embedding_model
#         self._client = ollama.Client(host=host) if host else ollama.Client()

#     async def embed(self, text: str) -> List[float]:
#         """
#         Generate an embedding for the given text.
#         """
#         try:
#             response = await run_in_threadpool(
#                 self._client.embed,
#                 input=text,
#                 model=self._model,
#             )
#             return response.embeddings
#         except Exception as e:
#             logger.error(f"ollama embedding error: {e}")
#             raise ProviderError(f"Ollama - Error generating embedding: {e}") from e
