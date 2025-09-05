import json

import logging
import ollama

from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

from .exceptions import GenerationError
from .base import ExtractionProvider, EmbeddingProvider, remove_image_special
from .prompt import PROMPT, SYSTEM


logger = logging.getLogger(__name__)


class OllamaExtractionProvider(ExtractionProvider):
    def __init__(self, model_name: str, use_vision: int, host: Optional[str] = None):
        logger.info("Running model with Ollama ........")
        super().__init__(use_vision)

        self.otps = {"temperature": 0, "num_ctx": 12000, "num_predict": -1}  # 16384
        self.model = model_name
        self._client = ollama.Client(host=host) if host else ollama.Client()

        installed_ollama_models = [
            model_class.model for model_class in self._client.list().models
        ]
        if model_name not in installed_ollama_models:
            raise GenerationError("Model has not installed !!!")

    def _preprocess_data(self, resume_data: bytes, prompt: str, file_suffix: str):
        converted_data = self.convert_data(resume_data, file_suffix)
        if not self.use_vision:
            converted_data = prompt + converted_data

        return converted_data

    def _postprocess(self, model_res: str):
        result = remove_image_special(model_res["response"].strip())

        try:
            result = json.loads(result)

        except:
            result = {}
            logger.error("Model return wrong json format !!!")

        return result

    def _generate_sync(
        self, resume_data: bytes, sys_mess: Optional[str], file_suffix: str
    ) -> str:
        """
        Generate a response from the model.
        """
        if not sys_mess:
            sys_mess = SYSTEM

        preprocessed_data = self._preprocess_data(resume_data, PROMPT, file_suffix)

        try:
            if not self.use_vision:
                logger.info(sys_mess + "\n" + preprocessed_data)

                response = self._client.generate(
                    system=sys_mess,
                    prompt=preprocessed_data,
                    model=self.model,
                    options=self.otps,
                    # think=True,
                )
            else:
                response = self._client.generate(
                    system=sys_mess,
                    prompt=PROMPT,
                    model=self.model,
                    options=self.otps,
                    images=preprocessed_data,
                )

            logger.info(response["response"].strip())

            return self._postprocess(response)

        except Exception as e:
            raise GenerationError(f"Ollama - Error generating response: {e}") from e

    async def __call__(
        self, resume_data: bytes, prompt: Optional[str], file_suffix: str
    ) -> str:
        return await run_in_threadpool(
            self._generate_sync, resume_data, prompt, file_suffix
        )


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
