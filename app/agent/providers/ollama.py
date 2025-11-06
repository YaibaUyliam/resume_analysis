import json

import logging
import ollama

from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

from .exceptions import GenerationError
from .base import ExtractionProvider, EmbeddingProvider, remove_image_special


logger = logging.getLogger(__name__)


class OllamaExtractionProvider(ExtractionProvider):
    def __init__(self, model_name: str, use_vision: int, host: Optional[str] = None):
        logger.info("Running model with Ollama ........")
        super().__init__(use_vision)

        self.otps = {
            "temperature": 0,
            "num_ctx": 12288,
            "num_predict": -1,
            "seed": 42,
            "top_k": 1,
            "top_p": 1,
        }
        self.model = model_name
        logger.info(f"Using model {model_name}")
        self._client = ollama.Client(host=host) if host else ollama.Client()

        installed_ollama_models = [
            model_class.model for model_class in self._client.list().models
        ]
        if model_name not in installed_ollama_models:
            raise GenerationError("Model has not installed !!!")

    def _preprocess_data(self, resume_data: bytes|str, prompt: str, file_suffix: str):
        converted_data = self.convert_data(resume_data, file_suffix)
        if not self.use_vision:
            data_input_model = prompt + converted_data

        return data_input_model, converted_data

    def _postprocess(self, model_res: str):
        result = remove_image_special(model_res["response"].strip())

        try:
            result = json.loads(result)

        except:
            result = {}
            logger.error("Model return wrong json format !!!")

        return result

    def _generate_sync(
        self, resume_data: bytes|str, prompt: str, sys_mess: str, file_suffix: str
    ) -> str:
        """
        Generate a response from the model.
        """
        preprocessed_data, original_data = self._preprocess_data(resume_data, prompt, file_suffix)

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
                    prompt=prompt,
                    model=self.model,
                    options=self.otps,
                    images=preprocessed_data,
                )

            # logger.info(response["response"].strip())

            return self._postprocess(response), original_data

        except Exception as e:
            raise GenerationError(f"Ollama - Error generating response: {e}") from e

    async def __call__(
        self, resume_data: bytes, prompt: str, sys_mess: str, file_suffix: str
    ) -> str:
        return await run_in_threadpool(
            self._generate_sync, resume_data, prompt, sys_mess, file_suffix
        )


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str,
        host: Optional[str] = None,
    ):
        super().__init__()

        self.otps = {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": -1,
            "seed": 42,
            "top_k": 1,
            "top_p": 1,
        }
        self._model = model_name
        logger.info(f"Using model {model_name}")
        self._client = ollama.Client(host=host) if host else ollama.Client()

        installed_ollama_models = [
            model_class.model for model_class in self._client.list().models
        ]
        if self._model not in installed_ollama_models:
            raise GenerationError("Model has not installed !!!")

    def _embed_sync(self, input_data: list[str], task: str, query: bool) -> str:
        preprocessed_data = []

        if query:
            for data in input_data:
                # Qwen3 have instruct
                preprocessed_data.append(f"Instruct: {task}\nQuery: {data}")
        else:
            preprocessed_data = input_data

        logger.info(preprocessed_data)
        try:
            response = self._client.embed(
                input=preprocessed_data,
                model=self._model,
                truncate=True,
                dimensions=1024,
            )

            return response

        except Exception as e:
            raise GenerationError(f"Ollama - Error generating response: {e}") from e

    async def __call__(self, input_data: str, task, query: bool = False) -> List[float]:
        response = await run_in_threadpool(self._embed_sync, input_data, task, query)
        return response.embeddings


if __name__ == "__main__":
    import asyncio
    import numpy as np

    client = OllamaEmbeddingProvider(model_name="qwen3-embedding:0.6b-fp16")
    resume_data = ["What is the capital of China?", "Explain gravity"]

    res = asyncio.run(client(resume_data, True))
    print(np.array(res).shape)
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    res_1 = asyncio.run(client(documents, False))

    scores = np.array(res) @ np.array(res_1).T
    print(scores.tolist())
