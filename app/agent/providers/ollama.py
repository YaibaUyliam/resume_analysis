from __future__ import annotations

import json
import time
import ollama
import subprocess
import os
import re

from loguru import logger
from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

from .exceptions import GenerationError
from .base import ExtractionProvider, EmbeddingProvider, remove_image_special


# logger = logging.getLogger(__name__)


def _simplify_model_name(model_name: str) -> str:
    model_name = (model_name or "").lower()
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    return re.sub(r"[^a-z0-9]+", "", model_name)


def _pick_fallback_model(
    requested_model: str, installed_models: list[str], embedding: bool
) -> str:
    if not installed_models:
        raise GenerationError("No Ollama models are installed.")

    simplified_requested = _simplify_model_name(requested_model)
    for installed_model in installed_models:
        simplified_installed = _simplify_model_name(installed_model)
        if (
            simplified_requested
            and simplified_requested in simplified_installed
            or simplified_installed in simplified_requested
        ):
            logger.warning(
                f"Configured model '{requested_model}' not found. "
                f"Using closest installed model '{installed_model}'."
            )
            return installed_model

    if embedding:
        embedding_candidates = [
            model for model in installed_models if "embed" in model.lower()
        ]
        if embedding_candidates:
            fallback_model = embedding_candidates[0]
            logger.warning(
                f"Configured embedding model '{requested_model}' not found. "
                f"Using installed embedding model '{fallback_model}'."
            )
            return fallback_model
    else:
        generation_candidates = [
            model for model in installed_models if "embed" not in model.lower()
        ]
        if generation_candidates:
            fallback_model = generation_candidates[0]
            logger.warning(
                f"Configured generation model '{requested_model}' not found. "
                f"Using installed generation model '{fallback_model}'."
            )
            return fallback_model

    available_models = ", ".join(installed_models)
    raise GenerationError(
        f"Model '{requested_model}' has not been installed. "
        f"Available Ollama models: {available_models}"
    )


class OllamaExtractionProvider(ExtractionProvider):
    def __init__(self, model_name: str, use_vision: bool = False, host: Optional[str] = None):
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
        self.model = _pick_fallback_model(
            model_name, installed_ollama_models, embedding=False
        )

    def _preprocess_data(self, resume_data: bytes | str, prompt: str):
        # converted_data = self.convert_data(resume_data, file_suffix)
        # if not self.use_vision:
        data_input_model = prompt + resume_data

        return data_input_model

    def _postprocess(self, model_res: str):
        result = remove_image_special(model_res["response"].strip())

        try:
            result = json.loads(result)

        except:
            logger.error("Model return wrong json format !!!")
            logger.error(result)
            result = {}

        return result

    def _generate_sync(
        self, resume_data: bytes | str, prompt: str, sys_mess: str
    ) -> str:
        """
        Generate a response from the model.
        """
        time_s = time.time()
        preprocessed_data = self._preprocess_data(resume_data, prompt)
        logger.info(f"Time preprocess data: {time.time()- time_s}")
        logger.info(preprocessed_data)

        sub_result = subprocess.run(
            ["ollama", "stop", os.environ.get("EMBEDDING_MODEL")],
            capture_output=True,
            text=True,
        )

        try:
            if not self.use_vision:
                # logger.info(sys_mess + "\n" + preprocessed_data)

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

            return self._postprocess(response), response

        except Exception as e:
            raise GenerationError(f"Ollama - Error generating response: {e}") from e

    async def __call__(self, resume_data: bytes, prompt: str, sys_mess: str) -> str:
        return await run_in_threadpool(
            self._generate_sync, resume_data, prompt, sys_mess
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
        self._model = _pick_fallback_model(
            model_name, installed_ollama_models, embedding=True
        )

    def _embed_sync(self, input_data: list[str], task: str, query: bool) -> str:
        sub_result = subprocess.run(
            ["ollama", "stop", os.environ.get("LL_MODEL")],
            capture_output=True,
            text=True,
        )

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
