import os

from .providers.base import ExtractionProvider
from ..core import settings


class GenerationManager:
    def __init__(self):
        self.model_provider = settings.LLM_PROVIDER
        self.model_name = settings.LL_MODEL
        self.model_path = settings.LL_MODEL_CKPT_PATH
        self.torch_dtype = settings.TORCH_DTYPE
        self.use_vision = settings.USE_VISION

    async def init_model(self) -> ExtractionProvider:
        otps = {}

        if self.model_provider == "ollama":
            from .providers import OllamaExtractionProvider

            return OllamaExtractionProvider(self.model_name, self.use_vision)
        # else:
        #     from .providers import TorchExtractionProvider

        #     return TorchExtractionProvider(self.model_path, self.torch_dtype, self.use_vision)
