import os

from .providers.base import Provider
from ..core import settings


class GenerationManager:
    def __init__(self):
        self.model_provider = settings.LLM_PROVIDER
        self.model_name = settings.LL_MODEL
        self.model_path = settings.LL_MODEL_CKPT_PATH
        self.torch_dtype = settings.TORCH_DTYPE

        self._init_model()

    async def _init_model(self) -> Provider:
        otps = {}

        if self.model_provider == "ollama":
            from .providers import OllamaExtractionProvider

            self.model = OllamaExtractionProvider(model_name=self.model_name, opts=otps)
        else:
            from .providers import TorchExtractionProvider

            self.model = TorchExtractionProvider(self.model_path, self.torch_dtype)

    async def __call__(self, *args, **kwds):
        return self.model
