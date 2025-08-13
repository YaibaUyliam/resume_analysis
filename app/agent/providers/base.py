from typing import Any
from abc import ABC, abstractmethod


class ExtractionProvider(ABC):
    """
    Abstract base class for providers.
    """

    @abstractmethod
    async def __call__(self, prompt: str, **generation_args: Any) -> str: ...


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @abstractmethod
    async def __call__(self, text: str) -> list[float]: ...
