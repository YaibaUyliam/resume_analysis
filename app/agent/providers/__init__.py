from .base import ExtractionProvider, EmbeddingProvider
from .ollama import OllamaExtractionProvider, OllamaEmbeddingProvider
# from .huggingface import TorchExtractionProvider


__all__ = [
    "ExtractionProvider",
    "EmbeddingProvider",
    "OllamaExtractionProvider",
    "OllamaEmbeddingProvider",
]
