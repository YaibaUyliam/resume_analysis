import os

from typing import Optional
from dotenv import load_dotenv


if os.environ.get("ENV") != "production":
    load_dotenv("/home/yaiba/project/resume_analysis/.env")

class Settings:
    # which is copied to the user's .env file upon setup.
    PROJECT_NAME: str = "Resume Matcher"
    ENV: str = os.environ.get("ENV")  # debug or production

    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER") # ollama or huggingface
    LL_MODEL: str = os.environ.get("LL_MODEL")
    LL_MODEL_CKPT_PATH: Optional[str] = os.environ.get("LL_MODEL_CKPT_PATH")
    TORCH_DTYPE: Optional[str] = os.environ.get("TORCH_DTYPE")
    USE_VISION: Optional[int] = int(os.environ.get("USE_VISION", 0))

    EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL")


settings = Settings()
