import os

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


if os.environ.get("APP_ENV") != "production":
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / ".env"
    dotenv_sample_path = project_root / ".env.sample"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    elif dotenv_sample_path.exists():
        load_dotenv(dotenv_sample_path)


class Settings:
    # which is copied to the user's .env file upon setup.
    PROJECT_NAME: str = "Resume Matcher"
    # ENV: str = os.environ.get("APP_ENV")  # debug or production

    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER")  # ollama or huggingface
    LL_MODEL: str = os.environ.get("LL_MODEL")
    LL_MODEL_CKPT_PATH: Optional[str] = os.environ.get("LL_MODEL_CKPT_PATH")
    TORCH_DTYPE: Optional[str] = os.environ.get("TORCH_DTYPE")
    USE_VISION: Optional[int] = int(os.environ.get("USE_VISION", 0))

    EMBEDDING_PROVIDER: str = os.environ.get("EMBEDDING_PROVIDER")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL")
    ES_HOST: str = os.environ.get("ES_HOST", "http://127.0.0.1:9200")
    ES_CV_INDEX: str = os.environ.get("ES_CV_INDEX", "resume-cv-index")
    ES_JD_INDEX: str = os.environ.get("ES_JD_INDEX", "resume-jd-index")
    ES_SEARCH_RESULT_INDEX: str = os.environ.get(
        "ES_SEARCH_RESULT_INDEX", "resume-search-result-index"
    )
    SIMILAR_THRESH: float = float(os.environ.get("SIMILAR_THRESH", 0.9))
    REMOTE_CV_API_KEY: Optional[str] = os.environ.get("REMOTE_CV_API_KEY")
    REMOTE_CV_TIMEOUT: int = int(os.environ.get("REMOTE_CV_TIMEOUT", 30))
    MAX_UPLOAD_SIZE_BYTES: int = int(
        os.environ.get("MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
    )
    ENABLE_ANTIVIRUS_SCAN: bool = bool(int(os.environ.get("ENABLE_ANTIVIRUS_SCAN", 0)))
    ENABLE_RATE_LIMIT: bool = bool(int(os.environ.get("ENABLE_RATE_LIMIT", 0)))
    DUPLICATE_AUTO_MERGE_THRESHOLD: float = float(
        os.environ.get("DUPLICATE_AUTO_MERGE_THRESHOLD", 0.85)
    )
    DUPLICATE_LLM_THRESHOLD: float = float(
        os.environ.get("DUPLICATE_LLM_THRESHOLD", 0.65)
    )
    DUPLICATE_CANDIDATE_SIZE: int = int(
        os.environ.get("DUPLICATE_CANDIDATE_SIZE", 20)
    )


settings = Settings()
