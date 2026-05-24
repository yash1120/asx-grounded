from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = Field(default="")
    generator_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-opus-4-7"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "asx_chunks"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/asx_grounded"

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    retrieval_top_k: int = 50
    rerank_top_k: int = 8
    min_relevance_score: float = 0.30

    asx_user_agent: str = "asx-grounded/0.1 (research)"
    asx_rate_limit_per_sec: float = 1.0
    asx_data_dir: str = "./data/raw"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
