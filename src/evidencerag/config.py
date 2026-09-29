from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "EvidenceRAG"
    database_url: str = "postgresql+asyncpg://evidencerag:evidencerag@localhost:5432/evidencerag"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    llm_provider: Literal["local", "openai"] = "local"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    embedding_dimension: int = Field(default=384, ge=384, le=384)
    retrieval_candidates: int = Field(default=30, ge=5, le=100)
    top_k_max: int = Field(default=10, ge=1, le=20)
    outbox_poll_seconds: float = Field(default=0.5, ge=0.1, le=30)
    log_level: str = "INFO"
    api_keys: dict[str, SecretStr] = Field(default_factory=dict)
    ingestion_lease_seconds: int = Field(default=120, ge=10, le=600)
    max_ingestion_attempts: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_provider(self) -> "Settings":
        keys = [secret.get_secret_value() for secret in self.api_keys.values()]
        if any(not 1 <= len(owner) <= 100 for owner in self.api_keys):
            raise ValueError("ID клиента должен содержать от 1 до 100 символов")
        if any(len(key) < 16 for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Ключи клиентов должны быть уникальными и не короче 16 символов")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("Для LLM_PROVIDER=openai требуется OPENAI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
