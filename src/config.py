"""Configuracion central de la aplicacion.

Lee variables del entorno (o del archivo .env) y expone un objeto `settings`
que el resto de la aplicacion consume. La deteccion del proveedor LLM ocurre
aqui: si el usuario fija `LLM_PROVIDER`, se respeta; si no, se elige el primero
disponible en orden Anthropic -> OpenAI -> Ollama. Si no hay ninguno, la API
arranca igual en modo degradado.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/tickets",
        description="URL SQLAlchemy para conectarse a Postgres.",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_path: str = "src/ml/models/ticket_classifier.pkl"

    llm_provider: LLMProviderName | None = None

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-7"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    ollama_host: str | None = None
    ollama_model: str = "llama3.2"

    def resolve_llm_provider(self) -> LLMProviderName | None:
        """Devuelve el proveedor LLM efectivo.

        Si el usuario fijo `LLM_PROVIDER`, se valida que la clave correspondiente
        exista y se retorna. Si no, se selecciona el primero disponible segun
        el orden documentado en el README.
        """
        if self.llm_provider is not None:
            if self._provider_available(self.llm_provider):
                return self.llm_provider
            return None

        for candidate in (
            LLMProviderName.ANTHROPIC,
            LLMProviderName.OPENAI,
            LLMProviderName.OLLAMA,
        ):
            if self._provider_available(candidate):
                return candidate
        return None

    def _provider_available(self, provider: LLMProviderName) -> bool:
        match provider:
            case LLMProviderName.ANTHROPIC:
                return bool(self.anthropic_api_key)
            case LLMProviderName.OPENAI:
                return bool(self.openai_api_key)
            case LLMProviderName.OLLAMA:
                return bool(self.ollama_host)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
