"""Interfaz comun a todos los proveedores LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    """Error del proveedor: red, autenticacion, rate limit, etc."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        """Retorna el texto del LLM (se espera un JSON cuando el prompt lo indica)."""
