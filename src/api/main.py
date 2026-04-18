"""Aplicacion FastAPI principal."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import agent as agent_routes
from src.api.routes import health as health_routes
from src.api.routes import metrics as metrics_routes
from src.api.routes import tickets as tickets_routes
from src.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="ai-ticket-agent",
        version=__version__,
        description=(
            "Backend que clasifica tickets de soporte con ML clasico y los procesa con un "
            "agente LLM multi-proveedor (Anthropic, OpenAI u Ollama). El usuario final trae "
            "su propia API key y la API arranca igual en modo degradado si no hay una."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router)
    app.include_router(tickets_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(agent_routes.router)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "service": "ai-ticket-agent",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
