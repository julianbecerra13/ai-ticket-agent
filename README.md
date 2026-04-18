# ai-ticket-agent

Backend que clasifica tickets de soporte con ML clasico y deja que un agente LLM decida automaticamente que hacer con cada uno. Disenado para ser portable: traes tu propia API key (Anthropic, OpenAI u Ollama local) y funciona.

> README extendido en la **Fase 9** del plan de implementacion. Por ahora solo el esqueleto.

## Stack

- Python 3.12 + FastAPI
- PostgreSQL 16 + SQLAlchemy 2 + Alembic
- scikit-learn (TF-IDF + LogisticRegression)
- Anthropic / OpenAI / Ollama (intercambiables)
- Docker + docker-compose

## Quickstart

```bash
git clone https://github.com/julianbecerra13/ai-ticket-agent.git
cd ai-ticket-agent
cp .env.example .env
docker compose up -d
make migrate
make train
make demo
curl http://localhost:8000/health
```

## Estado

Proyecto en construccion. Ver `docs/architecture.md` para los detalles.
