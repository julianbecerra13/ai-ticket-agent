.PHONY: help up down build logs shell train seed demo test lint format migrate

help:
	@echo "Comandos disponibles:"
	@echo "  make up        - Levanta la API y Postgres con Docker Compose"
	@echo "  make down      - Detiene los servicios"
	@echo "  make build     - Reconstruye la imagen de la API"
	@echo "  make logs      - Muestra logs de la API"
	@echo "  make shell     - Abre un shell dentro del contenedor de la API"
	@echo "  make migrate   - Aplica migraciones Alembic"
	@echo "  make train     - Entrena el clasificador y guarda el .pkl"
	@echo "  make seed      - Inserta tickets de ejemplo en la base"
	@echo "  make demo      - Dispara 100 tickets sinteticos contra la API"
	@echo "  make test      - Corre pytest con cobertura"
	@echo "  make lint      - Revisa el codigo con ruff"
	@echo "  make format    - Formatea el codigo con ruff"

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f api

shell:
	docker compose exec api sh

migrate:
	docker compose exec api alembic upgrade head

train:
	docker compose exec api python scripts/train_model.py
	docker compose restart api

seed:
	docker compose exec api python scripts/seed_db.py

demo:
	docker compose exec api python scripts/simulate_tickets.py --count 100

test:
	docker compose exec api pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts
