# Arquitectura

Documento tecnico complementario al README. Aqui se detallan capas, fronteras
y decisiones de diseno que un revisor tecnico suele buscar.

## Capas

```
┌──────────────────────────────────────────────┐
│                    API (FastAPI)             │
│     /health · /tickets · /metrics · /agent   │
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────▼───────────┐
         │   TicketService     │
         │  (capa de casos de  │
         │     uso / aplicacion│
         └────┬──────┬────┬────┘
              │      │    │
     ┌────────▼──┐ ┌─▼─┐ ┌▼────────┐
     │Clasificador│ │LLM│ │Repositorios│
     │  sklearn   │ │Agt│ │SQLAlchemy  │
     └────────────┘ └───┘ └──────┬─────┘
                                 │
                          ┌──────▼──────┐
                          │ PostgreSQL  │
                          └─────────────┘
```

Cada capa conoce solo la inmediatamente inferior. La API no toca la BD ni al
clasificador directamente: siempre pasa por el servicio. El servicio usa
repositorios (nunca SQL crudo) y objetos del dominio del ML.

## Modelo de datos

```
tickets
  id · user_id · subject · body · created_at
    │
    ├──1:1── predictions (category, urgency, confidences)
    └──1:1── agent_decisions (action, reasoning, response_text, provider, model)
                │
                └──1:N── actions (type, payload JSON, status, executed_at)
```

- Una **prediccion** por ticket (el clasificador corre una vez).
- Una **decision** por ticket (el agente corre una vez; `/agent/decide` sobrescribe).
- Una o mas **acciones** por decision (permite encadenar acciones en el futuro
  sin modificar el esquema).

## Componentes del ML

### Clasificador (`src/ml/classifier.py`)

Dos modelos independientes que comparten el mismo vectorizador TF-IDF:

- **Categoria**: 5 clases (`tecnico`, `facturacion`, `cuenta`, `informacion`, `queja`).
- **Urgencia**: 4 niveles (`baja`, `media`, `alta`, `critica`).

Usar modelos separados evita el problema de la correlacion entre dimensiones:
un ticket `tecnico` puede ser urgente o no, y esas son decisiones diferentes
para el modelo.

### Dataset (`src/ml/dataset.py`)

Dataset sintetico reproducible: 100 plantillas (20 por categoria) combinadas
con prefijos de urgencia y sufijos amables. Con `samples_per_template=8` salen
~800 filas con distribucion balanceada. Semilla fija para reproducibilidad.

## Agente

### Proveedores (`src/agent/providers/`)

```
LLMProvider (ABC)
  ├── AnthropicProvider
  ├── OpenAIProvider
  ├── OllamaProvider
  └── MockProvider     (para tests y demo sin API key)
```

La interfaz es deliberadamente delgada: un metodo `generate(system, user) -> str`.
Esto permite anadir proveedores nuevos (Gemini, Mistral, etc.) sin tocar la
logica del agente.

### Orquestador (`src/agent/agent.py`)

1. Arma el prompt (system + user) con la informacion del ticket y el historico.
2. Llama al proveedor.
3. Parsea el JSON de respuesta con un modelo Pydantic (`_RawDecision`).
4. Si parseo falla, **reintenta una vez**.
5. Si sigue fallando, devuelve `ESCALATE` con el motivo del fallback.

Esto garantiza que el flujo nunca se rompa por errores del LLM.

## Degradacion controlada

| Componente | Si falta |
|------------|----------|
| Postgres | `/health` reporta `database: unreachable`. La API arranca igual, pero cualquier operacion falla con 500. |
| Modelo `.pkl` | `/health` reporta `classifier: not_trained`. `POST /tickets` guarda el ticket sin prediccion ni decision. |
| Proveedor LLM | `/health` reporta `llm_status: not_configured`. `POST /tickets` clasifica pero no decide. `/agent/decide` devuelve 503. |

## Flujo de tests

```
tests/
├── conftest.py              # Fixtures: engine SQLite in-memory, MockProvider, overrides
├── test_classifier.py       # Unidad: entrenamiento y prediccion
├── test_providers.py        # Unidad: factory y MockProvider
├── test_agent.py            # Unidad: reintentos, fallback, parseo
├── test_repositories.py     # Unidad: queries y metricas
├── test_ingestor.py         # Unidad: ingestor CSV
├── test_api_tickets.py      # Integracion HTTP
├── test_api_metrics.py      # Integracion HTTP
├── test_api_agent.py        # Integracion HTTP
└── test_api_health.py       # Integracion HTTP
```

SQLite in-memory evita la dependencia de Docker o Postgres en CI. Los tests
reemplazan el agente por un `MockProvider` que nunca sale a red.

## Extensibilidad

### Agregar un proveedor nuevo (ej. Gemini)

1. Crear `src/agent/providers/gemini_provider.py` implementando `LLMProvider`.
2. Agregar rama en `src/agent/providers/factory.py` y la env var correspondiente.
3. Actualizar `LLMProviderName` en `src/config.py`.
4. Nada mas cambia: ni el agente, ni los servicios, ni la API.

### Agregar una accion nueva (ej. `REFUND`)

1. Agregar el valor al enum `AgentActionType` en `src/db/models.py`.
2. Crear la migracion Alembic (solo si quieres constraints adicionales).
3. Documentar la accion en `src/agent/prompts.py` para que el LLM la conozca.
4. Opcionalmente, manejar el payload en `src/db/repositories.py` si la accion
   requiere ejecutar algo externo.
