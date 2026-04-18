"""Dispara N tickets sinteticos contra la API (via HTTP).

Es la demo estrella del README: un comando y el pipeline completo se ve
trabajando.
"""

from __future__ import annotations

import argparse
import logging
import random
import time

import httpx

from src.logging_config import configure_logging
from src.ml.dataset import generate_dataset

log = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    df = generate_dataset(samples_per_template=max(1, args.count // 100 + 1))
    df = df.sample(n=min(args.count, len(df)), random_state=7).reset_index(drop=True)

    rng = random.Random(99)
    users = [f"user_{i}" for i in range(1, 21)]

    log.info("Disparando %s tickets a %s", len(df), args.url)

    counters = {"ok": 0, "error": 0, "classified": 0, "decided": 0}

    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        for _, row in df.iterrows():
            payload = {
                "user_id": rng.choice(users),
                "subject": row["subject"],
                "body": row["body"],
            }
            try:
                response = client.post("/tickets", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("Fallo en POST /tickets: %s", exc)
                counters["error"] += 1
                continue

            data = response.json()
            counters["ok"] += 1
            if data.get("classified"):
                counters["classified"] += 1
            if data.get("decided"):
                counters["decided"] += 1

            if args.sleep:
                time.sleep(args.sleep)

    log.info("Resultados: %s", counters)

    try:
        metrics = httpx.get(f"{args.url}/metrics").json()
        log.info("Metricas agregadas: %s", metrics)
    except httpx.HTTPError as exc:
        log.warning("No se pudo consultar /metrics: %s", exc)


if __name__ == "__main__":
    main()
