"""Ejecuta el scheduler APScheduler que procesa `data/raw/batch.csv` cada X minutos."""

from __future__ import annotations

import argparse

from src.automation.scheduler import build_scheduler
from src.logging_config import configure_logging


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10, help="Intervalo en minutos.")
    args = parser.parse_args()

    scheduler = build_scheduler(interval_minutes=args.interval)
    scheduler.start()


if __name__ == "__main__":
    main()
