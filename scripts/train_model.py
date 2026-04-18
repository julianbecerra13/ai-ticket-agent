"""Entrena el clasificador de tickets y guarda el .pkl.

Uso:
    python scripts/train_model.py [--samples 10]

Salidas:
    - src/ml/models/ticket_classifier.pkl
    - docs/images/confusion_category.png (si matplotlib disponible)
    - docs/images/confusion_urgency.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import settings
from src.logging_config import configure_logging
from src.ml.classifier import TicketClassifier, TrainingMetrics
from src.ml.dataset import save_dataset

log = logging.getLogger(__name__)


def _plot_confusion(metrics: TrainingMetrics, images_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        log.warning("matplotlib/seaborn no instalados; se omite la grafica de confusion.")
        return

    images_dir.mkdir(parents=True, exist_ok=True)

    for title, cm, labels, filename in [
        (
            "Matriz de confusion - Categoria",
            metrics.confusion_category,
            metrics.labels_category,
            "confusion_category.png",
        ),
        (
            "Matriz de confusion - Urgencia",
            metrics.confusion_urgency,
            metrics.labels_urgency,
            "confusion_urgency.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        plt.tight_layout()
        fig.savefig(images_dir / filename, dpi=120)
        plt.close(fig)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Entrena el clasificador de tickets.")
    parser.add_argument("--samples", type=int, default=8, help="Muestras por plantilla.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/tickets_synthetic.csv",
        help="Ruta donde guardar el dataset generado.",
    )
    args = parser.parse_args()

    log.info("Generando dataset sintetico (samples_per_template=%s)...", args.samples)
    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = save_dataset(str(dataset_path), samples_per_template=args.samples)
    log.info("Dataset guardado en %s (%s filas).", dataset_path, len(df))

    clf = TicketClassifier()
    log.info("Entrenando modelo...")
    metrics = clf.train(df)

    log.info("Accuracy categoria: %.3f", metrics.accuracy_category)
    log.info("Accuracy urgencia:  %.3f", metrics.accuracy_urgency)
    log.info("\nReporte categoria:\n%s", metrics.report_category)
    log.info("\nReporte urgencia:\n%s", metrics.report_urgency)

    model_path = Path(settings.model_path)
    clf.save(model_path)
    log.info("Modelo guardado en %s", model_path)

    _plot_confusion(metrics, Path("docs/images"))
    log.info("Matrices de confusion guardadas en docs/images/")


if __name__ == "__main__":
    main()
