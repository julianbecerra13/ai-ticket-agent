"""Clasificador de tickets.

Envuelve dos modelos sklearn (categoria y urgencia) compartiendo vectorizador
TF-IDF. La API publica es minima: `train`, `predict`, `save`, `load`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.db.models import TicketCategory, TicketUrgency


@dataclass
class PredictionResult:
    category: TicketCategory
    urgency: TicketUrgency
    confidence_category: float
    confidence_urgency: float


@dataclass
class TrainingMetrics:
    accuracy_category: float
    accuracy_urgency: float
    report_category: str
    report_urgency: str
    confusion_category: np.ndarray
    confusion_urgency: np.ndarray
    labels_category: list[str]
    labels_urgency: list[str]


def _compose_text(subject: str, body: str) -> str:
    return f"{subject}. {body}".strip()


class TicketClassifier:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.model_category: LogisticRegression | None = None
        self.model_urgency: LogisticRegression | None = None

    @property
    def is_ready(self) -> bool:
        return all(
            x is not None for x in (self.vectorizer, self.model_category, self.model_urgency)
        )

    def train(
        self, df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
    ) -> TrainingMetrics:
        texts = df.apply(lambda row: _compose_text(row["subject"], row["body"]), axis=1).tolist()
        y_cat = df["category"].tolist()
        y_urg = df["urgency"].tolist()

        X_train_txt, X_test_txt, y_cat_train, y_cat_test, y_urg_train, y_urg_test = (
            train_test_split(
                texts, y_cat, y_urg, test_size=test_size, random_state=random_state, stratify=y_cat
            )
        )

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            lowercase=True,
        )
        X_train = self.vectorizer.fit_transform(X_train_txt)
        X_test = self.vectorizer.transform(X_test_txt)

        self.model_category = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model_category.fit(X_train, y_cat_train)
        pred_cat = self.model_category.predict(X_test)

        self.model_urgency = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model_urgency.fit(X_train, y_urg_train)
        pred_urg = self.model_urgency.predict(X_test)

        labels_cat = sorted(set(y_cat))
        labels_urg = sorted(set(y_urg))

        return TrainingMetrics(
            accuracy_category=accuracy_score(y_cat_test, pred_cat),
            accuracy_urgency=accuracy_score(y_urg_test, pred_urg),
            report_category=classification_report(y_cat_test, pred_cat, zero_division=0),
            report_urgency=classification_report(y_urg_test, pred_urg, zero_division=0),
            confusion_category=confusion_matrix(y_cat_test, pred_cat, labels=labels_cat),
            confusion_urgency=confusion_matrix(y_urg_test, pred_urg, labels=labels_urg),
            labels_category=labels_cat,
            labels_urgency=labels_urg,
        )

    def predict(self, *, subject: str, body: str) -> PredictionResult:
        if not self.is_ready:
            raise RuntimeError("El clasificador no esta entrenado. Ejecuta `make train` primero.")

        assert self.vectorizer is not None
        assert self.model_category is not None
        assert self.model_urgency is not None

        X = self.vectorizer.transform([_compose_text(subject, body)])

        cat_proba = self.model_category.predict_proba(X)[0]
        cat_idx = int(np.argmax(cat_proba))
        cat_label = str(self.model_category.classes_[cat_idx])

        urg_proba = self.model_urgency.predict_proba(X)[0]
        urg_idx = int(np.argmax(urg_proba))
        urg_label = str(self.model_urgency.classes_[urg_idx])

        return PredictionResult(
            category=TicketCategory(cat_label),
            urgency=TicketUrgency(urg_label),
            confidence_category=float(cat_proba[cat_idx]),
            confidence_urgency=float(urg_proba[urg_idx]),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "model_category": self.model_category,
                "model_urgency": self.model_urgency,
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path) -> TicketClassifier:
        data = joblib.load(path)
        clf = cls()
        clf.vectorizer = data["vectorizer"]
        clf.model_category = data["model_category"]
        clf.model_urgency = data["model_urgency"]
        return clf
