"""Layer 2: a small fine-tuned transformer that scores text as injection / benign.

Model choice: a small encoder (DistilBERT / DeBERTa-v3-small). It fine-tunes on a free
Colab GPU in minutes and runs on CPU at inference. A large generative model as a judge
would likely score better, but it costs a model call per check, adds latency to every
request, and is itself promptable - a detector you can prompt-inject is a poor detector.

Operating point: the decision threshold is chosen on the VALIDATION split to meet a
stated false-positive budget (default 1%), never on test. A detector is only meaningful
together with the FPR it was tuned for.

Two inference paths:
  * TransformerClassifier: the fine-tuned checkpoint (torch + transformers).
  * TfidfClassifier: a scikit-learn baseline that trains in seconds with no GPU. It also
    keeps the whole pipeline runnable in CI and on machines without torch, and it is an
    honest baseline to compare the transformer against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.data.loaders import Example
from src.eval.metrics import threshold_at_fpr
from src.utils import get_logger

log = get_logger(__name__)


class Scorer(Protocol):
    """Anything that turns text into an injection score in [0, 1]."""

    def score(self, text: str) -> float: ...
    def score_batch(self, texts: list[str]) -> list[float]: ...


@dataclass
class ClassifierMetadata:
    """Everything needed to reproduce and audit an operating point."""

    kind: str
    base_model: str
    threshold: float
    target_fpr: float
    train_size: int
    val_size: int
    seed: int


# --------------------------------------------------------------------------- #
# TF-IDF baseline (no GPU, seconds to train)
# --------------------------------------------------------------------------- #
class TfidfClassifier:
    """Logistic regression over character + word n-grams.

    Character n-grams matter here: they survive spacing tricks and partial obfuscation
    that destroy word tokens.
    """

    kind = "tfidf"

    def __init__(self, seed: int = 1234) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        self.seed = seed
        self.threshold = 0.5
        self.base_model = "tfidf+logreg"
        self._pipeline = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
            ])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ])

    def fit(self, train: list[Example]) -> None:
        texts = [e.text for e in train]
        labels = [e.label for e in train]
        self._pipeline.fit(texts, labels)

    def score_batch(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(p[1]) for p in self._pipeline.predict_proba(texts)]

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfClassifier":
        import joblib

        blob = joblib.load(path)
        obj = cls()
        obj._pipeline = blob["pipeline"]
        obj.threshold = blob["threshold"]
        return obj


# --------------------------------------------------------------------------- #
# Transformer classifier
# --------------------------------------------------------------------------- #
class TransformerClassifier:
    """Fine-tuned sequence classifier. Training needs torch+transformers; inference is CPU-friendly."""

    kind = "transformer"

    def __init__(self, base_model: str = "distilbert-base-uncased", seed: int = 1234,
                 max_length: int = 256, device: str | None = None) -> None:
        self.base_model = base_model
        self.seed = seed
        self.max_length = max_length
        self.threshold = 0.5
        self._device = device
        self._model = None
        self._tokenizer = None

    # -- training ------------------------------------------------------------ #
    def fit(self, train: list[Example], val: list[Example] | None = None,
            epochs: int = 4, batch_size: int = 16, lr: float = 2e-5) -> dict:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=2)
        device = self._resolve_device()
        self._model.to(device)

        enc = self._tokenizer([e.text for e in train], truncation=True, padding="max_length",
                              max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([e.label for e in train])
        loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels),
                            batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
        history = []
        self._model.train()
        for epoch in range(epochs):
            total = 0.0
            for input_ids, mask, y in loader:
                optimizer.zero_grad()
                out = self._model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                                  labels=y.to(device))
                out.loss.backward()
                optimizer.step()
                total += float(out.loss.detach())
            history.append({"epoch": epoch + 1, "loss": total / max(1, len(loader))})
            log.info("epoch %d loss %.4f", epoch + 1, history[-1]["loss"])
        self._model.eval()
        return {"history": history}

    def _resolve_device(self) -> str:
        import torch

        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- inference ----------------------------------------------------------- #
    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("classifier not trained or loaded")
        device = self._resolve_device()
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self._tokenizer(chunk, truncation=True, padding=True,
                                      max_length=self.max_length, return_tensors="pt").to(device)
                logits = self._model(**enc).logits
                scores += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        return [float(s) for s in scores]

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        (directory / "operating_point.json").write_text(
            json.dumps({"threshold": self.threshold, "base_model": self.base_model}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> "TransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((directory / "operating_point.json").read_text(encoding="utf-8"))
        obj = cls(base_model=meta["base_model"], device=device)
        obj._tokenizer = AutoTokenizer.from_pretrained(directory)
        obj._model = AutoModelForSequenceClassification.from_pretrained(directory)
        obj._model.eval()
        obj.threshold = meta["threshold"]
        return obj


# --------------------------------------------------------------------------- #
# Threshold selection (validation only)
# --------------------------------------------------------------------------- #
def choose_threshold(model: Scorer, val: list[Example], target_fpr: float = 0.01) -> float:
    """Pick the score threshold meeting `target_fpr` on the VALIDATION split."""
    scores = model.score_batch([e.text for e in val])
    labels = [e.label for e in val]
    return threshold_at_fpr(scores, labels, target_fpr)


# --------------------------------------------------------------------------- #
# A keyword-free fallback so the pipeline always has a classifier layer available
# --------------------------------------------------------------------------- #
class UntrainedScorer:
    """Returns 0.0 for everything. Used when no checkpoint exists, so the firewall can
    still run with the classifier layer effectively disabled rather than crashing."""

    kind = "untrained"
    base_model = "none"
    threshold = 1.1  # nothing ever reaches it

    def score(self, text: str) -> float:
        return 0.0

    def score_batch(self, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)
