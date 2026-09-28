"""Metrics for honest reporting: proportions with confidence intervals, and
detection metrics (precision/recall/F1/AUC, TPR at fixed FPR).

Why Wilson intervals: attack success rate and false positive rate are proportions,
often measured on small n or near 0/1 where the normal approximation is wrong (it can
give negative lower bounds). The Wilson score interval stays inside [0, 1] and behaves
well at the extremes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Proportion:
    successes: int
    total: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.rate:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.total})"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Proportion:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if total == 0:
        return Proportion(0, 0, 0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return Proportion(successes, total, max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    auc: float
    tpr_at_fpr: dict[float, float]  # e.g. {0.01: 0.83, 0.001: 0.60}
    threshold: float
    n_pos: int
    n_neg: int


def _auc(scores: list[float], labels: list[int]) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) statistic, with tie handling."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank for ties (1-indexed)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def threshold_at_fpr(scores: list[float], labels: list[int], target_fpr: float) -> float:
    """Lowest score threshold whose FPR on negatives is <= target_fpr.

    Decision rule downstream is `score >= threshold => flag`. We pick the threshold on
    VALIDATION data so we can advertise a target false-positive budget honestly.
    """
    neg = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    if not neg:
        return 1.0
    k = int(math.floor(target_fpr * len(neg)))  # allow this many negatives above threshold
    if k <= 0:
        return neg[0] + 1e-9  # stricter than the top negative => ~0 FPR
    return neg[k - 1] + 1e-12


def tpr_at_threshold(scores: list[float], labels: list[int], threshold: float) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    if not pos:
        return float("nan")
    return sum(s >= threshold for s in pos) / len(pos)


def compute_detection_metrics(
    scores: list[float], labels: list[int], threshold: float,
    fpr_points: tuple[float, ...] = (0.01, 0.001),
) -> DetectionMetrics:
    """Full detection report at a chosen operating threshold."""
    tp = sum(s >= threshold and y == 1 for s, y in zip(scores, labels))
    fp = sum(s >= threshold and y == 0 for s, y in zip(scores, labels))
    fn = sum(s < threshold and y == 1 for s, y in zip(scores, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr_at = {
        fpr: tpr_at_threshold(scores, labels, threshold_at_fpr(scores, labels, fpr))
        for fpr in fpr_points
    }
    return DetectionMetrics(
        precision, recall, f1, _auc(scores, labels), tpr_at, threshold,
        n_pos=sum(labels), n_neg=len(labels) - sum(labels),
    )
