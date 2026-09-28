"""Metrics: Wilson intervals, AUC, and threshold selection at a target FPR."""

from __future__ import annotations

import math

import pytest

from src.eval.metrics import (
    compute_detection_metrics, threshold_at_fpr, tpr_at_threshold, wilson_interval,
)


# -- Wilson interval --------------------------------------------------------- #
def test_wilson_interval_stays_in_unit_range_at_extremes():
    """The normal approximation gives negative bounds at 0/n; Wilson must not."""
    low = wilson_interval(0, 10)
    high = wilson_interval(10, 10)
    assert low.rate == 0.0 and low.low == 0.0 and 0 < low.high < 1
    assert high.rate == 1.0 and high.high == 1.0 and 0 < high.low < 1


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large.high - large.low) < (small.high - small.low)


def test_wilson_interval_contains_point_estimate():
    for successes, total in [(3, 12), (1, 100), (7, 9)]:
        p = wilson_interval(successes, total)
        assert p.low <= p.rate <= p.high


def test_wilson_interval_handles_zero_total():
    p = wilson_interval(0, 0)
    assert p.rate == 0.0 and p.low == 0.0 and p.high == 0.0


# -- AUC --------------------------------------------------------------------- #
def test_auc_perfect_and_inverted_separation():
    scores, labels = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
    assert compute_detection_metrics(scores, labels, 0.5).auc == 1.0
    assert compute_detection_metrics(scores, [1, 1, 0, 0], 0.5).auc == 0.0


def test_auc_of_random_scores_is_half_with_ties():
    scores, labels = [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]
    assert compute_detection_metrics(scores, labels, 0.5).auc == pytest.approx(0.5)


def test_auc_is_nan_without_both_classes():
    assert math.isnan(compute_detection_metrics([0.1, 0.2], [0, 0], 0.5).auc)


# -- threshold selection ----------------------------------------------------- #
def test_threshold_at_fpr_respects_budget():
    """At most target_fpr of NEGATIVES may score at or above the threshold."""
    scores = [i / 100 for i in range(100)]          # 0.00 .. 0.99
    labels = [0] * 100
    threshold = threshold_at_fpr(scores, labels, 0.05)
    assert sum(s >= threshold for s in scores) <= 5


def test_threshold_at_zero_budget_excludes_all_negatives():
    scores = [0.1, 0.4, 0.9, 0.95]
    labels = [0, 0, 1, 1]
    threshold = threshold_at_fpr(scores, labels, 0.0)
    assert all(s < threshold for s, y in zip(scores, labels) if y == 0)


def test_tpr_at_threshold():
    scores, labels = [0.9, 0.8, 0.2], [1, 1, 1]
    assert tpr_at_threshold(scores, labels, 0.5) == pytest.approx(2 / 3)


def test_detection_metrics_reports_both_fpr_points():
    scores = [0.05, 0.1, 0.2, 0.7, 0.8, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    m = compute_detection_metrics(scores, labels, 0.5)
    assert set(m.tpr_at_fpr) == {0.01, 0.001}
    assert m.n_pos == 3 and m.n_neg == 3
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
