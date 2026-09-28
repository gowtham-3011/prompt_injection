"""Generate the report figures from saved experiment results.

Reads whatever exists under results/ and writes PNGs to results/plots/. Missing inputs
are skipped with a message rather than failing, so partial runs still produce figures.

Usage:
    python experiments/make_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data.loaders import build_dataset  # noqa: E402
from src.eval.metrics import threshold_at_fpr, tpr_at_threshold, wilson_interval  # noqa: E402
from src.eval.plots import (  # noqa: E402
    ablation_by_layer, asr_before_after, latency_distribution, roc_log_fpr, roc_points,
)
from src.firewall.classifier import TfidfClassifier  # noqa: E402
from src.firewall.heuristics import analyze  # noqa: E402
from src.utils import PROJECT_ROOT, get_logger, results_dir  # noqa: E402

log = get_logger("make_plots")
PLOTS = results_dir("plots")


def _prop(successes: int, total: int) -> tuple[float, float, float]:
    p = wilson_interval(successes, total)
    return p.rate, p.low, p.high


def _ablation_csv() -> Path | None:
    """Pick the ablation CSV to plot, preferring a real model over the mock.

    Headline figures must never be built from MockLLM runs: the mock only falls for
    attacks its regexes recognise, so its numbers would flatter every layer.
    """
    candidates = sorted(results_dir("phase7").glob("ablation_*.csv"))
    real = [p for p in candidates if "mock" not in p.stem]
    if real:
        return real[-1]
    if candidates:
        log.warning("only MockLLM ablation results found (%s); figures will be labelled "
                    "accordingly and must not be used as headline results.", candidates[-1].name)
        return candidates[-1]
    return None


def plot_asr_before_after() -> None:
    """Grouped bars from the ablation CSV: no_firewall vs all_layers, per category."""
    source = _ablation_csv()
    if source is None:
        log.warning("no ablation CSV; skipping ASR before/after plot")
        return
    df = pd.read_csv(source)
    attacks = df[df["kind"] == "attack"]
    if attacks.empty or "no_firewall" not in set(attacks["config"]):
        log.warning("ablation CSV lacks a no_firewall arm; skipping")
        return
    defended = "all_layers" if "all_layers" in set(attacks["config"]) else \
        sorted(set(attacks["config"]) - {"no_firewall"})[0]

    categories = sorted(attacks["category"].unique())
    before, after = [], []
    for category in categories:
        for config, bucket in ((("no_firewall"), before), ((defended), after)):
            rows = attacks[(attacks["config"] == config) & (attacks["category"] == category)]
            bucket.append(_prop(int(rows["success"].sum()), len(rows)))
    path = asr_before_after(categories, before, after, PLOTS / "asr_before_after.png",
                            after_label=f"firewall: {defended}")
    print(f"wrote {path}")


def plot_ablation() -> None:
    source = _ablation_csv()
    if source is None:
        log.warning("no ablation CSV; skipping ablation plot")
        return
    df = pd.read_csv(source)
    configs = list(dict.fromkeys(df["config"]))
    asr, benign = [], []
    for config in configs:
        a = df[(df["config"] == config) & (df["kind"] == "attack")]
        b = df[(df["config"] == config) & (df["kind"] == "benign")]
        asr.append(_prop(int(a["success"].sum()), len(a)))
        benign.append(_prop(int(b["success"].sum()), len(b)))
    path = ablation_by_layer(configs, asr, benign, PLOTS / "ablation_by_layer.png")
    print(f"wrote {path}")


def plot_roc() -> None:
    """ROC for the classifier on the held-out test split, log FPR axis."""
    checkpoint = PROJECT_ROOT / "checkpoints" / "classifier_tfidf.joblib"
    if not checkpoint.exists():
        log.warning("no classifier checkpoint; skipping ROC")
        return
    model = TfidfClassifier.load(checkpoint)
    ds, _ = build_dataset(seed=1234, use_public=True)
    texts = [e.text for e in ds.test]
    labels = [e.label for e in ds.test]
    scores = model.score_batch(texts)
    curves = {"classifier (tf-idf + logreg)": roc_points(scores, labels)}

    # Heuristics as a second curve: it is a scorer too, so it belongs on the same axes.
    heur = [analyze(t).score for t in texts]
    curves["heuristics (rules)"] = roc_points(heur, labels)

    op_fpr = 0.01
    op_threshold = threshold_at_fpr(scores, labels, op_fpr)
    op = {"classifier (tf-idf + logreg)": (op_fpr, tpr_at_threshold(scores, labels, op_threshold))}
    path = roc_log_fpr(curves, PLOTS / "roc_log_fpr.png", operating_points=op)
    print(f"wrote {path}")


def plot_latency() -> None:
    """Per-layer firewall processing time, measured directly on representative text."""
    import time

    from src.firewall.pipeline import Firewall, LayerSwitches, Mode, load_policy

    ds, _ = build_dataset(seed=1234, use_public=True)
    texts = [e.text for e in ds.test][:120] or ["sample text"]
    checkpoint = PROJECT_ROOT / "checkpoints" / "classifier_tfidf.joblib"
    classifier = TfidfClassifier.load(checkpoint) if checkpoint.exists() else None

    samples: dict[str, list[float]] = {}
    for label, switches in (("heuristics only", LayerSwitches.only("heuristics")),
                            ("classifier only", LayerSwitches.only("classifier")),
                            ("all layers", LayerSwitches())):
        if "classifier" in label and classifier is None:
            continue
        fw = Firewall(policy=load_policy(), switches=switches, mode=Mode.SANITIZE,
                      classifier=classifier, canary="CNRY-1111222233334444")
        values = []
        for text in texts:
            start = time.perf_counter()
            fw.inspect_input(text, source="read_inbox")
            values.append((time.perf_counter() - start) * 1000)
        samples[label] = values
    if samples:
        path = latency_distribution(samples, PLOTS / "latency_distribution.png")
        print(f"wrote {path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    plot_asr_before_after()
    plot_ablation()
    plot_roc()
    plot_latency()
    print(f"\nFigures in {PLOTS}")


if __name__ == "__main__":
    main()
