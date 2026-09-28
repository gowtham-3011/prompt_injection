"""Phase 4: train the injection classifier and report it honestly.

Protocol:
  * Train on TRAIN only.
  * Choose the decision threshold on VALIDATION at a stated false-positive budget.
  * Report on (a) the in-distribution TEST set, (b) HELD-OUT attack families the model
    never saw, and (c) hard negatives separately - a detector that only works on the
    attack styles it was trained on is not a defense.
  * Measure inference latency (p50/p95).

Usage:
    python experiments/train_classifier.py --kind tfidf
    python experiments/train_classifier.py --kind transformer --epochs 4
    python experiments/train_classifier.py --kind tfidf --target-fpr 0.01 --no-public
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.data.loaders import Example, build_dataset  # noqa: E402
from src.eval.metrics import compute_detection_metrics, wilson_interval  # noqa: E402
from src.firewall.classifier import (  # noqa: E402
    TfidfClassifier, TransformerClassifier, choose_threshold,
)
from src.utils import PROJECT_ROOT, get_logger, results_dir, set_seed  # noqa: E402

log = get_logger("train_classifier")


def evaluate(model, examples: list[Example], threshold: float, label: str) -> dict:
    if not examples:
        return {}
    scores = model.score_batch([e.text for e in examples])
    labels = [e.label for e in examples]
    if len(set(labels)) < 2:  # e.g. held-out split is positives only: report recall
        detected = sum(s >= threshold for s in scores)
        p = wilson_interval(detected, len(scores))
        print(f"{label:28s} recall {p}")
        return {"split": label, "recall": p.rate, "ci": [p.low, p.high], "n": len(scores)}
    m = compute_detection_metrics(scores, labels, threshold)
    print(f"{label:28s} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
          f"AUC={m.auc:.3f} TPR@1%FPR={m.tpr_at_fpr[0.01]:.3f} TPR@0.1%FPR={m.tpr_at_fpr[0.001]:.3f}")
    return {"split": label, **{k: v for k, v in m.__dict__.items()}}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-public", action="store_true", help="skip public datasets (offline)")
    args = ap.parse_args()

    set_seed(args.seed)
    ds, messages = build_dataset(seed=args.seed, use_public=not args.no_public)
    for m in messages:
        print(f"[data] {m}")

    if args.kind == "tfidf":
        model = TfidfClassifier(seed=args.seed)
        model.fit(ds.train)
    else:
        model = TransformerClassifier(base_model=args.base_model, seed=args.seed)
        model.fit(ds.train, epochs=args.epochs)

    # Threshold from VALIDATION only.
    model.threshold = choose_threshold(model, ds.val, target_fpr=args.target_fpr)
    print(f"\nthreshold={model.threshold:.4f} chosen on validation at target FPR={args.target_fpr}")

    print(f"\n{'=' * 78}\nCLASSIFIER: {model.kind} ({model.base_model})\n{'=' * 78}")
    results = [
        evaluate(model, ds.val, model.threshold, "validation"),
        evaluate(model, ds.test, model.threshold, "test (in-distribution)"),
        evaluate(model, ds.heldout, model.threshold, "held-out families (unseen)"),
    ]

    # Hard negatives alone: the FPR that actually matters. Only the TEST-split ones, so we
    # never report a false-positive rate on text the model was trained on.
    hard = ds.hard_negatives("test")
    if hard:
        hard_scores = model.score_batch([e.text for e in hard])
        fp = sum(s >= model.threshold for s in hard_scores)
        p = wilson_interval(fp, len(hard))
        print(f"{'hard negatives (FPR, test)':28s} {p}")
        results.append({"split": "hard_negatives_test", "fpr": p.rate,
                        "ci": [p.low, p.high], "n": len(hard)})
        for ex, score in zip(hard, hard_scores):
            if score >= model.threshold:
                print(f"    false positive: {ex.source} (score={score:.3f})")
    else:
        print(f"{'hard negatives (FPR, test)':28s} none landed in the test split this seed")

    # Latency.
    sample = [e.text for e in ds.test] * 4 or ["test"]
    lat = []
    for text in sample[:200]:
        t0 = time.perf_counter()
        model.score(text)
        lat.append((time.perf_counter() - t0) * 1000)
    device = "cpu" if args.kind == "tfidf" else model._resolve_device()
    print(f"\nlatency ({device}) p50={statistics.median(lat):.2f} ms "
          f"p95={sorted(lat)[int(0.95 * len(lat))]:.2f} ms (n={len(lat)})")

    out_dir = PROJECT_ROOT / "checkpoints" / f"classifier_{model.kind}"
    if args.kind == "tfidf":
        model.save(out_dir.with_suffix(".joblib"))
    else:
        model.save(out_dir)
    report = {"kind": model.kind, "base_model": model.base_model, "threshold": model.threshold,
              "target_fpr": args.target_fpr, "seed": args.seed,
              "data_counts": ds.counts(), "results": results,
              "latency_ms": {"p50": statistics.median(lat),
                             "p95": sorted(lat)[int(0.95 * len(lat))], "device": device}}
    path = results_dir("phase4") / f"classifier_{model.kind}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved model to {out_dir} and report to {path}")


if __name__ == "__main__":
    main()
