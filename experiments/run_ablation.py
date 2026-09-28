"""Phase 7: ablation - measure every firewall configuration on attacks AND benign tasks.

Answers the question a reviewer will ask: which layer is actually doing the work, and
what does it cost in usability and latency?

Usage:
    python experiments/run_ablation.py --backend mock --runs 5
    python experiments/run_ablation.py --backend ollama --runs 2
    python experiments/run_ablation.py --backend ollama --configs no_firewall all_layers
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.catalog import ALL_TEMPLATES  # noqa: E402
from src.eval.harness import (  # noqa: E402
    print_report, run_harness, standard_configurations,
)
from src.firewall.classifier import (  # noqa: E402
    TfidfClassifier, TransformerClassifier, UntrainedScorer,
)
from src.llm.client import LLMConfigError, create_llm_client  # noqa: E402
from src.utils import PROJECT_ROOT, get_logger, load_config, results_dir, set_seed  # noqa: E402

log = get_logger("run_ablation")


def load_classifier(kind: str = "tfidf"):
    """Load a trained classifier checkpoint, else a no-op scorer.

    Default is tf-idf, not the transformer, even though the transformer scores better
    in-distribution (AUC 0.988 vs 0.942). Two reasons, both measured in Phase 4:
      * the transformer caught 0/2 held-out attack families while tf-idf caught 1/2, so
        its advantage does not appear to survive contact with unseen attack types;
      * it is ~6x slower on CPU (p50 62 ms vs 9.7 ms), and this layer runs on every
        chunk of untrusted content.
    Pass --classifier transformer to compare. Neither is the load-bearing defense.
    """
    if kind == "transformer":
        path = PROJECT_ROOT / "checkpoints" / "classifier_transformer"
        if path.exists():
            model = TransformerClassifier.load(path)
            log.info("loaded transformer classifier (threshold=%.3f)", model.threshold)
            return model
    else:
        path = PROJECT_ROOT / "checkpoints" / "classifier_tfidf.joblib"
        if path.exists():
            model = TfidfClassifier.load(path)
            log.info("loaded tf-idf classifier (threshold=%.3f)", model.threshold)
            return model
    log.warning("no %s checkpoint at %s; classifier layer will score 0. "
                "Run experiments/train_classifier.py first.", kind, path)
    return UntrainedScorer()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["anthropic", "ollama", "mock"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--configs", nargs="*", help="subset of configuration names to run")
    ap.add_argument("--classifier", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    overrides = list(args.set) + ([f"llm.backend={args.backend}"] if args.backend else [])
    cfg = load_config(overrides=overrides)
    set_seed(cfg["seed"])

    try:
        llm = create_llm_client(cfg)
    except LLMConfigError as err:
        sys.exit(f"LLM configuration error: {err}")

    configs = standard_configurations()
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            sys.exit(f"No configurations matched {sorted(wanted)}")

    result = run_harness(configs, ALL_TEMPLATES, cfg["agent"], llm, runs=args.runs,
                         base_seed=cfg["seed"], classifier=load_classifier(args.classifier))
    print_report(result)

    out_dir = results_dir("phase7")
    rows = [asdict(r) for r in result.records]
    with (out_dir / f"ablation_{llm.name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": llm.model, "runs": args.runs,
        "configs": {
            c.name: {
                "asr": result.asr(c.name).rate,
                "asr_ci": [result.asr(c.name).low, result.asr(c.name).high],
                "benign_success": result.benign_success(c.name).rate,
                "benign_ci": [result.benign_success(c.name).low, result.benign_success(c.name).high],
                "median_latency_s": result.median_latency(c.name),
                "mean_latency_s": result.mean_latency(c.name),
                "asr_by_category": {cat: result.asr(c.name, cat).rate
                                    for cat in result.categories()},
            } for c in configs
        },
    }
    (out_dir / f"ablation_{llm.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
