"""Phase 7: run the ADAPTIVE attacks - the ones written to bypass this firewall.

Reports the bypass rate per layer and per adaptive family, so the result says WHICH
assumption broke, not just how many got through.

Methodology (stated in the README too): these attacks were written once, after reading
the firewall's rules. They are evaluated, used to improve the firewall exactly once, and
re-measured. Repeatedly tuning against them would turn this test set into a training set
and make the reported bypass rate meaningless.

Usage:
    python experiments/run_adaptive.py --backend ollama --runs 3
    python experiments/run_adaptive.py --backend mock --runs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.adaptive import ADAPTIVE_TEMPLATES  # noqa: E402
from src.attacks.generator import _payload_text  # noqa: E402
from src.eval.harness import Configuration, run_attack  # noqa: E402
from src.eval.metrics import wilson_interval  # noqa: E402
from src.firewall.heuristics import analyze  # noqa: E402
from src.firewall.pipeline import LayerSwitches, Mode  # noqa: E402
from src.llm.client import LLMConfigError, create_llm_client  # noqa: E402
from src.utils import load_config, results_dir, set_seed  # noqa: E402

from experiments.run_ablation import load_classifier  # noqa: E402


def detector_view(classifier) -> list[dict]:
    """Would the DETECTION layers even notice each adaptive payload?

    Separating this from end-to-end ASR matters: an attack can be undetected and still
    fail (gating stopped it), or detected and still succeed (the model was already fooled
    before the block took effect).
    """
    rows = []
    for template in ADAPTIVE_TEMPLATES:
        text = _payload_text(template, "session id")
        h = analyze(text)
        clf = classifier.score(text)
        threshold = getattr(classifier, "threshold", 0.5)
        rows.append({
            "attack_id": template.id, "family": template.family,
            "heuristic_score": round(h.score, 3),
            "heuristic_flagged": int(h.score >= 0.6),
            "classifier_score": round(clf, 3),
            "classifier_flagged": int(clf >= threshold),
            "top_reason": (h.reasons[0] if h.reasons else ""),
        })
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["anthropic", "ollama", "mock"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="before", help="tag for before/after firewall changes")
    ap.add_argument("--classifier", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(overrides=list(args.set) + ([f"llm.backend={args.backend}"] if args.backend else []))
    set_seed(cfg["seed"])
    try:
        llm = create_llm_client(cfg)
    except LLMConfigError as err:
        sys.exit(f"LLM configuration error: {err}")
    classifier = load_classifier(args.classifier)

    # 1. Detection view: which layers see the payload at all.
    print(f"\n{'=' * 92}\nADAPTIVE ATTACKS vs THE DETECTION LAYERS\n{'=' * 92}")
    print(f"{'attack':30s} {'family':22s} {'heur':>6s} {'clf':>7s}  flagged by")
    detection = detector_view(classifier)
    for row in detection:
        flagged = [n for n, k in (("heuristics", "heuristic_flagged"),
                                  ("classifier", "classifier_flagged")) if row[k]]
        print(f"{row['attack_id']:30s} {row['family']:22s} {row['heuristic_score']:>6.2f} "
              f"{row['classifier_score']:>7.3f}  {', '.join(flagged) or 'NOTHING'}")
    missed = [r for r in detection if not r["heuristic_flagged"] and not r["classifier_flagged"]]
    p = wilson_interval(len(missed), len(detection))
    print(f"\nundetected by BOTH detection layers: {p}")

    # 2. End-to-end: does the attack actually achieve its goal against the full firewall?
    configs = [Configuration("no_firewall", None),
               Configuration("all_layers", LayerSwitches(), Mode.SANITIZE)]
    records = []
    for config in configs:
        for i in range(args.runs):
            for template in ADAPTIVE_TEMPLATES:
                records.append(run_attack(config, template, cfg["agent"], llm,
                                          cfg["seed"] + i, classifier))

    print(f"\n{'=' * 92}\nADAPTIVE ATTACK SUCCESS (end to end)\n{'=' * 92}")
    print(f"{'family':24s} {'no firewall':30s} {'all layers':30s}")
    families = sorted({t.family for t in ADAPTIVE_TEMPLATES})
    for family in families:
        cells = []
        for config in configs:
            rows = [r for r in records if r.config == config.name
                    and any(t.id == r.item_id and t.family == family for t in ADAPTIVE_TEMPLATES)]
            cells.append(str(wilson_interval(sum(r.success for r in rows), len(rows))))
        print(f"{family:24s} {cells[0]:30s} {cells[1]:30s}")

    overall = []
    for config in configs:
        rows = [r for r in records if r.config == config.name]
        overall.append(wilson_interval(sum(r.success for r in rows), len(rows)))
    print(f"{'-' * 92}\n{'OVERALL BYPASS RATE':24s} {str(overall[0]):30s} {str(overall[1]):30s}")

    out_dir = results_dir("phase7")
    rows = [asdict(r) for r in records]
    with (out_dir / f"adaptive_{args.label}_{llm.name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"adaptive_{args.label}_{llm.name}.json").write_text(json.dumps({
        "label": args.label, "model": llm.model, "runs": args.runs,
        "detection": detection,
        "undetected_by_both": {"rate": p.rate, "ci": [p.low, p.high], "n": p.total},
        "asr": {c.name: {"rate": o.rate, "ci": [o.low, o.high]} for c, o in zip(configs, overall)},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
