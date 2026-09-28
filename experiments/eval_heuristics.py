"""Phase 3: evaluate the heuristic layer as a DETECTOR.

Reports, at a configurable threshold:
  * detection rate per attack category (recall on injected payload text),
  * false positive rate on benign prompts AND hard negatives (reported separately,
    because the hard negatives are the ones that actually matter),
  * per-call latency (p50/p95).

Detection rate is not the same as attack success rate: this measures whether the layer
NOTICES the payload text, not whether the agent was stopped. Phases 5-7 measure the ASR
impact end to end.

Usage:
    python experiments/eval_heuristics.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks.catalog import ALL_TEMPLATES  # noqa: E402
from src.attacks.generator import _payload_text  # noqa: E402
from src.data.benign import BENIGN_PROMPTS, HARD_NEGATIVES  # noqa: E402
from src.eval.metrics import wilson_interval  # noqa: E402
from src.firewall.heuristics import analyze  # noqa: E402
from src.utils import results_dir  # noqa: E402


def score_texts(texts: list[str]) -> tuple[list[float], list[float], list[list[str]]]:
    scores, latencies, reasons = [], [], []
    for text in texts:
        start = time.perf_counter()
        result = analyze(text)
        latencies.append((time.perf_counter() - start) * 1000)
        scores.append(result.score)
        reasons.append(result.reasons)
    return scores, latencies, reasons


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    th = args.threshold

    # Positives: the payload text an attacker would place in content.
    attacks = [(t, _payload_text(t, "session id")) for t in ALL_TEMPLATES]
    atk_scores, atk_lat, atk_reasons = score_texts([p for _, p in attacks])

    # Negatives, kept in two groups.
    benign_scores, ben_lat, _ = score_texts(BENIGN_PROMPTS)
    hard_texts = [t for t, _ in HARD_NEGATIVES]
    hard_scores, hard_lat, hard_reasons = score_texts(hard_texts)

    print(f"\n{'=' * 74}\nHEURISTIC LAYER (threshold = {th})\n{'=' * 74}")

    by_cat: dict[str, list[int]] = defaultdict(list)
    for (tmpl, _), score in zip(attacks, atk_scores):
        by_cat[tmpl.category.value].append(int(score >= th))
    print(f"{'category':22s} {'detection rate [95% CI]':32s} n")
    for cat, vals in sorted(by_cat.items()):
        print(f"{cat:22s} {str(wilson_interval(sum(vals), len(vals))):32s} {len(vals)}")

    det_all = [int(s >= th) for s in atk_scores]
    print(f"{'-' * 74}\n{'ALL ATTACKS':22s} {str(wilson_interval(sum(det_all), len(det_all))):32s} {len(det_all)}")

    fp_benign = [int(s >= th) for s in benign_scores]
    fp_hard = [int(s >= th) for s in hard_scores]
    print(f"\n{'FPR benign prompts':22s} {str(wilson_interval(sum(fp_benign), len(fp_benign))):32s} {len(fp_benign)}")
    print(f"{'FPR hard negatives':22s} {str(wilson_interval(sum(fp_hard), len(fp_hard))):32s} {len(fp_hard)}")

    all_lat = atk_lat + ben_lat + hard_lat
    print(f"\nlatency p50={statistics.median(all_lat):.3f} ms  "
          f"p95={sorted(all_lat)[int(0.95 * len(all_lat))]:.3f} ms  (n={len(all_lat)})")

    missed = [(t.id, t.category.value, round(s, 2))
              for (t, _), s in zip(attacks, atk_scores) if s < th]
    print(f"\nMISSED BY HEURISTICS ({len(missed)}) - motivates the classifier + gating layers:")
    for aid, cat, score in missed:
        print(f"  {aid:24s} {cat:20s} score={score}")

    false_pos = [(tag, round(s, 2), r) for (txt, tag), s, r in
                 zip(HARD_NEGATIVES, hard_scores, hard_reasons) if s >= th]
    if false_pos:
        print(f"\nFALSE POSITIVES ON HARD NEGATIVES ({len(false_pos)}):")
        for tag, score, reasons in false_pos:
            print(f"  {tag:24s} score={score} {reasons[:2]}")

    out = results_dir("phase3") / "heuristics_eval.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "kind", "category_or_tag", "score", "flagged"])
        for (t, _), s in zip(attacks, atk_scores):
            w.writerow([t.id, "attack", t.category.value, round(s, 3), int(s >= th)])
        for p, s in zip(BENIGN_PROMPTS, benign_scores):
            w.writerow([p[:40], "benign", "benign_prompt", round(s, 3), int(s >= th)])
        for (txt, tag), s in zip(HARD_NEGATIVES, hard_scores):
            w.writerow([txt[:40], "hard_negative", tag, round(s, 3), int(s >= th)])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
