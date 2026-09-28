"""Phase 2: run the UNDEFENDED app against the full attack suite and report baseline ASR.

For every attack template we run the agent `--runs` times (LLM output is stochastic) and
check success by observable EFFECTS (goal checkers). We report ASR per category with a
Wilson 95% confidence interval, and save per-run records for later comparison.

Usage:
    python experiments/run_baseline.py --backend ollama --runs 5
    python experiments/run_baseline.py --backend mock --runs 20   # fast sanity check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from src.app.agent import EmailAssistant  # noqa: E402
from src.attacks.catalog import ALL_TEMPLATES, Category  # noqa: E402
from src.attacks.generator import attack_succeeded, build_instance  # noqa: E402
from src.eval.metrics import wilson_interval  # noqa: E402
from src.llm.client import LLMConfigError, create_llm_client  # noqa: E402
from src.utils import get_logger, load_config, results_dir, set_seed  # noqa: E402

log = get_logger("run_baseline")


def run(cfg: dict, backend: str | None, runs: int) -> None:
    overrides = [f"llm.backend={backend}"] if backend else []
    cfg = load_config(overrides=overrides) if not cfg else cfg
    set_seed(cfg["seed"])
    llm = create_llm_client(cfg)
    agent_cfg = cfg["agent"]

    records = []
    for template in tqdm(ALL_TEMPLATES, desc="attacks"):
        for i in range(runs):
            seed = cfg["seed"] + i
            instance = build_instance(template, agent_cfg, seed=seed)
            agent = EmailAssistant(llm, instance.env, agent_cfg["max_steps"],
                                   agent_cfg["assistant_name"])
            trace = agent.run(instance.task, seed=seed)
            success = attack_succeeded(instance, trace)
            records.append({
                "attack_id": template.id, "category": template.category.value,
                "goal": template.goal.value, "family": template.family,
                "delivery": template.delivery, "obfuscation": template.obfuscation,
                "seed": seed, "success": int(success), "steps": trace.steps,
                "wall_time_s": round(trace.wall_time_s, 3),
            })

    out_dir = results_dir("phase2")
    with (out_dir / f"baseline_{llm.name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    report(records, llm)
    (out_dir / f"baseline_{llm.name}.json").write_text(
        json.dumps({"model": llm.model, "runs": runs, "records": records}, indent=2),
        encoding="utf-8")
    print(f"\nSaved to {out_dir}")


def report(records: list[dict], llm) -> None:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["success"])
    print(f"\n{'=' * 70}\nBASELINE ASR (undefended)  model={llm.name}:{llm.model}\n{'=' * 70}")
    print(f"{'category':22s} {'ASR [95% CI]':30s} runs")
    for cat in Category:
        vals = by_cat.get(cat.value, [])
        if vals:
            p = wilson_interval(sum(vals), len(vals))
            print(f"{cat.value:22s} {str(p):30s} {len(vals)}")
    overall = wilson_interval(sum(r["success"] for r in records), len(records))
    print(f"{'-' * 70}\n{'OVERALL':22s} {str(overall):30s} {len(records)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["anthropic", "ollama", "mock"])
    parser.add_argument("--runs", type=int, default=5, help="runs per attack template")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(overrides=list(args.set) + ([f"llm.backend={args.backend}"] if args.backend else []))
    try:
        run(cfg, None, args.runs)
    except LLMConfigError as err:
        sys.exit(f"LLM configuration error: {err}")


if __name__ == "__main__":
    main()
