"""
Reward-driven hyperparameter sweep over the episodic agent. Run from repo root:

  python scripts/train_baseline.py
  python scripts/train_baseline.py --tasks ticket_category,ticket_priority --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openai import OpenAI

import inference as inf


PRESETS: List[Tuple[str, inf.AgentHyperparams]] = [
    (
        "A_conservative",
        inf.AgentHyperparams(
            base_temperature=0.1,
            max_steps_cap=12,
            stagnation_hint_after=3,
        ),
    ),
    (
        "B_defaultish",
        inf.AgentHyperparams(
            base_temperature=0.2,
            max_steps_cap=16,
            stagnation_hint_after=2,
        ),
    ),
    (
        "C_exploratory",
        inf.AgentHyperparams(
            base_temperature=0.35,
            max_steps_cap=20,
            stagnation_hint_after=1,
        ),
    ),
]


async def _evaluate_config(
    client,
    name: str,
    hyp: inf.AgentHyperparams,
    tasks: Sequence[str],
    seed: int,
) -> dict:
    scores: List[float] = []
    successes = 0
    per_task: List[dict] = []

    for task in tasks:
        res = await inf.run_one_task(
            client,
            task,
            hyp=hyp,
            seed=seed,
            emit_logs=False,
            write_trajectory=False,
        )
        scores.append(res.score)
        successes += 1 if res.success else 0
        per_task.append(
            {
                "task": res.task,
                "score": res.score,
                "success": res.success,
                "steps": res.steps,
            }
        )

    mean_score = statistics.mean(scores) if scores else 0.0
    return {
        "name": name,
        "mean_score": mean_score,
        "mean_success_rate": successes / len(tasks) if tasks else 0.0,
        "hyperparams": asdict(hyp),
        "per_task": per_task,
    }


async def main_async(args: argparse.Namespace) -> None:
    if not inf.HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required (same as inference.py).")

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    if not tasks:
        raise ValueError("No tasks given.")

    client = OpenAI(base_url=inf.API_BASE_URL, api_key=inf.HF_TOKEN)

    results: List[dict] = []
    for name, hyp in PRESETS:
        row = await _evaluate_config(client, name, hyp, tasks, args.seed)
        results.append(row)

    results.sort(key=lambda r: r["mean_score"], reverse=True)
    best = results[0]

    out_path = args.output
    payload = {
        "seed": args.seed,
        "tasks": list(tasks),
        "ranked": results,
        "best": best["name"],
        "best_mean_score": best["mean_score"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Config search (mean final grader score over tasks)")
    print(f"{'Config':<18} {'mean_score':>12} {'success_rate':>14}")
    for r in results:
        print(
            f"{r['name']:<18} {r['mean_score']:12.3f} {r['mean_success_rate']:14.2%}"
        )
    print(f"\nBest: {best['name']} (mean_score={best['mean_score']:.3f})")
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter sweep for episodic agent")
    p.add_argument(
        "--tasks",
        default=",".join(inf.DEFAULT_TASKS),
        help="Comma-separated task ids (default: all four)",
    )
    p.add_argument("--seed", type=int, default=inf.SEED)
    p.add_argument(
        "--output",
        default="train_baseline_results.json",
        help="JSON output path",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
