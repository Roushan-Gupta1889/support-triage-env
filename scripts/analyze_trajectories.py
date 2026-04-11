"""
Aggregate metrics from trajectory.jsonl (or .json). Run from repo root:

  python scripts/analyze_trajectories.py -i trajectory.jsonl
  python scripts/analyze_trajectories.py -i trajectory.json -o report.csv --markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple


def load_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, list):
            return blob
        if isinstance(blob, dict) and "records" in blob:
            return blob["records"]
        raise ValueError("trajectory.json must be a JSON array or {records: [...]}")
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def aggregate(records: List[dict]) -> Tuple[Dict[str, dict], dict]:
    step_rewards: DefaultDict[str, List[float]] = defaultdict(list)
    ends: DefaultDict[str, List[dict]] = defaultdict(list)

    for r in records:
        task = r.get("task")
        if not task:
            continue
        if r.get("event") == "episode_end":
            ends[task].append(r)
        elif "reward" in r:
            step_rewards[task].append(float(r["reward"]))

    per_task: Dict[str, dict] = {}
    all_scores: List[float] = []
    all_success = 0
    all_eps = 0

    for task in sorted(set(ends.keys()) | set(step_rewards.keys())):
        ep_list = ends.get(task, [])
        sr_list = step_rewards.get(task, [])
        n_ep = len(ep_list)
        scores = [float(e.get("score", 0.0)) for e in ep_list]
        succ = sum(1 for e in ep_list if e.get("success"))
        mean_score = statistics.mean(scores) if scores else 0.0
        success_rate = succ / n_ep if n_ep else 0.0
        mean_step_reward = statistics.mean(sr_list) if sr_list else 0.0

        per_task[task] = {
            "task": task,
            "episodes": n_ep,
            "success_rate": success_rate,
            "mean_final_score": mean_score,
            "mean_step_reward": mean_step_reward,
            "total_steps_logged": len(sr_list),
        }
        all_scores.extend(scores)
        all_success += succ
        all_eps += n_ep

    global_metrics = {
        "episodes_total": all_eps,
        "overall_success_rate": all_success / all_eps if all_eps else 0.0,
        "overall_mean_final_score": statistics.mean(all_scores) if all_scores else 0.0,
    }
    return per_task, global_metrics


def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def format_markdown(per_task: Dict[str, dict], global_metrics: dict) -> str:
    lines = [
        "### Trajectory summary",
        "",
        f"- Episodes (total): **{global_metrics['episodes_total']}**",
        f"- Overall success rate: **{global_metrics['overall_success_rate']:.2%}**",
        f"- Overall mean final score: **{global_metrics['overall_mean_final_score']:.3f}**",
        "",
        "| Task | Episodes | Success rate | Mean final score | Mean step reward |",
        "|------|---------:|-------------:|-----------------:|-----------------:|",
    ]
    for t in sorted(per_task.keys()):
        r = per_task[t]
        lines.append(
            f"| `{r['task']}` | {r['episodes']} | {r['success_rate']:.2%} | "
            f"{r['mean_final_score']:.3f} | {r['mean_step_reward']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="trajectory.jsonl")
    ap.add_argument("-o", "--output", default="trajectory_summary.csv")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    records = load_records(args.input)
    per_task, global_metrics = aggregate(records)
    rows = list(per_task.values())
    write_csv(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} tasks)")

    if args.markdown:
        md_path = os.path.splitext(args.output)[0] + ".md"
        text = format_markdown(per_task, global_metrics)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
