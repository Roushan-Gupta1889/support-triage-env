"""
visualize_trajectory.py

Generates a learning curve visualization from the trajectory logs.
Run `python visualize_trajectory.py --demo` to generate from representative baseline data.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# Representative RL agent behavior traversing the dense reward space
# (Stagnation penalties manifest as small drops, tool checks provide +0.05 boosts)
DEMO_DATA = {
    "ticket_category": [0.0, 1.0],
    "ticket_priority": [0.0, 0.45, 0.40, 1.0],
    "full_resolution": [0.0, 0.35, 0.65, 0.65, 0.85, 0.80, 1.0],
    "escalation_detection": [0.0, 0.05, 0.4, 0.35, 0.70, 0.65, 0.95, 0.90, 1.0],
}

def _records_to_task_rewards(records):
    tasks = {}
    for data in records:
        try:
            if data.get("event") == "episode_end":
                continue
            task = data.get("task")
            if task:
                tasks.setdefault(task, []).append(float(data.get("reward", 0.0)))
        except (TypeError, ValueError):
            pass
    return tasks


def load_data(filepath=None):
    if filepath:
        paths = [filepath]
    else:
        paths = ["trajectory.json", "trajectory.jsonl"]

    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                if isinstance(blob, list):
                    recs = blob
                elif isinstance(blob, dict) and "records" in blob:
                    recs = blob["records"]
                else:
                    continue
                data = _records_to_task_rewards(recs)
                if data:
                    return data
            else:
                tasks = {}
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get("event") == "episode_end":
                                continue
                            task = data.get("task")
                            if task:
                                tasks.setdefault(task, []).append(
                                    float(data.get("reward", 0.0))
                                )
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                if tasks:
                    return tasks
        except (OSError, json.JSONDecodeError):
            continue
    return None

def plot_learning_curve(data, output_file="learning_curve.svg"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {
        "ticket_category": "#2ea043",        # GitHub green
        "ticket_priority": "#e3b341",        # GitHub yellow
        "full_resolution": "#d29922",        # GitHub orange
        "escalation_detection": "#f85149"    # GitHub red
    }
    
    markers = {
        "ticket_category": "o",
        "ticket_priority": "s",
        "full_resolution": "^",
        "escalation_detection": "D"
    }

    for task, rewards in data.items():
        if not rewards:
            continue
        xs = list(range(1, len(rewards) + 1))
        ax.plot(
            xs, rewards, 
            label=task, 
            color=colors.get(task, "white"), 
            linewidth=2.5,
            marker=markers.get(task, "o"), 
            markersize=8,
            alpha=0.85
        )
        
    ax.set_title("Agent Learning Story: Reward Trajectory per Episode", fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xlabel("Episode Step (Reasoning + Environment Feedback Iteration)", fontsize=12, color='#cccccc')
    ax.set_ylabel("Dense Reward Score", fontsize=12, color='#cccccc')
    ax.text(
        0.02,
        0.02,
        "Dense shaping: upward steps = partial credit & tool alignment; dips = stagnation / correction",
        transform=ax.transAxes,
        fontsize=9,
        color="#8b949e",
        verticalalignment="bottom",
    )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors='#cccccc')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    
    ax.legend(title="Target Tasks", loc="lower right", facecolor='#161b22', edgecolor='#30363d', fontsize=10, title_fontsize=11)
    
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    plt.tight_layout()
    fmt = "svg" if output_file.lower().endswith(".svg") else "png"
    plt.savefig(output_file, facecolor=fig.get_facecolor(), edgecolor="none", format=fmt)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Path to trajectory.json or trajectory.jsonl (default: try both)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="learning_curve.svg",
        help="Output image path (.svg or .png)",
    )
    args = parser.parse_args()

    data = DEMO_DATA if args.demo else load_data(args.input)
    if not data:
        print("Using demo data due to missing/empty log file.")
        data = DEMO_DATA

    plot_learning_curve(data, output_file=args.output)
