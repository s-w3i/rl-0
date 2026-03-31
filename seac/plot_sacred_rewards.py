#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot total and per-agent rewards from a Sacred metrics.json file."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a Sacred run directory containing metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <run_dir>/reward_plot.png",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    output_path = args.output.resolve() if args.output else run_dir / "reward_plot.png"

    metrics = load_metrics(metrics_path)

    total_key = "episode_reward"
    agent_keys = sorted(
        [key for key in metrics if key.startswith("agent") and key.endswith("/episode_reward")]
    )

    if total_key not in metrics:
        raise SystemExit(f"Missing '{total_key}' in {metrics_path}")
    if not agent_keys:
        raise SystemExit(f"No agent reward metrics found in {metrics_path}")

    try:
        plt.style.use("ggplot")
    except OSError:
        pass
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    total = metrics[total_key]
    axes[0].plot(total["steps"], total["values"], color="#1f77b4", linewidth=2)
    axes[0].set_title("Total Episode Reward")
    axes[0].set_ylabel("Reward")

    for key in agent_keys:
        agent_label = key.split("/")[0]
        series = metrics[key]
        axes[1].plot(series["steps"], series["values"], linewidth=1.8, label=agent_label)

    axes[1].set_title("Per-Agent Episode Reward")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Reward")
    axes[1].legend(ncol=3, frameon=True)

    fig.suptitle(f"Sacred Reward Curves: {run_dir.name}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
