#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot evaluation total and per-agent rewards from evaluation_report.csv."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to evaluation_report.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <csv_dir>/evaluation_reward_plot.png",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else csv_path.parent / "evaluation_reward_plot.png"
    )

    episodes = []
    total_rewards = []
    agent_rewards = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = int(row["episode"])
            episodes.append(episode)
            total_rewards.append(float(row["total_reward"]))

            for key, value in row.items():
                if key.startswith("agent") and key.endswith("_reward"):
                    agent_rewards.setdefault(key, []).append(float(value))

    try:
        plt.style.use("ggplot")
    except OSError:
        pass

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(episodes, total_rewards, color="#1f77b4", linewidth=2)
    axes[0].set_title("Evaluation Total Reward")
    axes[0].set_ylabel("Reward")

    for key in sorted(agent_rewards):
        axes[1].plot(episodes, agent_rewards[key], linewidth=1.6, label=key.replace("_reward", ""))

    axes[1].set_title("Evaluation Per-Agent Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].legend(ncol=3, frameon=True)

    fig.suptitle("Evaluation Reward Curves", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
