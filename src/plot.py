import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import FIGURE_DIR

def plot_ipd_results(game):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M")
    dated_dir = FIGURE_DIR / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
    graph_name = f"{game.agent1} vs {game.agent2}"

    # === Score Trend ===
    score_dir = dated_dir / "score_trend"
    os.makedirs(score_dir, exist_ok=True)
    score_path = score_dir / f"{graph_name}.png"

    rounds = list(range(1, len(game.scores_p1) + 1))
    cumulative_p1 = np.cumsum(game.scores_p1)
    cumulative_p2 = np.cumsum(game.scores_p2)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, cumulative_p1, label=str(game.agent1), linewidth=2)
    plt.plot(rounds, cumulative_p2, label=str(game.agent2), linewidth=2)
    plt.title("Cumulative Score Over Iterations")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(score_path)
    plt.close()

    # === Joint Action Frequencies ===
    bar_dir = dated_dir / "joint_actions"
    os.makedirs(bar_dir, exist_ok=True)
    bar_path = bar_dir / f"{graph_name}.png"

    agent1_name = str(game.agent1)
    agent2_name = str(game.agent2)

    joint_counts = {
        ("C", "C"): 0,
        ("C", "D"): 0,
        ("D", "C"): 0,
        ("D", "D"): 0,
    }

    for a1, a2 in zip(game.actions_p1, game.actions_p2):
        joint_counts[(str(a1), str(a2))] += 1

    labels = [f"{a1} vs {a2}" for (a1, a2) in joint_counts]
    values = list(joint_counts.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values)

    plt.title(f"Joint Action Distribution  ({agent1_name})  ({agent2_name})")
    plt.ylabel("Count")
    plt.xlabel("Action by Agent 1 vs Agent 2")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()