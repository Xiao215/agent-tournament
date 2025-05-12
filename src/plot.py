import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURE_DIR

def plot_ipd_results(game):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M")
    graph_name = f"{game.agent1} vs {game.agent2} - {timestamp}"

    # === Score Trend ===
    os.makedirs(FIGURE_DIR / "score_trend", exist_ok=True)
    score_path = FIGURE_DIR / "score_trend" / f"{graph_name}.png"

    rounds = list(range(1, len(game.scores_p1) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, game.scores_p1, label=str(game.agent1), linewidth=2)
    plt.plot(rounds, game.scores_p2, label=str(game.agent2), linewidth=2)
    plt.title("Cumulative Score Over Iterations")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(score_path)
    plt.close()

    # === Joint Action Frequencies ===
    os.makedirs(FIGURE_DIR / "joint_actions", exist_ok=True)
    bar_path = FIGURE_DIR / "joint_actions" / f"{graph_name}.png"

    joint_counts = {
        "C vs C": 0,
        "C vs D": 0,
        "D vs C": 0,
        "D vs D": 0,
    }

    for a1, a2 in zip(game.actions_p1, game.actions_p2):
        key = f"{a1} vs {a2}"
        joint_counts[key] += 1

    labels = list(joint_counts.keys())
    values = [joint_counts[label] for label in labels]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values)
    plt.title("Distribution of Joint Actions")
    plt.ylabel("Count")
    plt.xlabel("Joint Action")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()