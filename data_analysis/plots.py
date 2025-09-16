from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import csv
import math
import matplotlib.pyplot as plt


def plot_mechanism_effectiveness(csv_path: str | Path, out_dir: str | Path) -> None:
    # Read rows
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    # Aggregate mean cooperation by (game, mechanism)
    sums: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        game = (r.get("game") or "").strip()
        mech = (r.get("mechanism") or "").strip()
        if not game or not mech:
            continue
        try:
            val = float(r.get("coop_average", 0.0))
        except Exception:
            continue
        key = (game, mech)
        sums[key] = sums.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1

    by_game: Dict[str, List[Tuple[str, float]]] = {}
    for (game, mech), s in sums.items():
        avg = s / max(counts[(game, mech)], 1)
        by_game.setdefault(game, []).append((mech, avg))

    for game, items in by_game.items():
        items_sorted = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [m for m, _ in items_sorted]
        values = [v for _, v in items_sorted]
        ax.bar(labels, values, color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_title(f"Mechanism effectiveness (avg cooperation) — {game}")
        ax.set_ylabel("Average cooperation")
        ax.set_xlabel("Mechanism")
        plt.xticks(rotation=30, ha="right")
        out_path = Path(out_dir) / "figures" / f"mechanism_effectiveness_{game}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_agent_performance(payoffs_csv: str | Path, coop_csv: str | Path, out_dir: str | Path) -> None:
    pay_rows: List[Dict[str, str]] = []
    coop_rows: List[Dict[str, str]] = []
    with open(payoffs_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pay_rows.append(row)
    with open(coop_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            coop_rows.append(row)
    if not pay_rows:
        return

    # Index coop rows for quick lookup
    coop_map: Dict[Tuple[str, str, str, str], float] = {}
    for row in coop_rows:
        key = (
            (row.get("run_dir") or "").strip(),
            (row.get("game") or "").strip(),
            (row.get("mechanism") or "").strip(),
            (row.get("agent") or "").strip(),
        )
        try:
            coop_map[key] = float(row.get("cooperation_rate", 0.0))
        except Exception:
            continue

    # Combine and aggregate per (game, mechanism, agent)
    sums_pay: Dict[Tuple[str, str, str], float] = {}
    sums_coop: Dict[Tuple[str, str, str], float] = {}
    counts: Dict[Tuple[str, str, str], int] = {}
    for row in pay_rows:
        key_run = (
            (row.get("run_dir") or "").strip(),
            (row.get("game") or "").strip(),
            (row.get("mechanism") or "").strip(),
            (row.get("agent") or "").strip(),
        )
        game, mech, agent = key_run[1], key_run[2], key_run[3]
        if not game or not mech or not agent:
            continue
        key = (game, mech, agent)
        try:
            pay = float(row.get("expected_payoff", 0.0))
        except Exception:
            continue
        coop = coop_map.get(key_run)
        if coop is None:
            # allow missing coop rows
            coop = math.nan
        sums_pay[key] = sums_pay.get(key, 0.0) + pay
        # Only accumulate coop if exists
        if not math.isnan(coop):
            sums_coop[key] = sums_coop.get(key, 0.0) + coop
        counts[key] = counts.get(key, 0) + 1

    # Organize by (game, mechanism)
    by_group: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
    for (game, mech, agent), c in counts.items():
        avg_pay = sums_pay[(game, mech, agent)] / c
        avg_coop = sums_coop.get((game, mech, agent), math.nan)
        if not math.isnan(avg_coop):
            avg_coop = avg_coop / c
        by_group.setdefault((game, mech), []).append((agent, avg_pay, avg_coop))

    for (game, mechanism), items in by_group.items():
        # Barh by expected payoff
        items_sorted = sorted(items, key=lambda t: (-t[1], t[0]))
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [a for a, _, _ in items_sorted]
        pays = [p for _, p, _ in items_sorted]
        ax.barh(labels, pays, color="#72B7B2")
        ax.set_title(f"Agent expected payoff — {game} | {mechanism}")
        ax.set_xlabel("Expected payoff")
        plt.tight_layout()
        out_path = Path(out_dir) / "figures" / f"agent_payoffs_{game}_{mechanism}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        # Scatter payoff vs cooperation where coop is available
        xs: List[float] = []
        ys: List[float] = []
        labs: List[str] = []
        for a, p, c in items:
            if c is None or math.isnan(c):
                continue
            xs.append(c)
            ys.append(p)
            labs.append(a)
        if xs:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(xs, ys, color="#E45756")
            for x, y, lab in zip(xs, ys, labs):
                ax2.annotate(lab, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Cooperation rate")
            ax2.set_ylabel("Expected payoff")
            ax2.set_title(f"Payoff vs Cooperation — {game} | {mechanism}")
            plt.tight_layout()
            out_path2 = Path(out_dir) / "figures" / f"agent_payoff_vs_coop_{game}_{mechanism}.png"
            fig2.savefig(out_path2, dpi=150)
            plt.close(fig2)


def plot_pairwise_and_trajectories(
    pairwise_csv: str | Path,
    conditional_csv: str | Path,
    trajectory_csv: str | Path,
    out_dir: str | Path,
) -> None:
    # Pairwise heatmap-like bar plots per run (simple version: grouped bars per pair)
    def read_csv(path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows

    pair_rows = read_csv(Path(pairwise_csv)) if Path(pairwise_csv).exists() else []
    cond_rows = read_csv(Path(conditional_csv)) if Path(conditional_csv).exists() else []
    traj_rows = read_csv(Path(trajectory_csv)) if Path(trajectory_csv).exists() else []

    # Pairwise grouped bars per (game, mechanism)
    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in pair_rows:
        g = (r.get("game") or "").strip()
        m = (r.get("mechanism") or "").strip()
        if not g or not m:
            continue
        by_group.setdefault((g, m), []).append(r)

    for (game, mech), items in by_group.items():
        # show top 10 pairs by rounds
        items_sorted = sorted(items, key=lambda x: -int(x.get("rounds", 0)))[:10]
        labels = [f"{r['agent_i']} vs {r['agent_j']}" for r in items_sorted]
        pay_i = [float(r.get("avg_payoff_i_vs_j", 0.0)) for r in items_sorted]
        pay_j = [float(r.get("avg_payoff_j_vs_i", 0.0)) for r in items_sorted]
        coop_i = [float(r.get("coop_rate_i_vs_j", 0.0)) for r in items_sorted]
        coop_j = [float(r.get("coop_rate_j_vs_i", 0.0)) for r in items_sorted]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        x = range(len(labels))
        axs[0].bar(x, pay_i, label="payoff_i", alpha=0.7)
        axs[0].bar(x, pay_j, bottom=pay_i, label="payoff_j", alpha=0.7)
        axs[0].set_ylabel("Avg payoff (stacked)")
        axs[0].legend()
        axs[1].bar(x, coop_i, label="coop_i", alpha=0.7)
        axs[1].bar(x, coop_j, bottom=coop_i, label="coop_j", alpha=0.7)
        axs[1].set_ylabel("Coop rate (stacked)")
        axs[1].set_xticks(list(x))
        axs[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        fig.suptitle(f"Pairwise payoff & cooperation — {game} | {mech}")
        fig.tight_layout()
        out_path = Path(out_dir) / "figures" / f"pairwise_payoff_coop_{game}_{mech}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Conditional cooperation: scatter per agent
    by_group2: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in cond_rows:
        g = (r.get("game") or "").strip()
        m = (r.get("mechanism") or "").strip()
        if not g or not m:
            continue
        by_group2.setdefault((g, m), []).append(r)
    for (game, mech), items in by_group2.items():
        xs = [float(r.get("p_coop_given_opp_D", 0.0)) for r in items]
        ys = [float(r.get("p_coop_given_opp_C", 0.0)) for r in items]
        labs = [r.get("agent") or "" for r in items]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, color="#54A24B")
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("P(C | opp D)")
        ax.set_ylabel("P(C | opp C)")
        ax.set_title(f"Conditional cooperation — {game} | {mech}")
        plt.tight_layout()
        fig.savefig(Path(out_dir) / "figures" / f"conditional_coop_{game}_{mech}.png", dpi=150)
        plt.close(fig)

    # Round trajectory: line plot of avg_pair_coop over rounds (aggregated by run)
    by_group3: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for r in traj_rows:
        g = (r.get("game") or "").strip()
        m = (r.get("mechanism") or "").strip()
        if not g or not m:
            continue
        key = (g, m)
        by_group3.setdefault(key, {}).setdefault(int(r.get("round", 0)), []).append(float(r.get("avg_pair_coop", 0.0)))
    for (game, mech), series in by_group3.items():
        xs = sorted(series.keys())
        ys = [sum(series[t]) / max(1, len(series[t])) for t in xs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", color="#E19D29")
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg pair cooperation")
        ax.set_ylim(0, 1)
        ax.set_title(f"Cooperation trajectory — {game} | {mech}")
        plt.tight_layout()
        fig.savefig(Path(out_dir) / "figures" / f"round_trajectory_{game}_{mech}.png", dpi=150)
        plt.close(fig)


