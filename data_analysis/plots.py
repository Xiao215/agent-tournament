from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, List, Iterable

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set modern style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color palette for consistency
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'info': '#2196F3',
    'light': '#F5F5F5',
    'dark': '#333333'
}

# Create custom colormap
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#FF9800']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

def setup_plot_style():
    """Configure global plot styling"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    })

setup_plot_style()

def add_watermark(ax, text="Agent Tournament Analysis"):
    """Add a subtle watermark to the plot"""
    ax.text(0.99, 0.01, text, transform=ax.transAxes, fontsize=8,
            color='gray', alpha=0.5, ha='right', va='bottom')

def improve_legend(ax, **kwargs):
    """Improve legend styling"""
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, edgecolor='gray', **kwargs)
    if legend:
        legend.get_frame().set_facecolor('white')
    return legend

def set_title_with_subtitle(ax, title, subtitle=None, **kwargs):
    """Set a title with optional subtitle"""
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    else:
        full_title = title
    ax.set_title(full_title, fontweight='bold', **kwargs)

def add_grid_style(ax, alpha=0.3):
    """Add consistent grid styling"""
    ax.grid(True, alpha=alpha, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)


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
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [m for m, _ in items_sorted]
        values = [v for _, v in items_sorted]

        # Create gradient colors based on values
        colors = [custom_cmap(v) for v in values]

        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.85)

        # Add value labels on bars while keeping them within [0, 1]
        y_max = 1.0
        offset = 0.02
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height + offset <= y_max:
                y_pos = height + offset
                va = "bottom"
            else:
                y_pos = max(height - offset, offset)
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontweight="bold",
                fontsize=9,
            )

        ax.set_ylim(0, y_max)
        ax.set_title(f"Mechanism Effectiveness (Average Cooperation)\n{game}",
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel("Average Cooperation Rate", fontweight='bold')
        ax.set_xlabel("Mechanism", fontweight='bold')

        # Improved tick styling
        plt.xticks(rotation=35, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        # Add subtle background gradient
        ax.set_facecolor('#FAFAFA')

        out_path = Path(out_dir) / "figures" / f"mechanism_effectiveness_{game}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
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
        # Horizontal bar chart by expected payoff
        items_sorted = sorted(items, key=lambda t: (-t[1], t[0]))
        fig, ax = plt.subplots(figsize=(10, max(6, len(items_sorted) * 0.5)))
        labels = [a for a, _, _ in items_sorted]
        pays = [p for _, p, _ in items_sorted]

        # Create gradient colors for bars
        norm_pays = np.array(pays)
        if norm_pays.max() > norm_pays.min():
            normalized = (norm_pays - norm_pays.min()) / (norm_pays.max() - norm_pays.min())
        else:
            normalized = np.ones_like(norm_pays) * 0.5
        colors = [custom_cmap(n) for n in normalized]

        bars = ax.barh(labels, pays, color=colors, edgecolor='white', linewidth=1, alpha=0.85)

        # Add value labels on bars
        for bar, pay in zip(bars, pays):
            width = bar.get_width()
            ax.text(width + max(pays) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{pay:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

        ax.set_title(f"Agent Expected Payoff\n{game} | {mechanism}",
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Expected Payoff", fontweight='bold')
        ax.set_ylabel("Agent", fontweight='bold')

        # Style improvements
        ax.set_facecolor('#FAFAFA')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        out_path = Path(out_dir) / "figures" / f"agent_payoffs_{game}_{mechanism}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
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
            fig2, ax2 = plt.subplots(figsize=(8, 6))

            # Enhanced scatter plot with size based on payoff
            sizes = np.array(ys)
            if sizes.max() > sizes.min():
                norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
                point_sizes = 80 + norm_sizes * 120  # Scale from 80 to 200
            else:
                point_sizes = [100] * len(sizes)

            scatter = ax2.scatter(xs, ys, c=ys, s=point_sizes, alpha=0.7,
                                cmap=custom_cmap, edgecolors='white', linewidth=1.5)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Expected Payoff', fontweight='bold')

            # Better annotations with background boxes
            for x, y, lab in zip(xs, ys, labs):
                ax2.annotate(lab, (x, y), fontsize=9, fontweight='bold',
                           xytext=(5, 5), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                                   edgecolor='gray', alpha=0.8))

            ax2.set_xlim(-0.05, 1.05)
            ax2.set_xlabel("Cooperation Rate", fontweight='bold')
            ax2.set_ylabel("Expected Payoff", fontweight='bold')
            ax2.set_title(f"Payoff vs Cooperation Analysis\n{game} | {mechanism}",
                         fontsize=16, fontweight='bold', pad=20)

            # Grid and styling
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#FAFAFA')

            plt.tight_layout()
            out_path2 = Path(out_dir) / "figures" / f"agent_payoff_vs_coop_{game}_{mechanism}.png"
            fig2.savefig(out_path2, dpi=300, bbox_inches='tight', facecolor='white')
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


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _avg(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not (v != v)]
    return sum(vals) / len(vals) if vals else float("nan")


def plot_mechanism_summary(summary_csv: str | Path, out_dir: str | Path) -> None:
    rows = _load_csv(Path(summary_csv))
    if not rows:
        return

    by_game: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        game = row.get("game") or "Unknown"
        by_game.setdefault(game, []).append(row)

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for game, items in by_game.items():
        labels = [item.get("mechanism", "Unknown") for item in items]
        payoffs = [_safe_float(item.get("avg_expected_payoff")) for item in items]
        coop = [_safe_float(item.get("avg_cooperation_rate")) for item in items]
        delta = [_safe_float(item.get("avg_delta_expected_payoff")) for item in items]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.bar(x - width / 2, payoffs, width, label="Avg Payoff", color=COLOR_PALETTE['primary'], alpha=0.8)
        ax1.bar(x + width / 2, coop, width, label="Avg Cooperation", color=COLOR_PALETTE['success'], alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_ylabel("Value", fontweight='bold')
        ax1.set_title(f"Mechanism Summary – {game}", fontweight='bold', fontsize=16)
        ax1.legend()
        ax1.grid(alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

        ax2 = ax1.twinx()
        ax2.plot(x, delta, color=COLOR_PALETTE['accent1'], marker='o', linewidth=2, label="Δ Payoff vs Base")
        ax2.set_ylabel("Δ Payoff", color=COLOR_PALETTE['accent1'], fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE['accent1'])

        lines, labels_leg = ax1.get_legend_handles_labels()
        l2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + l2, labels_leg + labels2, loc='upper left')

        fig.tight_layout()
        fig.savefig(fig_dir / f"mechanism_summary_{game}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def plot_agent_metrics(agent_csv: str | Path, out_dir: str | Path) -> None:
    rows = _load_csv(Path(agent_csv))
    if not rows:
        return

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scatter: expected payoff vs cooperation rate colored by mechanism
    mechanisms = sorted(set(row.get("mechanism", "Unknown") for row in rows))
    mech_colors = {mech: plt.get_cmap("tab10")(i % 10) for i, mech in enumerate(mechanisms)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        mech = row.get("mechanism", "Unknown")
        ax.scatter(
            _safe_float(row.get("expected_payoff")),
            _safe_float(row.get("coop_rate")),
            color=mech_colors.get(mech, COLOR_PALETTE['primary']),
            alpha=0.7,
            label=mech,
        )
        ax.annotate(
            row.get("agent", ""),
            (_safe_float(row.get("expected_payoff")), _safe_float(row.get("coop_rate"))),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("Expected Payoff", fontweight='bold')
    ax.set_ylabel("Cooperation Rate", fontweight='bold')
    ax.set_title("Agent Payoff vs Cooperation", fontweight='bold', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='best', frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "agent_payoff_vs_coop.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Horizontal bars of expected payoff per mechanism
    clusters: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        mech = row.get("mechanism", "Unknown")
        clusters[mech].append(_safe_float(row.get("expected_payoff")))

    fig, ax = plt.subplots(figsize=(10, max(4, len(clusters) * 0.8)))
    mechanisms = sorted(clusters)
    means = [_avg(clusters[m]) for m in mechanisms]
    bars = ax.barh(mechanisms, means, color=COLOR_PALETTE['primary'], alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va='center', fontsize=9)
    ax.set_xlabel("Average Expected Payoff", fontweight='bold')
    ax.set_title("Average Agent Payoff by Mechanism", fontweight='bold', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "agent_payoff_by_mechanism.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
