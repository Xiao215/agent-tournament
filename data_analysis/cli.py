from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate
from .plots import (
    plot_agent_performance,
    plot_mechanism_effectiveness,
    plot_pairwise_and_trajectories,
)
from .advanced import (
    pairwise_metrics_for_run,
    conditional_cooperation_for_run,
    round_trajectory_for_run,
    selection_trajectory_for_run,
    baseline_deltas,
)
from .aggregate import write_csv


def main() -> None:
    p = argparse.ArgumentParser(description="LLM Tournament Data Analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize", help="Aggregate runs to CSV/JSON summaries")
    p_sum.add_argument("--root", type=Path, default=Path("results"))
    p_sum.add_argument("--out", type=Path, default=Path("outputs/analysis"))

    p_plot = sub.add_parser("plot", help="Generate plots from aggregated CSVs")
    p_plot.add_argument("--out", type=Path, default=Path("outputs/analysis/figures"))
    p_plot.add_argument(
        "--tables-root",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing generated CSV files",
    )

    p_beh = sub.add_parser("behavior", help="Agent vs payoff/coop matrices and conditional stats")
    p_beh.add_argument("--root", type=Path, default=Path("outputs"))
    p_beh.add_argument("--out", type=Path, default=Path("outputs/analysis"))

    p_sel = sub.add_parser("selection", help="Replicator trajectory from payoffs.json")
    p_sel.add_argument("--root", type=Path, default=Path("outputs"))
    p_sel.add_argument("--out", type=Path, default=Path("outputs/analysis"))

    p_base = sub.add_parser("baseline", help="Baseline deltas between mechanisms")
    p_base.add_argument("--root", type=Path, default=Path("outputs"))
    p_base.add_argument("--out", type=Path, default=Path("outputs/analysis"))

    args = p.parse_args()

    if args.cmd == "summarize":
        aggregate(args.root, args.out)
    elif args.cmd == "plot":
        mech_csv = Path(args.tables_root) / "mechanism_effectiveness.csv"
        pay_csv = Path(args.tables_root) / "agent_expected_payoffs.csv"
        coop_csv = Path(args.tables_root) / "agent_cooperation_rates.csv"
        plot_mechanism_effectiveness(mech_csv, args.out)
        plot_agent_performance(pay_csv, coop_csv, args.out)
    elif args.cmd == "behavior":
        from .io import find_runs as _fr
        # Accept either a specific run dir, a day dir, or outputs root
        runs = _fr(args.root)
        all_pair: list[dict] = []
        all_cond: list[dict] = []
        all_traj: list[dict] = []
        for run in runs:
            all_pair.extend(pairwise_metrics_for_run(run))
            all_cond.extend(conditional_cooperation_for_run(run))
            all_traj.extend(round_trajectory_for_run(run))
        p_csv = Path(args.out) / "pairwise_metrics.csv"
        c_csv = Path(args.out) / "conditional_cooperation.csv"
        t_csv = Path(args.out) / "round_trajectory.csv"
        write_csv(all_pair, p_csv)
        write_csv(all_cond, c_csv)
        write_csv(all_traj, t_csv)
        # Generate plots for these
        plot_pairwise_and_trajectories(p_csv, c_csv, t_csv, args.out)
    elif args.cmd == "selection":
        from .io import find_runs as _fr
        runs = _fr(args.root)
        all_rows: list[dict] = []
        for run in runs:
            all_rows.extend(selection_trajectory_for_run(run))
        write_csv(all_rows, Path(args.out) / "selection_trajectory.csv")
    elif args.cmd == "baseline":
        rows = baseline_deltas(args.root)
        write_csv(rows, Path(args.out) / "baseline_deltas.csv")


if __name__ == "__main__":
    main()


