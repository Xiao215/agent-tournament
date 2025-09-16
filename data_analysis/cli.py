from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate
from .plots import plot_agent_performance, plot_mechanism_effectiveness


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

    args = p.parse_args()

    if args.cmd == "summarize":
        aggregate(args.root, args.out)
    elif args.cmd == "plot":
        mech_csv = Path(args.tables_root) / "mechanism_effectiveness.csv"
        pay_csv = Path(args.tables_root) / "agent_expected_payoffs.csv"
        coop_csv = Path(args.tables_root) / "agent_cooperation_rates.csv"
        plot_mechanism_effectiveness(mech_csv, args.out)
        plot_agent_performance(pay_csv, coop_csv, args.out)


if __name__ == "__main__":
    main()


