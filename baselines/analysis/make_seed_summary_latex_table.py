from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines.analysis.table_utils import latex_escape


METRICS = ["cil_score", "mae", "accuracy", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LaTeX mean/std table from multi-seed baseline results."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", nargs="+", required=True)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    return parser.parse_args()


def format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    metrics = [metric for metric in METRICS if metric in df.columns]
    summary = df.groupby(args.group_by, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column[0] if column[1] == "" else f"{column[0]}_{column[1]}"
        for column in summary.columns.to_flat_index()
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(args.group_by) + "r" * len(metrics)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(args.caption)}}}",
        rf"\label{{{latex_escape(args.label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join([latex_escape(column) for column in args.group_by] + [latex_escape(metric) for metric in metrics]) + r" \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        group_values = [latex_escape(str(row[column])) for column in args.group_by]
        metric_values = [
            format_mean_std(float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]))
            for metric in metrics
        ]
        lines.append(" & ".join(group_values + metric_values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
