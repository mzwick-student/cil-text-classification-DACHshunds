from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines.analysis.table_utils import latex_escape


MODEL_LABELS = {
    "mlp": "MLP",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
}
MODEL_ORDER = ["mlp", "textcnn", "bilstm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the compact report table for fastText DNN baselines."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--caption", default="DNN-based baseline classifier performance results.")
    parser.add_argument("--label", default="tab:dnn-baselines")
    return parser.parse_args()


def mean_std(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = row[f"{metric}_mean"]
    std = row[f"{metric}_std"]
    if pd.isna(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "seed" not in df.columns:
        df.insert(0, "seed", 0)

    summary = (
        df.groupby("model")[
            ["best_epoch", "best_epoch_train_loss", "best_epoch_val_loss", "mae", "cil_score"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column[0] if column[1] == "" else f"{column[0]}_{column[1]}"
        for column in summary.columns.to_flat_index()
    ]
    summary["order"] = summary["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    summary = summary.sort_values("order").drop(columns=["order"])

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
        print(f"Wrote {args.summary_csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(args.caption)}}}",
        rf"\label{{{latex_escape(args.label)}}}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Best epoch & Train loss & Val. loss & Val. MAE & CIL-score \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        values = [
            latex_escape(MODEL_LABELS.get(row["model"], str(row["model"]))),
            mean_std(row, "best_epoch", digits=1),
            mean_std(row, "best_epoch_train_loss"),
            mean_std(row, "best_epoch_val_loss"),
            mean_std(row, "mae"),
            mean_std(row, "cil_score"),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

