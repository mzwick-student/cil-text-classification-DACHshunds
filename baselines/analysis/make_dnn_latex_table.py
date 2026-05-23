from __future__ import annotations

import argparse
from pathlib import Path

from baselines.analysis.table_utils import (
    add_total_seconds,
    fastest_by_group,
    ordered_columns,
    read_ok_results,
    write_latex_table,
)


FASTEST_COLUMNS = [
    "seed",
    "model",
    "embedding",
    "representation",
    "best_epoch",
    "cil_score",
    "macro_f1",
    "mae",
    "total_seconds",
]
FASTEST_HEADERS = [
    "Seed",
    "Model",
    "Embedding",
    "Representation",
    "Best epoch",
    "CIL-score",
    "Macro-F1",
    "MAE",
    "Seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LaTeX tables from classic DNN baseline results."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-output", type=Path, default=None)
    parser.add_argument("--group-by", nargs="+", default=["model"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name("dnn_fastest_families.tex")
    results = read_ok_results(args.input)
    fastest = fastest_by_group(results, args.group_by, ["train_seconds"])
    fastest = fastest.sort_values("total_seconds", ascending=True).reset_index(drop=True)
    columns = ordered_columns(fastest, FASTEST_COLUMNS)
    headers = [FASTEST_HEADERS[FASTEST_COLUMNS.index(column)] for column in columns]
    write_latex_table(
        fastest,
        output,
        columns,
        headers,
        caption="Fastest DNN baseline per model family.",
        label="tab:dnn-fastest-families",
    )
    print(f"Wrote {output}")

    if args.full_output:
        full = add_total_seconds(results, ["train_seconds"])
        full = full.sort_values(["model", "total_seconds", "cil_score"], ascending=[True, True, False])
        columns = ordered_columns(full, FASTEST_COLUMNS)
        headers = [FASTEST_HEADERS[FASTEST_COLUMNS.index(column)] for column in columns]
        write_latex_table(
            full,
            args.full_output,
            columns,
            headers,
            caption="All DNN baseline validation results.",
            label="tab:dnn-full-results",
        )
        print(f"Wrote {args.full_output}")


if __name__ == "__main__":
    main()
