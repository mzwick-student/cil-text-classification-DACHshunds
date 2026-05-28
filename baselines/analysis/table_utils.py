from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


METRIC_COLUMNS = {
    "accuracy",
    "macro_f1",
    "mae",
    "cil_score",
    "quadratic_weighted_kappa",
}

TIME_COLUMNS = {
    "train_seconds",
    "predict_seconds",
    "eval_seconds",
    "total_seconds",
}


def read_ok_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df.reset_index(drop=True)


def add_total_seconds(df: pd.DataFrame, time_columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    available = [column for column in time_columns if column in df.columns]
    if available:
        df["total_seconds"] = df[available].fillna(0.0).sum(axis=1)
    else:
        df["total_seconds"] = pd.NA
    return df


def fastest_by_group(
    df: pd.DataFrame,
    group_columns: list[str],
    time_columns: Iterable[str],
) -> pd.DataFrame:
    df = add_total_seconds(df, time_columns)
    sort_columns = group_columns + ["total_seconds", "cil_score"]
    ascending = [True] * len(group_columns) + [True, False]
    df = df.sort_values(sort_columns, ascending=ascending)
    return df.groupby(group_columns, as_index=False).first().reset_index(drop=True)


def ordered_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def format_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "--"
    if column in METRIC_COLUMNS:
        return f"{float(value):.4f}"
    if column in TIME_COLUMNS:
        return f"{float(value):.1f}"
    if column == "best_epoch":
        return f"{float(value):g}"
    return latex_escape(str(value))


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * max(1, len(columns))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        values = [format_value(column, row[column]) for column in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

