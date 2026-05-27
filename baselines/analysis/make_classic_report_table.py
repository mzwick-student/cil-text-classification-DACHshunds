from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines.analysis.table_utils import latex_escape


FAMILY_LABELS = {
    "majority": "Majority",
    "bow": "Count vectorization",
    "tfidf_word": "TF-IDF word",
    "tfidf_char": "TF-IDF character",
    "glove": "GloVe embedding",
    "fasttext": "fastText embedding",
}
CLASSIFIER_LABELS = {
    "dummy_most_frequent": "Majority",
    "logreg": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "ridge_classifier": "Ridge Classifier",
    "complement_nb": "Complement NB",
}
FAMILY_ORDER = ["bow", "tfidf_word", "tfidf_char", "glove", "fasttext"]
CLASSIFIER_ORDER = [
    "logreg",
    "linear_svm",
    "ridge_classifier",
    "complement_nb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the report table for classical ML baseline families."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain-output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--metric", choices=["mae", "cil_score", "accuracy", "macro_f1"], default="mae")
    parser.add_argument("--caption", default="Results of classical ML baselines.")
    parser.add_argument("--label", default="tab:classic-baselines")
    return parser.parse_args()


def best_per_seed_family_classifier(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "seed" not in df.columns:
        df.insert(0, "seed", 0)
    ascending = metric == "mae"
    return (
        df.sort_values(metric, ascending=ascending)
        .groupby(["seed", "family", "classifier"], as_index=False)
        .first()
    )


def format_cell(mean: float, std: float, best: bool, metric: str, include_std: bool) -> str:
    if pd.isna(mean):
        return "--"
    value = f"{mean:.3f}" if metric == "mae" else f"{100.0 * mean:.2f}"
    if include_std and pd.notna(std) and std > 0:
        value = f"{value} $\\pm$ {std:.3f}" if metric == "mae" else f"{value} $\\pm$ {100.0 * std:.2f}"
    if best:
        return rf"\textbf{{{value}}}"
    return value


def write_table(
    output_path: Path,
    summary: pd.DataFrame,
    metric: str,
    caption: str,
    label: str,
    include_std: bool,
) -> None:
    families = [family for family in FAMILY_ORDER if family in set(summary["family"])]
    classifiers = [
        classifier for classifier in CLASSIFIER_ORDER if classifier in set(summary["classifier"])
    ]
    lookup = {
        (row["family"], row["classifier"]): (row["mean"], row["std"])
        for _, row in summary.iterrows()
    }

    best_by_classifier = {}
    for classifier in classifiers:
        values = [
            lookup[(family, classifier)][0]
            for family in families
            if (family, classifier) in lookup
        ]
        if values:
            best_by_classifier[classifier] = min(values) if metric == "mae" else max(values)

    align = "l" + "r" * len(classifiers)
    metric_label = "Validation MAE" if metric == "mae" else metric.replace("_", "-")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        "Method family & " + " & ".join(latex_escape(CLASSIFIER_LABELS[c]) for c in classifiers) + r" \\",
        r"\midrule",
    ]
    for family in families:
        cells = []
        for classifier in classifiers:
            mean, std = lookup.get((family, classifier), (float("nan"), float("nan")))
            is_best = pd.notna(mean) and mean == best_by_classifier.get(classifier)
            cells.append(format_cell(mean, std, is_best, metric, include_std))
        lines.append(latex_escape(FAMILY_LABELS.get(family, family)) + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\par\smallskip\footnotesize Metric: {latex_escape(metric_label)}. Each cell reports the best variant within the family, averaged across seeds.",
            r"\end{table}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    best = best_per_seed_family_classifier(df, args.metric)
    summary = (
        best.groupby(["family", "classifier"])[args.metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
        print(f"Wrote {args.summary_csv}")

    write_table(args.output, summary, args.metric, args.caption, args.label, include_std=True)
    plain_output = args.plain_output or args.output.with_name(f"{args.output.stem}_no_variance.tex")
    write_table(
        plain_output,
        summary,
        args.metric,
        args.caption,
        f"{args.label}-no-variance",
        include_std=False,
    )


if __name__ == "__main__":
    main()
