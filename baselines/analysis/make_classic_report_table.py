from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines.analysis.table_utils import latex_escape


FAMILY_LABELS = {
    "majority": "Majority",
    "bow": "BoW",
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


def variant_label(variant: str) -> str:
    if pd.isna(variant):
        return ""
    variant = str(variant)
    if variant.startswith("word_"):
        return variant.removeprefix("word_").replace("_", "--")
    if variant.startswith("char_wb_"):
        return "char " + variant.removeprefix("char_wb_").replace("_", "--")
    return variant.replace("_", " ")


def family_label_with_variant(best: pd.DataFrame, family: str, metric: str) -> str:
    base = FAMILY_LABELS.get(family, family)
    if family not in {"bow", "tfidf_word", "tfidf_char"}:
        return base
    rows = best[best["family"] == family]
    if rows.empty or "variant" not in rows.columns:
        return base
    ascending = metric == "mae"
    variant = rows.sort_values(metric, ascending=ascending).iloc[0]["variant"]
    return f"{base} ({variant_label(variant)})"


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
    family_labels = {
        family: family_label_with_variant(best_per_seed_family_classifier.last_best, family, metric)
        for family in families
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
    best_by_family = {}
    for family in families:
        values = [
            lookup[(family, classifier)][0]
            for classifier in classifiers
            if (family, classifier) in lookup
        ]
        if values:
            best_by_family[family] = min(values) if metric == "mae" else max(values)

    majority = None
    majority_rows = best_per_seed_family_classifier.last_best
    majority_rows = majority_rows[majority_rows["family"] == "majority"]
    if not majority_rows.empty:
        majority = majority_rows.groupby("seed")[metric].first().mean()

    align = "l" + "r" * len(classifiers)
    metric_label = "Validation MAE" if metric == "mae" else metric.replace("_", "-")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        "Method family & " + " & ".join(latex_escape(CLASSIFIER_LABELS[c]) for c in classifiers) + r" \\",
        r"\midrule",
    ]
    if majority is not None:
        value = f"{majority:.3f}" if metric == "mae" else f"{100.0 * majority:.2f}"
        lines.append(
            latex_escape(FAMILY_LABELS["majority"])
            + rf" & \multicolumn{{{len(classifiers)}}}{{c}}{{{value}}} \\"
        )
    for family in families:
        cells = []
        for classifier in classifiers:
            mean, std = lookup.get((family, classifier), (float("nan"), float("nan")))
            is_best = pd.notna(mean) and (
                mean == best_by_classifier.get(classifier)
                or mean == best_by_family.get(family)
            )
            cells.append(format_cell(mean, std, is_best, metric, include_std))
        lines.append(latex_escape(family_labels.get(family, family)) + " & " + " & ".join(cells) + r" \\")
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
    best_per_seed_family_classifier.last_best = best
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
