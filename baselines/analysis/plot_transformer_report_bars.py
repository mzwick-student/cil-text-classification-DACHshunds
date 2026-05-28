from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from baselines.analysis.table_utils import latex_escape


MODEL_LABELS = {
    "distilbert-base-uncased": "DistilBERT",
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "xlm-roberta-large": "XLM-R-large",
    "nlptown/bert-base-multilingual-uncased-sentiment": "NLP-Town BERT",
}
COLORS = ["#21c7d9", "#2f7ebc", "#f28e2b", "#7f58af", "#59a14f"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create report bar plots for transformer baseline comparisons."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="transformer_model_comparison")
    parser.add_argument("--metric", choices=["mae", "cil_score", "accuracy", "macro_f1"], default="mae")
    parser.add_argument("--caption", default="Transformer baseline comparison across seeds.")
    parser.add_argument("--label", default="fig:transformer-baselines")
    return parser.parse_args()


def display_name(model: str, variant: str) -> str:
    name = MODEL_LABELS.get(model, model)
    if variant == "eval_only":
        return f"{name}\n(eval-only)"
    return f"{name}\n(fine-tuned)"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df["display_name"] = [
        display_name(model, variant) for model, variant in zip(df["model"], df["variant"])
    ]
    summary = (
        df.groupby(["display_name"], sort=False)[args.metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    ascending = args.metric != "mae"
    summary = summary.sort_values("mean", ascending=ascending).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(summary))
    ax.bar(
        x,
        summary["mean"],
        yerr=summary["std"].fillna(0.0),
        capsize=4,
        color=[COLORS[idx % len(COLORS)] for idx in x],
        edgecolor="#222222",
        linewidth=0.8,
    )
    ylabel = {
        "mae": "Validation MAE",
        "cil_score": "CIL-score",
        "accuracy": "Validation accuracy",
        "macro_f1": "Macro-F1",
    }[args.metric]
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["display_name"], rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if args.metric == "mae":
        ax.set_ylim(max(0.0, summary["mean"].min() - 0.04), summary["mean"].max() + 0.04)
    else:
        ax.set_ylim(max(0.0, summary["mean"].min() - 0.03), min(1.0, summary["mean"].max() + 0.03))
    for idx, row in summary.iterrows():
        ax.text(idx, row["mean"], f"{row['mean']:.3f}", ha="center", va="bottom", fontsize=13)
    fig.tight_layout()

    pdf_path = args.output_dir / f"{args.prefix}.pdf"
    png_path = args.output_dir / f"{args.prefix}.png"
    tex_path = args.output_dir / f"{args.prefix}.tex"
    csv_path = args.output_dir / f"{args.prefix}.csv"
    summary.to_csv(csv_path, index=False)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    tex_path.write_text(
        "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.72\linewidth]{{{latex_escape(pdf_path.name)}}}",
                rf"\caption{{{latex_escape(args.caption)}}}",
                rf"\label{{{latex_escape(args.label)}}}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for path in [csv_path, pdf_path, png_path, tex_path]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
