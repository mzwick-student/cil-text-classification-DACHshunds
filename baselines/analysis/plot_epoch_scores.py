from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from baselines.analysis.table_utils import latex_escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot epoch against CIL-score for DNN or transformer baselines."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--title", type=str, default="Epoch analysis")
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def plot_epoch_scores(
    input_path: Path,
    output_dir: Path,
    prefix: str,
    title: str,
    caption: str,
    label: str,
) -> tuple[Path, Path, Path]:
    history = pd.read_csv(input_path)
    required = {"experiment", "epoch", "cil_score"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
    if history.empty:
        raise ValueError(f"{input_path} has no epoch rows to plot")

    output_dir.mkdir(parents=True, exist_ok=True)
    history = history.sort_values(["experiment", "epoch"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for experiment, rows in history.groupby("experiment", sort=False):
        ax.plot(rows["epoch"], rows["cil_score"], marker="o", linewidth=1.8, label=experiment)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CIL-score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    pdf_path = output_dir / f"{prefix}.pdf"
    png_path = output_dir / f"{prefix}.png"
    tex_path = output_dir / f"{prefix}.tex"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tex_path.write_text(
        "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{latex_escape(pdf_path.name)}}}",
                rf"\caption{{{latex_escape(caption)}}}",
                rf"\label{{{latex_escape(label)}}}",
                r"\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pdf_path, png_path, tex_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent
    prefix = args.prefix or args.input.stem
    caption = args.caption or args.title
    label = args.label or f"fig:{prefix.replace('_', '-')}"
    paths = plot_epoch_scores(args.input, output_dir, prefix, args.title, caption, label)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
