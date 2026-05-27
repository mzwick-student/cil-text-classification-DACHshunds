from __future__ import annotations

import argparse
import csv
import inspect
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from baselines.classic_ml_baselines import clean_text, load_data, score_predictions


N_LABELS = 5


@dataclass(frozen=True)
class TransformerExperimentResult:
    experiment: str
    family: str
    model: str
    variant: str
    status: str
    best_epoch: float | None = None
    best_epoch_train_loss: float | None = None
    best_epoch_val_loss: float | None = None
    train_seconds: float | None = None
    eval_seconds: float | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    mae: float | None = None
    cil_score: float | None = None
    quadratic_weighted_kappa: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class EpochResult:
    experiment: str
    epoch: float
    train_loss: float | None
    val_loss: float
    accuracy: float
    macro_f1: float
    mae: float
    cil_score: float
    quadratic_weighted_kappa: float


class ReviewDataset(Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels.astype(int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def make_experiment_name(model_name: str, variant: str) -> str:
    safe_model_name = model_name.replace("/", "__")
    return f"transformer__{safe_model_name}__{variant}"


def tokenize_texts(tokenizer, texts: pd.Series, max_length: int) -> dict[str, list[int]]:
    return tokenizer(
        texts.tolist(),
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return score_predictions(np.asarray(labels, dtype=int), predictions)


def build_training_arguments(args: argparse.Namespace, run_output_dir: Path) -> TrainingArguments:
    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    strategy_name = (
        "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    )
    kwargs = {
        "output_dir": str(run_output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "cil_score",
        "greater_is_better": True,
        "seed": args.random_state,
        "report_to": "none",
        "save_total_limit": args.save_total_limit,
        "fp16": args.fp16,
        "lr_scheduler_type": args.lr_scheduler_type,
    }
    kwargs[strategy_name] = "epoch"
    if "dataloader_num_workers" in signature.parameters:
        kwargs["dataloader_num_workers"] = args.dataloader_num_workers
    return TrainingArguments(**kwargs)


def extract_epoch_history(experiment: str, log_history: list[dict]) -> list[EpochResult]:
    train_loss_by_epoch = []
    rows = []
    for log in log_history:
        epoch = log.get("epoch")
        if epoch is None:
            continue
        rounded_epoch = round(float(epoch), 6)
        if "loss" in log:
            train_loss_by_epoch.append((rounded_epoch, float(log["loss"])))
        if "eval_loss" in log:
            latest_train_loss = None
            for train_epoch, train_loss in train_loss_by_epoch:
                if train_epoch <= rounded_epoch:
                    latest_train_loss = train_loss
            rows.append(
                EpochResult(
                    experiment=experiment,
                    epoch=rounded_epoch,
                    train_loss=latest_train_loss,
                    val_loss=float(log["eval_loss"]),
                    accuracy=float(log["eval_accuracy"]),
                    macro_f1=float(log["eval_macro_f1"]),
                    mae=float(log["eval_mae"]),
                    cil_score=float(log["eval_cil_score"]),
                    quadratic_weighted_kappa=float(
                        log["eval_quadratic_weighted_kappa"]
                    ),
                )
            )
    return rows


def evaluate_epoch_zero(
    trainer: Trainer,
    experiment: str,
) -> tuple[dict[str, float], list[EpochResult], float]:
    start = time.perf_counter()
    metrics = trainer.evaluate()
    eval_seconds = time.perf_counter() - start
    row = EpochResult(
        experiment=experiment,
        epoch=0.0,
        train_loss=None,
        val_loss=float(metrics["eval_loss"]),
        accuracy=float(metrics["eval_accuracy"]),
        macro_f1=float(metrics["eval_macro_f1"]),
        mae=float(metrics["eval_mae"]),
        cil_score=float(metrics["eval_cil_score"]),
        quadratic_weighted_kappa=float(metrics["eval_quadratic_weighted_kappa"]),
    )
    return metrics, [row], eval_seconds


def run_experiment(args: argparse.Namespace) -> tuple[TransformerExperimentResult, list[EpochResult]]:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'transformers' package is required. Install Hugging Face "
            "Transformers in the active Python environment before running "
            "train_review_model.py."
        ) from exc

    variant = "eval_only" if args.eval_only else "fine_tuned"
    experiment = args.experiment_name or make_experiment_name(args.model_name, variant)
    run_output_dir = args.output_dir / "checkpoints" / experiment

    df = load_data(args.train_path)
    df["text_clean"] = df["sentence"].map(clean_text)
    if args.sample_size is not None and args.sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=args.sample_size,
            stratify=df["label"],
            random_state=args.random_state,
        )
        df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_size,
        stratify=df["label"],
        random_state=args.random_state,
    )
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    print(f"Model: {args.model_name}", flush=True)
    print(f"Variant: {variant}", flush=True)
    print(f"Train examples: {len(train_df)}", flush=True)
    print(f"Validation examples: {len(val_df)}", flush=True)
    class_counts = {int(label): int(count) for label, count in sorted(Counter(y_train).items())}
    print(f"Class counts: {class_counts}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=N_LABELS,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    train_dataset = ReviewDataset(
        tokenize_texts(tokenizer, train_df["text_clean"], args.max_length),
        y_train,
    )
    val_dataset = ReviewDataset(
        tokenize_texts(tokenizer, val_df["text_clean"], args.max_length),
        y_val,
    )

    training_args = build_training_arguments(args, run_output_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    train_seconds = 0.0
    if args.eval_only:
        eval_metrics, epoch_history, eval_seconds = evaluate_epoch_zero(trainer, experiment)
        best_epoch = epoch_history[0]
    else:
        start = time.perf_counter()
        checkpoint = args.resume_from_checkpoint
        if checkpoint is None:
            checkpoints = sorted(
                run_output_dir.glob("checkpoint-*"),
                key=lambda path: int(path.name.split("-")[-1]),
            )
            if checkpoints:
                checkpoint = str(checkpoints[-1])
                print(f"Resuming from checkpoint: {checkpoint}", flush=True)

        trainer.train(resume_from_checkpoint=checkpoint)
        train_seconds = time.perf_counter() - start
        epoch_history = extract_epoch_history(experiment, trainer.state.log_history)
        if not epoch_history:
            eval_metrics, epoch_history, eval_seconds = evaluate_epoch_zero(trainer, experiment)
        else:
            start = time.perf_counter()
            eval_metrics = trainer.evaluate()
            eval_seconds = time.perf_counter() - start
        trainer.save_model(run_output_dir / "best_model")
        tokenizer.save_pretrained(run_output_dir / "best_model")
        best_epoch = max(epoch_history, key=lambda row: row.cil_score)

    result = TransformerExperimentResult(
        experiment=experiment,
        family="transformer",
        model=args.model_name,
        variant=variant,
        status="ok",
        best_epoch=best_epoch.epoch,
        best_epoch_train_loss=best_epoch.train_loss,
        best_epoch_val_loss=best_epoch.val_loss,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        accuracy=best_epoch.accuracy,
        macro_f1=best_epoch.macro_f1,
        mae=best_epoch.mae,
        cil_score=best_epoch.cil_score,
        quadratic_weighted_kappa=best_epoch.quadratic_weighted_kappa,
        notes=f"Final eval score={eval_metrics['eval_cil_score']:.5f}",
    )
    return result, epoch_history


def write_results(
    result: TransformerExperimentResult,
    epoch_history: list[EpochResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "transformer_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

    best_path = output_dir / "transformer_best_by_family.csv"
    with best_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

    history_path = output_dir / "transformer_epoch_history.csv"
    fieldnames = [
        "experiment",
        "epoch",
        "train_loss",
        "val_loss",
        "accuracy",
        "macro_f1",
        "mae",
        "cil_score",
        "quadratic_weighted_kappa",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in epoch_history)

    print(f"Wrote {results_path}", flush=True)
    print(f"Wrote {best_path}", flush=True)
    print(f"Wrote {history_path}", flush=True)


def print_result(result: TransformerExperimentResult) -> None:
    print(
        f"[ok] {result.experiment}: "
        f"best_epoch={result.best_epoch} "
        f"score={result.cil_score:.5f} "
        f"mae={result.mae:.5f} "
        f"acc={result.accuracy:.5f} "
        f"macro_f1={result.macro_f1:.5f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune or evaluate a Hugging Face transformer for review sentiment."
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/transformers"))
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--logging-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ignore-mismatched-sizes", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import set_seed

        set_seed(args.random_state)
    except ModuleNotFoundError:
        pass
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    result, epoch_history = run_experiment(args)
    write_results(result, epoch_history, args.output_dir)
    print_result(result)


if __name__ == "__main__":
    main()
