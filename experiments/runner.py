from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from datasets import Dataset, Value
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments

from .config import ExperimentConfig, config_to_dict
from .models import SentimentModel
from .trainers import SentimentTrainer


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.run_name = self._run_name()
        self.output_dir = Path(config.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        self._set_seed()
        self._setup_wandb()
        self._write_json("config.json", config_to_dict(self.config) | {"seed": self.seed})

        started = time.time()
        train_df, val_df = self._load_split()
        tokenizer, train_ds, val_ds = self._tokenize(train_df, val_df)
        model = SentimentModel.from_config(self.config.model)

        trainer = SentimentTrainer(
            method=self.config.trainer,
            model=model,
            args=self._training_args(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=SentimentTrainer.metrics,
            tokenizer=tokenizer,
        )

        train_result = trainer.train()
        train_metrics = trainer.evaluate(train_ds, metric_key_prefix="train")
        val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")

        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        if self.config.model.hub_model_id:
            trainer.push_to_hub()

        metrics = {
            "seed": self.seed,
            "runtime_seconds": round(time.time() - started, 3),
            "train_loss": float(train_metrics["train_loss"]),
            "training_loss": float(train_result.training_loss),
            "train_mae": float(train_metrics["train_mae"]),
            "train_rounded_mae": float(train_metrics["train_rounded_mae"]),
            "val_loss": float(val_metrics["val_loss"]),
            "val_mae": float(val_metrics["val_mae"]),
            "val_rounded_mae": float(val_metrics["val_rounded_mae"]),
        }
        trainer.log({f"final/{key}": value for key, value in metrics.items()})
        self._write_json("metrics.json", metrics)
        return metrics

    def _load_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = self.config.data
        df = pd.read_csv(data.train)
        if data.lang != "all" and data.lang_column in df.columns:
            df = df[df[data.lang_column] == data.lang].copy()

        stratify = df[data.label_column] if df[data.label_column].nunique() > 1 else None
        train_df, val_df = train_test_split(
            df,
            test_size=data.val_size,
            random_state=self.seed,
            stratify=stratify,
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def _tokenize(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        data = self.config.data
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

        def preprocess(batch):
            tokenized = tokenizer(
                batch[data.text_column],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            tokenized["labels"] = [float(label) for label in batch[data.label_column]]
            if data.lang_column in batch:
                tokenized["lang"] = [
                    0 if lang == "eng_Latn" else 1 for lang in batch[data.lang_column]
                ]
            return tokenized

        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        val_ds = Dataset.from_pandas(val_df, preserve_index=False)
        remove_columns = list(train_df.columns)
        train_ds = train_ds.map(preprocess, batched=True, remove_columns=remove_columns)
        val_ds = val_ds.map(preprocess, batched=True, remove_columns=remove_columns)
        train_ds = train_ds.cast_column("labels", Value("float32"))
        val_ds = val_ds.cast_column("labels", Value("float32"))
        if "lang" in train_ds.column_names:
            train_ds = train_ds.cast_column("lang", Value("int64"))
            val_ds = val_ds.cast_column("lang", Value("int64"))
        train_ds.set_format("torch")
        val_ds.set_format("torch")
        return tokenizer, train_ds, val_ds

    def _training_args(self) -> TrainingArguments:
        training = self.config.training
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            per_device_train_batch_size=training.batch_size,
            per_device_eval_batch_size=training.eval_batch_size,
            learning_rate=training.learning_rate,
            lr_scheduler_type=training.scheduler,
            warmup_steps=training.warmup_steps,
            num_train_epochs=training.epochs,
            eval_strategy="steps",
            eval_steps=training.eval_steps,
            logging_steps=training.logging_steps,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=training.fp16,
            report_to=["wandb"],
            run_name=self.run_name,
            remove_unused_columns=False,
            seed=self.seed,
            push_to_hub=bool(self.config.model.hub_model_id),
            hub_model_id=self._hub_model_id(),
            hub_private_repo=self.config.model.hub_private,
        )

    def _hub_model_id(self) -> str | None:
        hub_model_id = self.config.model.hub_model_id
        if not hub_model_id:
            return None
        if "{seed}" in hub_model_id:
            return hub_model_id.format(seed=self.seed)
        if len(self.config.seeds) > 1:
            return f"{hub_model_id}-seed{self.seed}"
        return hub_model_id

    def _run_name(self) -> str:
        model = self.config.model
        return f"{self.config.name}_seed{self.seed}_{model.kind}_{model.geometry}_{self.config.trainer}"

    def _setup_wandb(self) -> None:
        os.environ.setdefault("WANDB_PROJECT", self.config.logging.wandb_project)
        if self.config.logging.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", self.config.logging.wandb_entity)

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

    def _write_json(self, name: str, payload: dict) -> None:
        with (self.output_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def run_experiment(config: ExperimentConfig, seed: int) -> dict:
    return ExperimentRunner(config, seed).run()
