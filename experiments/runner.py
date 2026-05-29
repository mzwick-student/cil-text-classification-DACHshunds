from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from datasets import Dataset, Value
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, TrainingArguments

from .config import ExperimentConfig, config_to_dict
from .models import SentimentModel
from .ordinal import bayes_mae_decode, build_compute_metrics, decode_predictions, softmax_np
from .trainers import SentimentTrainer


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.run_name = self._run_name()
        self.output_dir = Path(config.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extra_metrics: dict[str, float] = {}

    def run(self) -> dict:
        self._set_seed()
        self._setup_wandb()
        wandb_exit_code = 0
        try:
            self._write_json("config.json", config_to_dict(self.config) | {"seed": self.seed})

            started = time.time()
            train_df, val_df = self._load_split()
            tokenizer, train_ds, val_ds = self._tokenize(train_df, val_df)
            class_prior = self._class_prior(train_df)
            model = SentimentModel.from_config(self.config.model, self.config.objective, class_prior)

            trainer = SentimentTrainer(
                method=self.config.trainer,
                model=model,
                args=self._training_args(),
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=build_compute_metrics(self.config.objective.name, self.config.objective.decoder),
                #tokenizer=tokenizer,
            )

            train_result = trainer.train()
            train_metrics = trainer.evaluate(train_ds, metric_key_prefix="train")
            val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")

            trainer.save_model(str(self.output_dir / "final_model"))
            tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            val_prediction_metrics = self._save_val_error_dataframe(trainer, val_ds, val_df)
            metrics = {
                "seed": self.seed,
                "runtime_seconds": round(time.time() - started, 3),
                "train_loss": float(train_metrics["train_loss"]),
                "training_loss": float(train_result.training_loss),
                "val_loss": float(val_metrics["val_loss"]),
            }
            metrics.update(self._prefixed_numeric_metrics("train", train_metrics))
            metrics.update(self._prefixed_numeric_metrics("val", val_metrics))
            metrics.update(val_prediction_metrics)
            metrics.update(self.extra_metrics)
            trainer.log({f"final/{key}": value for key, value in metrics.items()})
            self._write_json("metrics.json", metrics)

            if self.config.model.hub_model_id:
                trainer.push_to_hub()
                self._push_huggingface_report()

            return metrics
        except Exception:
            wandb_exit_code = 1
            raise
        finally:
            self._finish_wandb(wandb_exit_code)

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

    def _class_prior(self, train_df: pd.DataFrame) -> list[float]:
        objective = self.config.objective
        if objective.prior_mode == "uniform":
            return [1.0 / objective.n_classes] * objective.n_classes
        counts = (
            train_df[self.config.data.label_column]
            .value_counts()
            .reindex(range(objective.n_classes), fill_value=0)
            .sort_index()
            .to_numpy(dtype=np.float64)
        )
        counts = counts + 1e-12
        return (counts / counts.sum()).tolist()

    def _tokenize(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        if self.config.noise_weighting.enabled:
            train_df = self._add_noise_weights(train_df, tokenizer)

        train_ds = self._to_dataset(train_df, tokenizer)
        val_ds = self._to_dataset(val_df, tokenizer)
        return tokenizer, train_ds, val_ds

    def _to_dataset(self, frame: pd.DataFrame, tokenizer):
        data = self.config.data
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
            if "sample_weight" in batch:
                tokenized["sample_weight"] = [float(weight) for weight in batch["sample_weight"]]
            return tokenized

        ds = Dataset.from_pandas(frame, preserve_index=False)
        ds = ds.map(preprocess, batched=True, remove_columns=list(frame.columns))
        ds = ds.cast_column("labels", Value("float32"))
        if "lang" in ds.column_names:
            ds = ds.cast_column("lang", Value("int64"))
        if "sample_weight" in ds.column_names:
            ds = ds.cast_column("sample_weight", Value("float32"))
        ds.set_format("torch")
        return ds

    def _add_noise_weights(self, train_df: pd.DataFrame, tokenizer) -> pd.DataFrame:
        if self.config.objective.name != "classification":
            raise ValueError("noise_weighting currently expects objective.name='classification'")

        noise = self.config.noise_weighting
        data = self.config.data
        labels = train_df[data.label_column].to_numpy(dtype=int)
        oof_probs = np.full((len(train_df), self.config.objective.n_classes), np.nan, dtype=np.float32)
        oof_fold = np.full(len(train_df), -1, dtype=int)

        splitter = StratifiedKFold(n_splits=noise.oof_n_splits, shuffle=True, random_state=self.seed)
        for fold, (fit_idx, oof_idx) in enumerate(splitter.split(train_df[data.text_column], labels)):
            print(
                f"Noise OOF fold {fold}: fit={len(fit_idx)} oof={len(oof_idx)} epochs={noise.oof_epochs}",
                flush=True,
            )
            fold_train_ds = self._to_dataset(train_df.iloc[fit_idx].reset_index(drop=True), tokenizer)
            fold_oof_ds = self._to_dataset(train_df.iloc[oof_idx].reset_index(drop=True), tokenizer)
            class_prior = self._class_prior(train_df.iloc[fit_idx])
            model = SentimentModel.from_config(self.config.model, self.config.objective, class_prior)
            trainer = SentimentTrainer(
                method="default",
                model=model,
                args=self._pilot_training_args(fold),
                train_dataset=fold_train_ds,
                compute_metrics=build_compute_metrics(self.config.objective.name, self.config.objective.decoder),
            )
            trainer.train()
            oof_probs[oof_idx] = softmax_np(trainer.predict(fold_oof_ds).predictions).astype(np.float32)
            oof_fold[oof_idx] = fold

            del trainer, model, fold_train_ds, fold_oof_ds
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        available_oof = np.isfinite(oof_probs).all(axis=1)
        oof_map_preds = np.full(len(train_df), -1, dtype=int)
        oof_bayes_preds = np.full(len(train_df), -1, dtype=int)
        oof_abs_error = np.full(len(train_df), np.nan, dtype=np.float32)
        oof_ce_loss = np.full(len(train_df), np.nan, dtype=np.float32)
        covered_idx = np.flatnonzero(available_oof)

        oof_map_preds[available_oof] = oof_probs[available_oof].argmax(axis=1).astype(int)
        oof_bayes_preds[available_oof] = bayes_mae_decode(oof_probs[available_oof])
        oof_abs_error[available_oof] = np.abs(oof_bayes_preds[available_oof] - labels[available_oof])
        true_probs = np.clip(oof_probs[available_oof, labels[available_oof]], 1e-12, 1.0)
        oof_ce_loss[available_oof] = -np.log(true_probs)

        weights = np.ones(len(train_df), dtype=np.float32)
        noisy = np.zeros(len(train_df), dtype=bool)
        n_noisy = max(1, int(np.ceil(len(covered_idx) * noise.q_percent / 100.0)))
        order = np.lexsort((-oof_ce_loss[covered_idx], -oof_abs_error[covered_idx]))
        noisy_idx = covered_idx[order[:n_noisy]]
        noisy[noisy_idx] = True
        weights[noisy_idx] = noise.noisy_weight

        diagnostics = train_df.reset_index(drop=True).copy()
        diagnostics["oof_fold"] = oof_fold
        diagnostics["oof_map_pred"] = oof_map_preds
        diagnostics["oof_bayes_pred"] = oof_bayes_preds
        diagnostics["oof_abs_error"] = oof_abs_error
        diagnostics["oof_ce_loss"] = oof_ce_loss
        diagnostics["is_noisy"] = noisy
        diagnostics["sample_weight"] = weights
        for cls in range(self.config.objective.n_classes):
            diagnostics[f"oof_p_{cls}"] = oof_probs[:, cls]
        diagnostics.to_csv(self.output_dir / "oof_diagnostics.csv", index=False)

        summary = {
            "noise_oof_n_splits": noise.oof_n_splits,
            "noise_oof_epochs": noise.oof_epochs,
            "noise_q_percent": noise.q_percent,
            "noise_noisy_weight": noise.noisy_weight,
            "noise_oof_covered": int(available_oof.sum()),
            "noise_n_noisy": int(noisy.sum()),
            "noise_effective_train_weight": float(weights.sum()),
            "noise_mean_noisy_abs_error": float(np.nanmean(oof_abs_error[noisy])),
            "noise_mean_noisy_ce_loss": float(np.nanmean(oof_ce_loss[noisy])),
        }
        self.extra_metrics.update(summary)
        self._write_json("noise_weighting_summary.json", summary)

        weighted_train_df = train_df.reset_index(drop=True).copy()
        weighted_train_df["sample_weight"] = weights
        return weighted_train_df

    def _pilot_training_args(self, fold: int) -> TrainingArguments:
        training = self.config.training
        return TrainingArguments(
            output_dir=str(self.output_dir / "oof_checkpoints" / f"fold_{fold}"),
            per_device_train_batch_size=training.batch_size,
            per_device_eval_batch_size=training.eval_batch_size,
            learning_rate=training.learning_rate,
            lr_scheduler_type=training.scheduler,
            warmup_steps=training.warmup_steps,
            num_train_epochs=self.config.noise_weighting.oof_epochs,
            eval_strategy="no",
            logging_steps=training.logging_steps,
            save_strategy="no",
            fp16=training.fp16,
            report_to=[],
            remove_unused_columns=False,
            seed=self.seed + fold,
        )

    def _training_args(self) -> TrainingArguments:
        training = self.config.training
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            #overwrite_output_dir=True,
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
        objective = self.config.objective
        return (
            f"{self.config.name}_seed{self.seed}_{model.kind}_{model.geometry}_"
            f"{objective.name}_{objective.decoder}_{self.config.trainer}"
        )

    def _setup_wandb(self) -> None:
        os.environ.setdefault("WANDB_PROJECT", self.config.logging.wandb_project)
        if self.config.logging.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", self.config.logging.wandb_entity)

    def _finish_wandb(self, exit_code: int) -> None:
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        print(f"Finishing W&B run for {self.run_name} with exit_code={exit_code}", flush=True)
        wandb.finish(exit_code=exit_code)
        print(f"Finished W&B run for {self.run_name}", flush=True)

    def _set_seed(self) -> None:
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
            from transformers import set_seed as hf_set_seed

            hf_set_seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    def _write_json(self, name: str, payload: dict) -> None:
        with (self.output_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @staticmethod
    def _prefixed_numeric_metrics(prefix: str, metrics: dict) -> dict[str, float]:
        result = {}
        for key, value in metrics.items():
            if key == f"{prefix}_loss":
                continue
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    def _save_val_error_dataframe(self, trainer: SentimentTrainer, val_ds, val_df: pd.DataFrame) -> dict[str, float]:
        prediction = trainer.predict(val_ds)
        bundle = decode_predictions(
            self.config.objective.name,
            prediction.predictions,
            prediction.label_ids,
            self.config.objective.decoder,
        )
        out_df = val_df.reset_index(drop=True).copy()
        out_df["pred"] = bundle.primary.astype(int)
        out_df["error"] = np.abs(out_df[self.config.data.label_column].to_numpy(dtype=int) - out_df["pred"].to_numpy(dtype=int))
        for name, values in bundle.columns.items():
            out_df[name] = values
        out_df.to_csv(self.output_dir / "val_error_dataframe.csv", index=False)

        decoder_payload = {
            "objective": self.config.objective.name,
            "decoder": self.config.objective.decoder,
            "tau": self.config.objective.tau,
            "rho": self.config.objective.rho,
            "artifacts": bundle.artifacts,
        }
        self._write_json("decoder_config.json", decoder_payload)
        with (self.output_dir / "final_model" / "decoder_config.json").open("w", encoding="utf-8") as f:
            json.dump(decoder_payload, f, indent=2, sort_keys=True)

        metrics = {f"val_prediction_{key}": float(value) for key, value in bundle.metrics.items()}
        metrics["val_prediction_primary_mae"] = float(out_df["error"].mean())
        return metrics

    def _push_huggingface_report(self) -> None:
        from huggingface_hub import HfApi

        repo_id = self._hub_model_id()
        if repo_id is None:
            return

        api = HfApi()
        filenames = ["metrics.json", "decoder_config.json", "val_error_dataframe.csv"]
        if self.config.noise_weighting.enabled:
            filenames.extend(["noise_weighting_summary.json", "oof_diagnostics.csv"])
        for filename in filenames:
            api.upload_file(
                path_or_fileobj=str(self.output_dir / filename),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {filename} for {self.run_name}",
            )


def run_experiment(config: ExperimentConfig, seed: int) -> dict:
    return ExperimentRunner(config, seed).run()
