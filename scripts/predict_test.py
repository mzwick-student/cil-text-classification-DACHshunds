#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import load_config
from experiments.models import SentimentModel
from experiments.ordinal import decode_predictions


def load_model(config_path: Path, checkpoint: str, device: torch.device):
    exp_config = load_config(config_path)
    hf_config = AutoConfig.from_pretrained(exp_config.model.name)
    hf_config.num_labels = exp_config.objective.n_classes

    if exp_config.model.kind == "lora_bert":
        from peft import PeftModel

        base_model = SentimentModel.from_pretrained(
            exp_config.model.name,
            config=hf_config,
            geometry=exp_config.model.geometry,
            objective=exp_config.objective,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        model = SentimentModel.from_pretrained(
            checkpoint,
            config=hf_config,
            geometry=exp_config.model.geometry,
            objective=exp_config.objective,
        )

    model.to(device)
    model.eval()
    return exp_config, model


def load_tokenizer(checkpoint: str, fallback_model_name: str):
    try:
        return AutoTokenizer.from_pretrained(checkpoint)
    except OSError:
        return AutoTokenizer.from_pretrained(fallback_model_name)


@torch.inference_mode()
def predict_logits(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: torch.device):
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        chunks.append(outputs["logits"].detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a saved experiment model on data/test.csv.")
    parser.add_argument("--config", required=True, type=Path, help="Experiment config JSON used for training.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path or Hub repo for final_model, e.g. outputs/.../final_model.",
    )
    parser.add_argument("--test", default="data/test.csv", type=Path)
    parser.add_argument("--submission", default="submission.csv", type=Path)
    parser.add_argument("--details", default="test_predictions.csv", type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    exp_config, model = load_model(args.config, args.checkpoint, device)
    tokenizer = load_tokenizer(args.checkpoint, exp_config.model.name)

    test_df = pd.read_csv(args.test)
    texts = test_df[exp_config.data.text_column].fillna("").astype(str).tolist()
    logits = predict_logits(model, tokenizer, texts, args.batch_size, args.max_length, device)
    bundle = decode_predictions(
        exp_config.objective.name,
        logits,
        labels=None,
        decoder=exp_config.objective.decoder,
    )

    details = test_df.copy()
    details["pred"] = bundle.primary.astype(int)
    for name, values in bundle.columns.items():
        details[name] = values

    submission = pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "label": bundle.primary.astype(int),
        }
    )

    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details, index=False)
    submission.to_csv(args.submission, index=False)
    print(f"Wrote {args.submission}")
    print(f"Wrote {args.details}")


if __name__ == "__main__":
    main()
