from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    train: str = "data/train_lang.csv"
    test: str = "data/test.csv"
    text_column: str = "sentence"
    label_column: str = "label"
    lang_column: str = "lang"
    lang: str = "all"
    val_size: float = 0.1


@dataclass
class ModelConfig:
    kind: str = "lora_bert"
    name: str = "xlm-roberta-base"
    geometry: str = "default"
    lora_r: int = 128
    lora_alpha: int = 64
    lora_dropout: float = 0.01
    hub_model_id: str | None = None
    hub_private: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 64
    eval_batch_size: int = 512
    learning_rate: float = 1.5e-4
    scheduler: str = "cosine"
    epochs: int = 5
    warmup_steps: int = 100
    logging_steps: int = 500
    eval_steps: int = 500
    fp16: bool = True


@dataclass
class ObjectiveConfig:
    name: str = "regression"
    decoder: str = "round"
    n_classes: int = 5
    loss: str = "huber"
    coral_num_thresholds: int = 4
    tau: list[float] = field(default_factory=lambda: [0.60, 0.60, 0.60, 0.60, 0.60])
    rho: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.05, 0.05])
    prior_mode: str = "empirical"


@dataclass
class LoggingConfig:
    wandb_project: str = "cil-sentiment"
    wandb_entity: str | None = None


@dataclass
class ExperimentConfig:
    name: str = "sentiment"
    seeds: list[int] = field(default_factory=lambda: [42])
    trainer: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "outputs"


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        raw = _merge_dict(raw, overrides)
    return ExperimentConfig(
        name=raw.get("name", "sentiment"),
        seeds=raw.get("seeds", [raw.get("seed", 42)]),
        trainer=raw.get("trainer", "default"),
        model=ModelConfig(**raw.get("model", {})),
        objective=ObjectiveConfig(**raw.get("objective", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        output_dir=raw.get("output_dir", "outputs"),
    )


def configs_for_seeds(config: ExperimentConfig) -> list[tuple[int, ExperimentConfig]]:
    return [(seed, config) for seed in config.seeds]


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
