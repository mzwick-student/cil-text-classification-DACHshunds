# cil-text-classification-DACHshunds

## Reproducible experiments

The experiment harness keeps the config small. In most cases you only set:

- `seeds`
- `trainer`
- `model`
- `data`
- `training`

The runner does only the boring parts: read the train CSV, make a 90/10 split, tokenize, train, log train/val loss and MAE metrics to W&B, save locally, and push to Hugging Face when `model.hub_model_id` is set.

Install the experiment dependencies in your environment:

```bash
pip install -r requirements-experiments.txt
```

Run plain BERT:

```bash
python scripts/run_experiment.py --config configs/bert_base.json
```

Run LoRA-BERT:

```bash
python scripts/run_experiment.py --config configs/lora_bert_base.json
```

Before pushing/logging, make sure you are logged in:

```bash
huggingface-cli login
wandb login
```

Override any config field from the CLI:

```bash
python scripts/run_experiment.py \
  --config configs/lora_bert_base.json \
  --set seeds='[1,2,3,4,5]' \
  --set trainer='"pcgrad"' \
  --set model.geometry='"circular"' \
  --set training.scheduler='"constant"'
```

Submit the Slurm template:

```bash
sbatch --export=CONFIG=configs/lora_bert_base.json,SEED=1 slurm/run_experiment.sbatch
```

From Jupyter:

```python
from experiments import configs_for_seeds, load_config, run_experiment

cfg = load_config("configs/lora_bert_base.json")
for seed, run_cfg in configs_for_seeds(cfg):
    metrics = run_experiment(run_cfg, seed)
    print(seed, metrics)
```

Outputs are written to `outputs/<run-name>/`:

- `config.json`: exact run config
- `metrics.json`: final train/val loss, MAE, and rounded MAE
- `final_model/`: saved final model
- `decoder_config.json`: objective/decoder metadata, including tuned thresholds when used
- `val_error_dataframe.csv`: validation rows with `id`, `sentence`, `lang`, `label`, `pred`, `error`, and available raw/probability columns

Minimal config example:

```json
{
  "name": "lora_bert_base",
  "seeds": [1, 2, 3],
  "trainer": "gradvac",
  "model": {
    "kind": "lora_bert",
    "name": "xlm-roberta-base",
    "geometry": "mobius",
    "lora_r": 128,
    "lora_alpha": 64,
    "lora_dropout": 0.01,
    "hub_model_id": "YOUR_HF_USERNAME/cil-lora-bert-base"
  },
  "training": {
    "batch_size": 64,
    "eval_batch_size": 512,
    "learning_rate": 1.5e-4,
    "scheduler": "cosine",
    "epochs": 5,
    "logging_steps": 500,
    "eval_steps": 500
  },
  "data": {
    "train": "data/train_lang.csv"
  },
  "logging": {
    "wandb_project": "cil-sentiment"
  }
}
```

Allowed values are still available when needed: `trainer` can be `default`,
`pcgrad`, or `gradvac`; `model.kind` can be `bert` or `lora_bert`;
`model.geometry` can be `default`, `circular`, or `mobius`.

## Ordinal objective experiments

The Slurm-ready ordinal experiments all use `xlm-roberta-base` LoRA and run seeds `[1, 2, 3]`:

- `configs/ordinal_ce_xlmr.json`: CE classifier, reporting both posterior argmax/MAP and Bayes-MAE decoding
- `configs/ordinal_regression_xlmr.json`: regression, reporting both rounding and validation-tuned MAE threshold decoding
- `configs/ordinal_coral_xlmr.json`: CORAL-style ordinal regression
- `configs/ordinal_soft_label_ce_xlmr.json`: ordinal soft-label CE with `tau=[0.60]*5`, `rho=[0.05]*5`, empirical prior, decoded with Bayes-MAE

Submit one experiment:

```bash
sbatch slurm/run_ordinal_ce.sbatch
```

Submit all ordinal objective experiments:

```bash
slurm/submit_all_ordinal_objective.sh
```
