# CIL Text Classification DACHshunds

Code for the ETH CIL 2026 multilingual sentiment classification project:

**Capacity, Interference, and Ordinality in Multilingual Sentiment Classification**

The paper studies English/German product-review rating prediction with ordered labels `0..4`. This repository contains the code paths used for the main paper experiments: classical and neural baselines, transformer baselines, language-interference ablations, ordinal/noise-aware objectives, and the final XLM-R-large submission pipeline.

## Repository Structure

- `data/`: competition data files.
- `baselines/`: reusable Python modules for classical ML, DNN, and transformer baseline runs.
- `baselines.ipynb`: classical sparse and static-embedding baseline workflow.
- `dnn_baselines.ipynb`: fastText MLP/TextCNN/BiLSTM baseline workflow.
- `transformer_baselines.ipynb`: full fine-tuned transformer baseline workflow.
- `load_embeddings.ipynb`: downloads/prepares GloVe and aligned fastText vectors.
- `experiments/`: config-driven XLM-R/LoRA experiment harness for gradient surgery, ordinal objectives, and noise weighting.
- `configs/`: JSON configs for the config-driven experiment harness.
- `scripts/run_experiment.py`: train/evaluate one config-driven experiment.
- `scripts/predict_test.py`: run a saved config-driven checkpoint on `data/test.csv`.
- `sota_runner.ipynb`: final XLM-R-large classifier/regressor and submission workflow.
- `slurm/`: cluster launch scripts for the config-driven experiment groups.
- `visualizations/`: W&B plotting notebooks and English/German gradient-cosine diagnostic.

Generated experiment outputs are intentionally not part of the main source tree. Notebooks/scripts write results under `experiments/...`, `outputs/...`, `results/...`, or submission CSV paths depending on the workflow.

## Setup

Install the shared Python dependencies:

```bash
pip install -r requirements-experiments.txt
```

The main code expects these files:

- `data/train.csv`: original labeled training data with `id`, `sentence`, `label`
- `data/train_lang.csv`: training data plus language labels with `id`, `sentence`, `label`, `lang`
- `data/test.csv`: unlabeled test data with `id`, `sentence`
- `data/example_submission.csv`: expected submission format

For the static-embedding baselines, run `load_embeddings.ipynb` before `baselines.ipynb` or `dnn_baselines.ipynb`. It creates the expected embedding files under `experiments/embeddings/`.

Most transformer and LoRA runs log to W&B and may push models to Hugging Face if a config has `model.hub_model_id`. For local/offline runs, disable both:

```bash
WANDB_MODE=offline python scripts/run_experiment.py \
  --config configs/ordinal_ce_xlmr.json \
  --set model.hub_model_id=null
```

## Metrics

The main validation metric is mean absolute error:

```text
MAE = mean(abs(y_true - y_pred))
```

The competition score is:

```text
score = 1 - MAE / 4
```

## Reproducing the Paper Experiments

### Experiment 1: Representation Capacity

This experiment compares sparse lexical models, static embeddings, neural fastText models, and transformer baselines.

Run static-embedding preparation:

```text
load_embeddings.ipynb
```

Run classical ML baselines:

```text
baselines.ipynb
```

This covers BoW, word TF-IDF, character TF-IDF, GloVe pooling, fastText pooling, and linear classifiers.

Run fastText DNN baselines:

```text
dnn_baselines.ipynb
```

This covers MLP, TextCNN, and BiLSTM baselines.

Run full fine-tuned transformer baselines:

```text
transformer_baselines.ipynb
```

This covers the transformer baseline comparison reported in the paper. The helper code lives in `baselines/train_review_model.py`.

### Experiment 2: Language Interference

This experiment compares standard joint multilingual LoRA fine-tuning against PCGrad and GradVac. All three configs use XLM-R-base with Huber regression.

```bash
python scripts/run_experiment.py --config configs/lora_bert_default.json --set model.hub_model_id=null
python scripts/run_experiment.py --config configs/lora_bert_pcgrad.json --set model.hub_model_id=null
python scripts/run_experiment.py --config configs/lora_bert_gradvac.json --set model.hub_model_id=null
```

On SLURM:

```bash
sbatch slurm/run_grad_surgery_experiments.sbatch
```

The English/German gradient-cosine diagnostic notebook is:

```text
visualizations/eng_ger_batch_cosine_similarity.ipynb
```

### Experiment 3: Ordinal and Noise-Aware Objectives

Run cross-entropy classification with Bayes-MAE decoding:

```bash
python scripts/run_experiment.py --config configs/ordinal_ce_xlmr.json --set model.hub_model_id=null
```

Run Huber regression with validation-tuned thresholds:

```bash
python scripts/run_experiment.py --config configs/ordinal_regression_xlmr.json --set model.hub_model_id=null
```

Run ordinal soft-label cross entropy:

```bash
python scripts/run_experiment.py --config configs/ordinal_soft_label_ce_xlmr.json --set model.hub_model_id=null
```

Run noise-weighted cross entropy:

```bash
python scripts/run_experiment.py --config configs/noise_weight_q5_xlmr.json --set model.hub_model_id=null
```

On SLURM, run the ordinal objective group:

```bash
sbatch slurm/run_ordinal_ce.sbatch
sbatch slurm/run_ordinal_regression.sbatch
sbatch slurm/run_ordinal_soft_label_ce.sbatch
```

or:

```bash
slurm/submit_all_ordinal_objective.sh
```

Run noise weighting on SLURM:

```bash
sbatch slurm/run_noise_weight_q5.sbatch
```

### Experiment 4: Final Scaling and Submission

Use:

```text
sota_runner.ipynb
```

Recommended workflow:

1. Set the config cell to train the regressor or classifier.
2. Run the notebook through the training sections.
3. Optionally validate the regressor/classifier ensemble.
4. Run the final submission section, which caches test logits and writes the submission CSV.

The config-driven branch also keeps a large noise-weighted XLM-R classifier config:

```bash
python scripts/run_experiment.py --config configs/noise_weight_q5_xlmr_large.json --set model.hub_model_id=null
```

On SLURM:

```bash
sbatch slurm/run_noise_weight_q5_xlmr_large.sbatch
```

## Config-Driven Harness

The reusable LoRA harness is useful for Experiments 2 and 3.

Run any config:

```bash
python scripts/run_experiment.py --config configs/ordinal_ce_xlmr.json
```

Override config fields:

```bash
python scripts/run_experiment.py \
  --config configs/ordinal_ce_xlmr.json \
  --set seeds='[42]' \
  --set training.epochs=1 \
  --set model.hub_model_id=null
```

Outputs are written to `outputs/<run-name>/`:

- `config.json`: resolved config plus seed
- `metrics.json`: train/validation metrics
- `final_model/`: saved model and tokenizer
- `decoder_config.json`: objective and decoder metadata
- `val_error_dataframe.csv`: validation rows with predictions and errors

Generate predictions from a saved config-driven checkpoint:

```bash
python scripts/predict_test.py \
  --config configs/ordinal_ce_xlmr.json \
  --checkpoint outputs/<run-name>/final_model \
  --submission submissions/submission.csv \
  --details submissions/test_predictions.csv
```

For LoRA checkpoints, `--checkpoint` can also point to a Hugging Face model repository if the model was pushed there.

## Visualization

Kept analysis notebooks:

- `visualizations/eng_ger_batch_cosine_similarity.ipynb`
- `visualizations/wandb_group_mae_plot.ipynb`
- `visualizations/wandb_objective_group_mae_plot.ipynb`

The W&B notebooks assume access to the corresponding W&B project history.

## AI Usage Declaration

**Tool used:** Claude Opus 4.7  
**File(s) affected:** `sota_runner.ipynb`  
**Purpose:** Debugging numerical instability issues in training, including hyperparameters, model architectures, and objective functions. Also used to help implement validation, final test submission templates, and config formatting.

**Tool used:** Claude Sonnet 4.6  
**File(s) affected:** `sota_runner.ipynb`  
**Purpose:** Fixing smaller syntax issues and implementing small train/validation lines where tensor or array shape mismatches were unclear.

## Notes

- Config-driven experiments use `data/train_lang.csv` and a stratified 90/10 train-validation split.
- Baseline notebooks use `data/train.csv`.
- SLURM scripts source `${HOME}/.slurm_tokens` for private cluster credentials. Local runs do not need this file.
- XLM-R-large runs require a CUDA GPU with enough memory for large-model LoRA fine-tuning.
