# cil-text-classification-DACHshunds

Classic and neural baselines for the ETH CIL 2026 sentiment classification
project.

## Main Workflow for Experiment 1

1. Download static embeddings with `load_embeddings.ipynb`.
2. Run and analyze classic ML baselines with `baselines.ipynb`.
3. Run and analyze fastText DNN baselines with `dnn_baselines.ipynb`.
4. Inspect experiment outputs under `experiments/`.

Older exploratory notebooks are kept in `exploratory_notebooks/`.

## Main Workflow for Experiment 4

1. Use the file `sota_runner.ipynb`. 
2. Changing the Config cell accordingly, train the regressor and the classifier (up to Section 9).
3. Optionally, validate the regressor and classifier in Section 9.
4. Build the submission in Section 10, caching the test logits for both regression and classification.  

## AI Usage Declaration

**Tool used:** Claude Opus 4.7
**File(s) affected:** sota_runner.ipynb
**Purpose:** Debugging numerical instability issues in training (suggesting hyperparameters, model architectures, and objective functions). Also to help implement validation and final test submission templates and config formatting. 

**Tool used:** Claude Sonnet 4.6
**File(s) affected:** sota_runner.ipynb
**Purpose:** Smaller syntax issues and implementing small lines of code in train/validation where tensor/array shape mismatches are unclear. 
# CIL Text Classification DACHshunds

This branch contains the cleaned code needed to reproduce the transformer-based experiments from the paper:

**Capacity, Interference, and Ordinality in Multilingual Sentiment Classification**

The repository is intentionally smaller than the full exploration history. Classical ML, static embedding, DNN baseline notebooks, and exploratory scratch work were removed from this submission branch. The remaining code covers the paper experiments that use the reusable XLM-R/LoRA training harness: language-interference training, ordinal objectives, noise weighting, and the large XLM-R classifier run.

## Repository Structure

- `configs/`: JSON experiment definitions used by `scripts/run_experiment.py`.
- `experiments/`: reusable training code, model definitions, objective/decoder logic, and custom trainers.
- `scripts/run_experiment.py`: main entry point for training/evaluation.
- `scripts/predict_test.py`: run a saved model checkpoint on `data/test.csv` and write a submission CSV.
- `slurm/`: cluster launch scripts for the reproducible experiment groups.
- `data/`: expected competition CSVs.
- `visualizations/`: notebooks used to inspect W&B curves and English/German gradient cosine similarity.

## Setup

Install the experiment dependencies:

```bash
pip install -r requirements-experiments.txt
```

The scripts expect the following files:

- `data/train_lang.csv`: training data with columns `id`, `sentence`, `label`, `lang`
- `data/test.csv`: test data with columns `id`, `sentence`
- `data/example_submission.csv`: competition submission format reference

Training logs to W&B through Hugging Face `TrainingArguments`. To run without online logging, use:

```bash
WANDB_MODE=offline python scripts/run_experiment.py --config configs/ordinal_ce_xlmr.json --set model.hub_model_id=null
```

Use `--set model.hub_model_id=null` when you do not want the trained model pushed to Hugging Face.

## General Usage

Run a config:

```bash
python scripts/run_experiment.py --config configs/ordinal_ce_xlmr.json
```

Override config fields from the command line:

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

## Reproducing Paper Experiments

### 1. Language Interference and Gradient Surgery

Paper section: Experiment 2, language interference.

This compares standard joint multilingual fine-tuning against PCGrad and GradVac. All three use XLM-R-base with LoRA and Huber regression.

```bash
python scripts/run_experiment.py --config configs/lora_bert_default.json --set model.hub_model_id=null
python scripts/run_experiment.py --config configs/lora_bert_pcgrad.json --set model.hub_model_id=null
python scripts/run_experiment.py --config configs/lora_bert_gradvac.json --set model.hub_model_id=null
```

On SLURM:

```bash
sbatch slurm/run_grad_surgery_experiments.sbatch
```

The gradient cosine similarity diagnostic notebook is:

```text
visualizations/eng_ger_batch_cosine_similarity.ipynb
```

### 2. Ordinal and Noise-Aware Objectives

Paper section: Experiment 3, ordinal and noise-aware objectives.

Cross-entropy classifier with Bayes-MAE decoding:

```bash
python scripts/run_experiment.py --config configs/ordinal_ce_xlmr.json --set model.hub_model_id=null
```

Huber regressor with validation-tuned thresholds:

```bash
python scripts/run_experiment.py --config configs/ordinal_regression_xlmr.json --set model.hub_model_id=null
```

Ordinal soft-label cross entropy with Bayes-MAE decoding:

```bash
python scripts/run_experiment.py --config configs/ordinal_soft_label_ce_xlmr.json --set model.hub_model_id=null
```

Noise-weighted cross entropy with Bayes-MAE decoding:

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

Run the noise-weighting experiment on SLURM:

```bash
sbatch slurm/run_noise_weight_q5.sbatch
```

### 3. Large XLM-R Classifier

Paper section: final scaling component.

This branch keeps the large XLM-R noise-weighted classifier config:

```bash
python scripts/run_experiment.py --config configs/noise_weight_q5_xlmr_large.json --set model.hub_model_id=null
```

On SLURM:

```bash
sbatch slurm/run_noise_weight_q5_xlmr_large.sbatch
```

The exact final classifier-regressor ensemble described in the paper is not represented as a standalone script in this cleaned branch. The reusable code supports the classifier and regression objectives separately, and `scripts/predict_test.py` can generate test predictions for a saved model checkpoint.

## Prediction and Submission

After training, run a saved checkpoint on the test set:

```bash
python scripts/predict_test.py \
  --config configs/ordinal_ce_xlmr.json \
  --checkpoint outputs/<run-name>/final_model \
  --submission submissions/submission.csv \
  --details submissions/test_predictions.csv
```

For LoRA checkpoints, `--checkpoint` can also point to a Hugging Face model repository if the model was pushed there.

## Visualization

The following notebooks were kept for reproducing plots/diagnostics used during analysis:

- `visualizations/eng_ger_batch_cosine_similarity.ipynb`
- `visualizations/wandb_group_mae_plot.ipynb`
- `visualizations/wandb_objective_group_mae_plot.ipynb`

They assume access to the corresponding W&B project/history.

## Notes

- All configs use `data/train_lang.csv` and a stratified 90/10 train-validation split.
- The main reported metric is validation MAE; the competition score is `1 - MAE / 4`.
- The SLURM scripts source `${HOME}/.slurm_tokens` for private cluster credentials. Local runs do not need this file.
- Large model runs require a CUDA GPU with enough memory for XLM-R-large LoRA fine-tuning.
