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
