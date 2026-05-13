# Classic Baseline Grid

This project predicts review sentiment labels `0..4` for mixed English/German
product reviews. The official score is:

```text
score = 1 - MAE / 4
```

The runner in `baselines/classic_ml_baselines.py` evaluates a broad classic ML
grid and writes two CSV files:

- `classic_ml_results.csv`: every explored combination.
- `classic_ml_best_by_family.csv`: one best validation result per family, the
  XOR-style table for the report.

## Core combinations

| Family | Variants explored | Classifiers |
|---|---|---|
| majority | most frequent label | dummy constant predictor |
| BoW | word `1-1`, `1-2`, `1-3` | Logistic Regression, Linear SVM, Ridge Classifier, Complement NB |
| TF-IDF word | word `1-1`, `1-2`, `1-3` | Logistic Regression, Linear SVM, Ridge Classifier, Complement NB |
| TF-IDF char | `char_wb 3-5`, `char_wb 3-6` | Logistic Regression, Linear SVM, Ridge Classifier, Complement NB |
| GloVe | average, TF-IDF weighted, mean+max, SIF | Logistic Regression, Linear SVM, Ridge Classifier |
| fastText | average, TF-IDF weighted, mean+max, SIF | Logistic Regression, Linear SVM, Ridge Classifier |
| word2vec | average, TF-IDF weighted, mean+max, SIF | Logistic Regression, Linear SVM, Ridge Classifier |

The report should usually show the majority baseline plus the best member of
each feature family. The full grid remains available for appendix/ablation
discussion.

The notebook `baselines.ipynb` is the recommended entry point for running the
grid and creating the final analysis tables under `experiments/classic_ml/`.

## Running

Install the declared project dependencies first:

```bash
pip install -r requirements.txt
```

Full sparse run:

```bash
python -m baselines.classic_ml_baselines --mode sparse
```

Full run with static embeddings:

```bash
python -m baselines.classic_ml_baselines \
  --mode all \
  --glove-path embeddings/glove.6B.300d.txt \
  --fasttext-path embeddings/wiki.de.vec \
  --word2vec-path embeddings/word2vec.txt
```

Embedding files are optional. Families without a path are skipped. The loader
expects text-format vector files such as GloVe `.txt`, fastText `.vec`, or
word2vec text `.vec/.txt` with an optional `num_words dim` header.
