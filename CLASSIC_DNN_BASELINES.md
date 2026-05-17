# Classic DNN Baseline Grid

This runner evaluates neural classifiers on static GloVe and fastText
representations. It is intentionally narrower than the classic ML grid because
each combination is trained across epochs.

| ID | Embedding | Input Type | Representation | Model | Report Rule |
|---|---|---|---|---|---|
| D1 | GloVe | document vector | average | MLP | XOR with D2 |
| D2 | GloVe | document vector | TF-IDF weighted average | MLP | best GloVe MLP |
| D3 | fastText | document vector | average | MLP | XOR with D4 |
| D4 | fastText | document vector | TF-IDF weighted average | MLP | best fastText MLP |
| D5 | GloVe | token sequence | pretrained token vectors | TextCNN | compare to D7 |
| D6 | GloVe | token sequence | pretrained token vectors | BiLSTM | compare to D8 |
| D7 | fastText | token sequence | pretrained token vectors | TextCNN | likely report |
| D8 | fastText | token sequence | pretrained token vectors | BiLSTM | likely report |
| D9 | fastText | token sequence | pretrained token vectors | Small Transformer | optional diagnostic report |
| D10 | GloVe | token sequence | pretrained token vectors | Small Transformer | optional, lowest priority |

By default the small Transformer experiments are skipped. Add
`--include-transformers` to run D9 and D10.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Download embeddings first with `load_embeddings.ipynb`, then run:

```bash
python -m baselines.classic_dnn_baselines
```

The runner writes:

- `experiments/classic_dnn/classic_dnn_results.csv`
- `experiments/classic_dnn/classic_dnn_epoch_history.csv`

