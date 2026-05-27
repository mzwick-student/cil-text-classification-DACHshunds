#!/usr/bin/env bash
set -euo pipefail

PY=/cluster/courses/cil/envs/envs/text-5060/bin/python
RUN=experiments/transformers/20260525_101750_TRANSFORMER_BASELINES

SEEDS="42 43 44"
EPOCHS=3
LR=2e-5

run_one () {
  MODEL="$1"
  VARIANT="$2"
  RUN_NAME="${MODEL//\//__}__${VARIANT}"

  for SEED in $SEEDS; do
    OUT="$RUN/$RUN_NAME/seed_$SEED"

    if [ -f "$OUT/transformer_results.csv" ]; then
      echo "SKIP done: $RUN_NAME seed $SEED"
      continue
    fi

    echo "RUNNING: $RUN_NAME seed $SEED"

    CMD=(
      "$PY" -m baselines.train_review_model
      --model-name "$MODEL"
      --experiment-name "transformer__${RUN_NAME}"
      --train-path data/train.csv
      --output-dir "$OUT"
      --validation-size 0.1
      --random-state "$SEED"
      --max-length 256
      --epochs "$EPOCHS"
      --batch-size 16
      --eval-batch-size 32
      --learning-rate "$LR"
      --weight-decay 0.01
      --warmup-ratio 0.06
      --lr-scheduler-type linear
      --logging-steps 500
      --fp16
    )

    if [ "$VARIANT" = "eval_only" ]; then
      CMD+=(--eval-only)
    fi

    "${CMD[@]}"
  done
}

run_one distilbert-base-uncased fine_tuned
run_one bert-base-uncased fine_tuned
run_one roberta-base fine_tuned
run_one nlptown/bert-base-multilingual-uncased-sentiment fine_tuned
run_one nlptown/bert-base-multilingual-uncased-sentiment eval_only
