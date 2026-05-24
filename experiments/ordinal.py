from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error


N_CLASSES = 5


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_mae_risk(probs: np.ndarray) -> np.ndarray:
    classes = np.arange(probs.shape[1])
    return np.stack(
        [np.sum(probs * np.abs(pred - classes), axis=1) for pred in classes],
        axis=1,
    )


def bayes_mae_decode(probs: np.ndarray) -> np.ndarray:
    return expected_mae_risk(probs).argmin(axis=1).astype(int)


def apply_thresholds(scores: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    return np.searchsorted(thresholds, scores, side="right").astype(int)


def tune_mae_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    unique_scores, group_starts = np.unique(sorted_scores, return_index=True)
    group_ends = np.r_[group_starts[1:], len(sorted_scores)]
    n_groups = len(unique_scores)

    group_cost = np.zeros((n_classes, n_groups), dtype=np.float64)
    for group, (start, end) in enumerate(zip(group_starts, group_ends)):
        y = sorted_labels[start:end]
        for cls in range(n_classes):
            group_cost[cls, group] = np.abs(cls - y).sum()

    prefix_cost = np.c_[np.zeros(n_classes), np.cumsum(group_cost, axis=1)]
    dp = np.full((n_classes, n_groups + 1), np.inf, dtype=np.float64)
    back = np.zeros((n_classes, n_groups + 1), dtype=int)
    dp[0] = prefix_cost[0]

    for cls in range(1, n_classes):
        best_value = np.inf
        best_split = 0
        for j in range(n_groups + 1):
            candidate = dp[cls - 1, j] - prefix_cost[cls, j]
            if candidate < best_value:
                best_value = candidate
                best_split = j
            dp[cls, j] = prefix_cost[cls, j] + best_value
            back[cls, j] = best_split

    cuts = []
    j = n_groups
    for cls in range(n_classes - 1, 0, -1):
        j = back[cls, j]
        cuts.append(j)
    cuts = cuts[::-1]

    thresholds = []
    eps = 1e-6
    for cut in cuts:
        if cut <= 0:
            thresholds.append(float(unique_scores[0] - eps))
        elif cut >= n_groups:
            thresholds.append(float(unique_scores[-1] + eps))
        else:
            thresholds.append(float((unique_scores[cut - 1] + unique_scores[cut]) / 2.0))

    thresholds_array = np.array(thresholds, dtype=np.float32)
    tuned_preds = apply_thresholds(scores, thresholds_array)
    return thresholds_array, tuned_preds, float(mean_absolute_error(labels, tuned_preds))


def ordinal_soft_targets(
    labels: torch.Tensor,
    tau_by_label: list[float],
    rho_by_label: list[float],
    prior: list[float],
    n_classes: int = N_CLASSES,
) -> torch.Tensor:
    labels = labels.long().view(-1)
    classes = torch.arange(n_classes, device=labels.device, dtype=torch.float32)
    tau = torch.tensor(tau_by_label, device=labels.device, dtype=torch.float32)[labels].clamp_min(1e-6)
    rho = torch.tensor(rho_by_label, device=labels.device, dtype=torch.float32)[labels].clamp(0.0, 1.0)
    prior_tensor = torch.as_tensor(prior, device=labels.device, dtype=torch.float32)
    prior_tensor = prior_tensor / prior_tensor.sum().clamp_min(1e-12)

    distances = torch.abs(classes[None, :] - labels[:, None].float())
    ordinal_kernel = torch.softmax(-distances / tau[:, None], dim=1)
    q = (1.0 - rho[:, None]) * ordinal_kernel + rho[:, None] * prior_tensor[None, :]
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


@dataclass
class PredictionBundle:
    labels: np.ndarray
    raw: np.ndarray
    primary: np.ndarray
    columns: dict[str, np.ndarray]
    metrics: dict[str, float]
    artifacts: dict[str, Any]


def decode_predictions(
    objective_name: str,
    logits: np.ndarray,
    labels: np.ndarray | None,
    decoder: str,
    tuned_thresholds: list[float] | np.ndarray | None = None,
) -> PredictionBundle:
    raw = np.asarray(logits)
    labels_array = np.asarray(labels).reshape(-1).astype(int) if labels is not None else np.array([], dtype=int)
    columns: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    artifacts: dict[str, Any] = {}

    if objective_name in {"classification", "ordinal_soft_ce"}:
        probs = softmax_np(raw)
        map_preds = probs.argmax(axis=1).astype(int)
        bayes_preds = bayes_mae_decode(probs)
        expected_score = probs @ np.arange(probs.shape[1])
        columns.update(
            {
                "map_pred": map_preds,
                "bayes_mae_pred": bayes_preds,
                "expected_score": expected_score,
            }
        )
        for cls in range(probs.shape[1]):
            columns[f"p_{cls}"] = probs[:, cls]
        primary = bayes_preds if decoder == "bayes_mae" else map_preds
        if labels is not None:
            metrics.update(
                {
                    "accuracy": float(accuracy_score(labels_array, map_preds)),
                    "map_mae": float(mean_absolute_error(labels_array, map_preds)),
                    "bayes_mae": float(mean_absolute_error(labels_array, bayes_preds)),
                    "expected_score_mae": float(mean_absolute_error(labels_array, expected_score)),
                }
            )
        return PredictionBundle(labels_array, raw, primary, columns, metrics, artifacts)

    if objective_name == "coral":
        probs_gt = 1.0 / (1.0 + np.exp(-raw))
        expected_score = probs_gt.sum(axis=1)
        coral_preds = (probs_gt >= 0.5).sum(axis=1).astype(int)
        rounded = np.rint(np.clip(expected_score, 0, N_CLASSES - 1)).astype(int)
        columns.update({"coral_pred": coral_preds, "expected_score": expected_score, "rounded_pred": rounded})
        for cls in range(N_CLASSES - 1):
            columns[f"p_gt_{cls}"] = probs_gt[:, cls]
        primary = rounded if decoder == "expected_round" else coral_preds
        if labels is not None:
            metrics.update(
                {
                    "mae": float(mean_absolute_error(labels_array, expected_score)),
                    "coral_mae": float(mean_absolute_error(labels_array, coral_preds)),
                    "rounded_mae": float(mean_absolute_error(labels_array, rounded)),
                    "accuracy": float(accuracy_score(labels_array, coral_preds)),
                }
            )
        return PredictionBundle(labels_array, raw, primary, columns, metrics, artifacts)

    scores = raw.reshape(-1)
    clipped = np.clip(scores, 0, N_CLASSES - 1)
    rounded = np.rint(clipped).astype(int)
    columns.update({"score": scores, "rounded_pred": rounded})
    primary = rounded
    if tuned_thresholds is not None:
        tuned = apply_thresholds(clipped, tuned_thresholds)
        columns["tuned_threshold_pred"] = tuned
        primary = tuned if decoder == "tuned_thresholds" else primary
        artifacts["thresholds"] = np.asarray(tuned_thresholds, dtype=float).tolist()
    elif decoder == "tuned_thresholds" and labels is not None:
        thresholds, tuned, tuned_mae = tune_mae_thresholds(clipped, labels_array)
        columns["tuned_threshold_pred"] = tuned
        primary = tuned
        metrics["tuned_threshold_mae"] = tuned_mae
        artifacts["thresholds"] = thresholds.astype(float).tolist()
    if labels is not None:
        metrics.update(
            {
                "mae": float(mean_absolute_error(labels_array, scores)),
                "rounded_mae": float(mean_absolute_error(labels_array, rounded)),
            }
        )
        if "tuned_threshold_pred" in columns:
            metrics["tuned_threshold_mae"] = float(
                mean_absolute_error(labels_array, columns["tuned_threshold_pred"])
            )
    return PredictionBundle(labels_array, raw, primary, columns, metrics, artifacts)


def build_compute_metrics(objective_name: str, decoder: str):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        return decode_predictions(objective_name, logits, labels, decoder).metrics

    return compute_metrics
