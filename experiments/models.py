from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, XLMRobertaModel, XLMRobertaPreTrainedModel

from .config import ModelConfig, ObjectiveConfig
from .ordinal import ordinal_soft_targets


def make_mlp(hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden // 2, hidden // 4),
        nn.GELU(),
        nn.Linear(hidden // 4, out_dim),
    )


class CoralHead(nn.Module):
    def __init__(self, hidden: int, n_thresholds: int):
        super().__init__()
        self.score = make_mlp(hidden, 1)
        self.thresholds = nn.Parameter(torch.arange(n_thresholds).float())

    def forward(self, pooled):
        return self.score(pooled) - torch.sort(self.thresholds).values


class SentimentModel(XLMRobertaPreTrainedModel):
    def __init__(
        self,
        config,
        geometry: str = "default",
        objective: ObjectiveConfig | None = None,
        class_prior: list[float] | None = None,
    ):
        super().__init__(config)
        self.geometry = geometry
        self.objective = objective or ObjectiveConfig()
        self.class_prior = class_prior or [1.0 / self.objective.n_classes] * self.objective.n_classes
        self.roberta = XLMRobertaModel(config)
        hidden = config.hidden_size

        if self.objective.name in {"classification", "ordinal_soft_ce"}:
            self.classifier = self._mlp(hidden, self.objective.n_classes)
        elif self.objective.name == "coral":
            self.coral_head = CoralHead(hidden, self.objective.coral_num_thresholds)
        elif geometry == "mobius":
            self.intensity_head = self._mlp(hidden, 1)
            self.sarcasm_head = self._mlp(hidden, 1)
            self.sarcasm_head[-1].bias.data.fill_(-3.0)
        else:
            out_dim = 2 if geometry == "circular" else 1
            self.regressor = self._mlp(hidden, out_dim)

        self.loss_fct = nn.HuberLoss(delta=0.75)
        self.post_init()

    @staticmethod
    def _mlp(hidden: int, out_dim: int) -> nn.Sequential:
        return make_mlp(hidden, out_dim)

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        objective: ObjectiveConfig | None = None,
        class_prior: list[float] | None = None,
    ):
        objective = objective or ObjectiveConfig()
        config = AutoConfig.from_pretrained(model_config.name)
        config.num_labels = objective.n_classes
        model = cls.from_pretrained(
            model_config.name,
            config=config,
            geometry=model_config.geometry,
            objective=objective,
            class_prior=class_prior,
        )

        if model_config.kind == "lora_bert":
            from peft import LoraConfig, get_peft_model

            if objective.name in {"classification", "ordinal_soft_ce"}:
                modules_to_save = ["classifier"]
            elif objective.name == "coral":
                modules_to_save = ["coral_head"]
            elif model_config.geometry == "mobius":
                modules_to_save = ["intensity_head", "sarcasm_head"]
            else:
                modules_to_save = ["regressor"]
            lora_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
                modules_to_save=modules_to_save,
                lora_dropout=model_config.lora_dropout,
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, lora_config)
        elif model_config.kind != "bert":
            raise ValueError("model.kind must be 'bert' or 'lora_bert'")

        return model

    def forward(self, input_ids=None, attention_mask=None, labels=None, sample_weight=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self._predict(pooled)

        loss = None
        if labels is not None:
            loss = self._loss(logits, labels, sample_weight)
        return {"loss": loss, "logits": logits}

    @staticmethod
    def _weighted_mean(losses, sample_weight):
        if sample_weight is None:
            return losses.mean()
        sample_weight = sample_weight.to(losses.device, dtype=losses.dtype).view(-1)
        return (losses.view(-1) * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

    def _loss(self, logits, labels, sample_weight=None):
        labels = labels.view(-1)
        if self.objective.name == "classification":
            losses = F.cross_entropy(logits, labels.long(), reduction="none")
            return self._weighted_mean(losses, sample_weight)
        if self.objective.name == "ordinal_soft_ce":
            targets = ordinal_soft_targets(
                labels,
                tau_by_label=self.objective.tau,
                rho_by_label=self.objective.rho,
                prior=self.class_prior,
                n_classes=self.objective.n_classes,
            ).to(logits.dtype)
            losses = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            return self._weighted_mean(losses, sample_weight)
        if self.objective.name == "coral":
            thresholds = torch.arange(
                self.objective.coral_num_thresholds,
                device=labels.device,
                dtype=labels.dtype,
            )
            targets = (labels[:, None] > thresholds[None, :]).float()
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
            return self._weighted_mean(losses, sample_weight)
        if self.objective.loss == "mse":
            losses = F.mse_loss(logits.view(-1), labels.float(), reduction="none")
        else:
            losses = F.huber_loss(logits.view(-1), labels.float(), delta=0.75, reduction="none")
        return self._weighted_mean(losses, sample_weight)

    def _predict(self, pooled):
        if self.objective.name in {"classification", "ordinal_soft_ce"}:
            return self.classifier(pooled)

        if self.objective.name == "coral":
            return self.coral_head(pooled)

        if self.geometry == "default":
            return 4.0 * torch.sigmoid(self.regressor(pooled))

        if self.geometry == "circular":
            vecs = F.normalize(self.regressor(pooled), p=2, dim=1)
            theta = torch.atan2(vecs[:, 1], vecs[:, 0])
            return ((theta + 2 * torch.pi) % (2 * torch.pi)) / (2 * torch.pi) * 4.0

        if self.geometry == "mobius":
            intensity = torch.sigmoid(self.intensity_head(pooled)) * 4.0
            sarcasm = torch.sigmoid(self.sarcasm_head(pooled))
            return (1 - sarcasm) * intensity + sarcasm * (4.0 - intensity)

        raise ValueError("model.geometry must be 'default', 'circular', or 'mobius'")
