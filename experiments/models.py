from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, XLMRobertaModel, XLMRobertaPreTrainedModel

from .config import ModelConfig


class SentimentModel(XLMRobertaPreTrainedModel):
    def __init__(self, config, geometry: str = "default"):
        super().__init__(config)
        self.geometry = geometry
        self.roberta = XLMRobertaModel(config)
        hidden = config.hidden_size

        if geometry == "mobius":
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
        return nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, out_dim),
        )

    @classmethod
    def from_config(cls, model_config: ModelConfig):
        config = AutoConfig.from_pretrained(model_config.name)
        model = cls.from_pretrained(
            model_config.name,
            config=config,
            geometry=model_config.geometry,
        )

        if model_config.kind == "lora_bert":
            from peft import LoraConfig, get_peft_model

            modules_to_save = ["intensity_head", "sarcasm_head"] if model_config.geometry == "mobius" else ["regressor"]
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self._predict(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
        return {"loss": loss, "logits": logits}

    def _predict(self, pooled):
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
