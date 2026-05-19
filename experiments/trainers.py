from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from transformers import Trainer


class SentimentTrainer(Trainer):
    def __init__(self, *args, method: str = "default", gradvac_beta: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.gradvac_beta = gradvac_beta
        self.gradvac_rho = 0.0

    @staticmethod
    def metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.asarray(logits).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        rounded = np.rint(np.clip(preds, 0, 4))
        return {
            "mae": float(mean_absolute_error(labels, preds)),
            "rounded_mae": float(mean_absolute_error(labels, rounded)),
        }

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.method == "default":
            return super().training_step(model, inputs, num_items_in_batch, **kwargs)

        inputs = self._prepare_inputs(inputs)
        langs = inputs.pop("lang", None)
        if langs is None:
            return super().training_step(model, inputs, num_items_in_batch, **kwargs)

        en_idx = (langs == 0).nonzero(as_tuple=True)[0]
        de_idx = (langs == 1).nonzero(as_tuple=True)[0]
        if len(en_idx) == 0 or len(de_idx) == 0:
            loss = self.compute_loss(model, inputs)
            self.accelerator.backward(loss)
            return loss.detach()

        inputs_en = {key: value[en_idx] for key, value in inputs.items()}
        inputs_de = {key: value[de_idx] for key, value in inputs.items()}
        loss_en, grads_en = self._loss_and_grads(model, inputs_en)
        loss_de, grads_de = self._loss_and_grads(model, inputs_de)

        model.zero_grad()
        params = [p for p in model.parameters() if p.requires_grad]
        if self.method == "pcgrad":
            self._apply_pcgrad(params, grads_en, grads_de)
        elif self.method == "gradvac":
            self._apply_gradvac(params, grads_en, grads_de)
        else:
            raise ValueError("trainer must be 'default', 'pcgrad', or 'gradvac'")

        return (loss_en + loss_de).detach() / 2.0

    def _loss_and_grads(self, model, inputs):
        model.zero_grad()
        loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        return loss, grads

    @staticmethod
    def _apply_pcgrad(params, grads_en, grads_de):
        with torch.no_grad():
            for param, grad_en, grad_de in zip(params, grads_en, grads_de):
                dot = torch.dot(grad_en.flatten(), grad_de.flatten())
                if dot < 0:
                    en_norm = torch.dot(grad_en.flatten(), grad_en.flatten()) + 1e-8
                    de_norm = torch.dot(grad_de.flatten(), grad_de.flatten()) + 1e-8
                    param.grad = grad_en - (dot / de_norm) * grad_de + grad_de - (dot / en_norm) * grad_en
                else:
                    param.grad = grad_en + grad_de

    def _apply_gradvac(self, params, grads_en, grads_de):
        with torch.no_grad():
            flat_en = torch.cat([grad.flatten() for grad in grads_en])
            flat_de = torch.cat([grad.flatten() for grad in grads_de])
            norm_en = torch.norm(flat_en)
            norm_de = torch.norm(flat_de)
            cos_theta = torch.dot(flat_en, flat_de) / (norm_en * norm_de + 1e-8)
            self.gradvac_rho = self.gradvac_beta * self.gradvac_rho + (1 - self.gradvac_beta) * cos_theta.item()

            if cos_theta < self.gradvac_rho:
                target = torch.tensor(self.gradvac_rho, device=cos_theta.device)
                sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-8)
                target_sin = torch.sqrt(1 - target**2 + 1e-8)
                phi = (norm_en * (target * sin_theta - cos_theta * target_sin)) / (norm_de * target_sin + 1e-8)
                for param, grad_en, grad_de in zip(params, grads_en, grads_de):
                    param.grad = grad_en + phi * grad_de + grad_de
            else:
                for param, grad_en, grad_de in zip(params, grads_en, grads_de):
                    param.grad = grad_en + grad_de
