"""Abstract base class for all prediction models (GAN, supervised, classifier, etc.)."""
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn


def init_weights_xavier(module: nn.Module):
    """Apply Xavier (Glorot) initialization, as in the source notebook (mx.init.Xavier).

    Linear/Conv1d weights get xavier_uniform; LSTM input-hidden weights get
    xavier_uniform and hidden-hidden weights get orthogonal init (standard
    practice for recurrent weights). All biases are zeroed.
    MultiheadAttention and BatchNorm already use suitable defaults in PyTorch.
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


class PredictionModel(ABC):
    """Common interface for all model types.

    Every model must support:
    - train_epoch(train_loader) -> dict with at least "loss" key
    - predict(xb: Tensor) -> (y_reg, y_cls_logits, y_quantiles) on CPU
    - step_schedulers() -> advance LR schedulers
    - state_dict property for early stopping checkpoint
    """

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """One epoch of training. Returns dict with loss values."""

    @abstractmethod
    def predict(self, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference. Returns (y_reg [B], y_cls [B], y_q [B,Q]) on CPU.

        Models that don't produce all three outputs should return zeros for unused ones.
        """

    def step_schedulers(self):
        """Advance LR schedulers. No-op if not applicable."""

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Return model state for checkpointing (early stopping)."""

    @abstractmethod
    def load_state_dict(self, state: dict):
        """Restore model state from checkpoint."""
