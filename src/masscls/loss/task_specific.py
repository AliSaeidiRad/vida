from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw model outputs
            targets: (N, C) one-hot encoded targets
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = logits.size(1)
            targets = (
                targets * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            )

        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Cross entropy with focal weight
        loss = -targets * focal_weight * log_probs

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
            alpha = self.alpha.view(1, -1).to(logits.device)
            loss = loss * alpha

        # Reduction
        if self.reduction == "mean":
            return loss.sum(dim=1).mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.sum(dim=1)


__all__ = ["TaskSpecificFocalLoss"]
