from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        task_losses: Dict[str, nn.Module],
        task_weights: Optional[Dict[str, float]] = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.task_losses = nn.ModuleDict(task_losses)

        if learnable_weights:
            # Learnable task weights (uncertainty weighting)
            self.log_vars = nn.ParameterDict(
                {task: nn.Parameter(torch.zeros(1)) for task in task_losses.keys()}
            )
            self.task_weights = None
        else:
            self.log_vars = None
            if task_weights is None:
                task_weights = {task: 1.0 for task in task_losses.keys()}
            self.task_weights = task_weights

    def forward(
        self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        total_loss = 0.0

        for task_name, loss_fn in self.task_losses.items():
            task_loss = loss_fn(logits[task_name], targets[task_name])

            if self.log_vars is not None:
                # Uncertainty weighting: L = (1/2σ²)L_task + log(σ)
                precision = torch.exp(-self.log_vars[task_name])
                weighted_loss = precision * task_loss + self.log_vars[task_name]
            else:
                weight = self.task_weights.get(task_name, 1.0)  # type: ignore
                weighted_loss = weight * task_loss

            total_loss = total_loss + weighted_loss

        return total_loss


__all__ = ["MultiTaskLoss"]
