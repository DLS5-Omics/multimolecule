# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

from math import prod
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from chanfig import Registry
from torch import Tensor

LossBalancerRegistry = Registry()


class LossBalancer(nn.Module):
    """Base class for loss balancers in multi-task learning.

    This class provides an interface for implementing various strategies
    to balance the losses of different tasks in a multi-task learning setup.
    """

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        """Compute the balanced total loss.

        Args:
            losses (Dict[str, Tensor]): A dictionary of task names to their respective losses.

        Returns:
            Tensor: The computed balanced loss.
        """
        return {k: v["loss"] for k, v in ret.items()}


@LossBalancerRegistry.register("equal", default=True)
class EqualWeightBalancer(LossBalancer):
    """Equal Weighting Balancer.

    This method assigns equal weight to each task's loss, effectively averaging the losses across all tasks.
    """

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)
        return sum(losses.values()) / len(losses)


@LossBalancerRegistry.register("random")
class RandomLossWeightBalancer(LossBalancer):
    """Random Loss Weighting Balancer.

    This method assigns random weights to each task's loss, which are sampled from a softmax distribution,
    as described in the paper "Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning"
    by Liang et al. (https://openreview.net/forum?id=jjtFD8A1Wx).
    """

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)
        loss = torch.stack(list(losses.values()))
        weight = F.softmax(torch.randn(len(losses), device=loss.device, dtype=loss.dtype), dim=-1)
        return loss.T @ weight


@LossBalancerRegistry.register("geometric")
class GeometricLossBalancer(LossBalancer):
    """Geometric Loss Strategy Balancer.

    This method computes the geometric mean of the task losses, which can be useful for balancing tasks with different
    scales, as described in the paper "MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for
    Multi-Task Learning" by Chennupati et al. (https://arxiv.org/abs/1904.08492).
    """

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)
        return prod(losses.values()) ** (1 / len(losses))


@LossBalancerRegistry.register("uncertainty")
class UncertaintyWeightBalancer(LossBalancer):
    """Uncertainty Weighting Balancer.

    This method uses task uncertainty to weight the losses, as described in the paper "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics" by Kendall et al. (https://arxiv.org/abs/1705.07115).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_vars = nn.ParameterDict()

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)
        for task, loss in losses.items():
            if task not in self.log_vars:
                self.log_vars[task] = nn.Parameter(torch.zeros(1, device=loss.device))

        weighted_losses = [
            torch.exp(-self.log_vars[task]) * loss + self.log_vars[task] for task, loss in losses.items()
        ]
        return sum(weighted_losses) / len(weighted_losses)


@LossBalancerRegistry.register("dynamic")
class DynamicWeightAverageBalancer(LossBalancer):
    """Dynamic Weight Average Balancer.

    This method dynamically adjusts the weights of task losses based on their relative changes over time, as described
    in the paper "End-to-End Multi-Task Learning with Attention" by Liu et al. (https://arxiv.org/abs/1803.10704).
    """

    def __init__(self, *args, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.task_losses_history: List[List[float]] = []

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)
        if len(self.task_losses_history) < 2:
            self.task_losses_history.append([loss.item() for loss in losses.values()])
            return sum(losses.values()) / len(losses)

        curr_losses = [loss.item() for loss in losses.values()]
        prev_losses = self.task_losses_history[-1]
        loss_ratios = [c / (p + 1e-8) for c, p in zip(curr_losses, prev_losses)]

        exp_weights = torch.exp(torch.tensor(loss_ratios) / self.temperature)
        weights = len(losses) * F.softmax(exp_weights, dim=-1)

        self.task_losses_history.append(curr_losses)
        if len(self.task_losses_history) > 2:
            self.task_losses_history.pop(0)

        return sum(w * l for w, l in zip(weights, losses.values())) / len(losses)


@LossBalancerRegistry.register("gradnorm")
class GradNormBalancer(LossBalancer):
    """GradNorm Balancer.

    This method balances task losses by normalizing gradients, as described in the paper "GradNorm: Gradient
    Normalization for Adaptive Loss Balancing in Deep Multitask Networks" by Chen et al.
    (https://arxiv.org/abs/1711.02257).
    """

    def __init__(self, *args, alpha: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.task_weights = nn.ParameterDict()
        self.initial_losses: Dict[str, Tensor] = {}

    def forward(self, ret: Dict[str, Tensor]) -> Tensor:
        losses = super().forward(ret)

        for task, loss in losses.items():
            if task not in self.task_weights:
                self.task_weights[task] = nn.Parameter(torch.ones(1, device=loss.device))
                self.initial_losses[task] = loss.detach()

        loss_ratios = {task: loss / (self.initial_losses[task] + 1e-8) for task, loss in losses.items()}
        avg_loss_ratio = sum(loss_ratios.values()) / len(loss_ratios)

        relative_inverse_rates = {
            task: (ratio / (avg_loss_ratio + 1e-8)) ** self.alpha for task, ratio in loss_ratios.items()
        }

        weighted_losses = {task: self.task_weights[task] * loss for task, loss in losses.items()}
        grad_norms = {
            task: torch.norm(torch.autograd.grad(weighted_loss, self.task_weights[task], retain_graph=True)[0])
            for task, weighted_loss in weighted_losses.items()
        }
        mean_grad_norm = sum(grad_norms.values()) / len(grad_norms)

        for task in losses.keys():
            target_grad = mean_grad_norm * relative_inverse_rates[task]
            grad_norm = grad_norms[task]
            self.task_weights[task].data = torch.clamp(
                self.task_weights[task] * (target_grad / (grad_norm + 1e-8)), min=0.0
            )
        weight_sum = sum(w.item() for w in self.task_weights.values())
        scale = len(losses) / (weight_sum + 1e-8)
        for task in losses.keys():
            self.task_weights[task].data *= scale

        return sum(self.task_weights[task] * loss for task, loss in losses.items())
