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

from danling import NestedTensor
from torch import Tensor, nn

from .backbones import BACKBONES
from .heads import HEADS, HeadOutput
from .necks import NECKS
from .registry import MODELS


@MODELS.register(default=True)
class Model(nn.Module):

    whitelist: list[str] = ["weight", "conv", "fc"]
    blacklist: list[str] = ["bias", "bn", "norm"]

    backbone: nn.Module
    neck: nn.Module | None
    head: nn.Module

    def __init__(
        self,
        backbone: dict,
        head: dict,
        neck: dict | None = None,
    ):
        super().__init__()

        # Backbone
        self.backbone = BACKBONES.build(**backbone)
        backbone = self.backbone.config
        out_channels = self.backbone.out_channels

        # Neck
        if neck:
            num_discrete = self.backbone.num_discrete
            num_continuous = self.backbone.num_continuous
            hidden_size = self.backbone.sequence.config.hidden_size
            neck.update(
                {
                    "num_discrete": num_discrete,
                    "num_continuous": num_continuous,
                    "hidden_size": hidden_size,
                }
            )
            self.neck = NECKS.build(**neck)
            out_channels = self.neck.out_channels
        else:
            self.neck = None

        # Head
        if "hidden_size" not in head or head["hidden_size"] is None:
            head["hidden_size"] = out_channels
        self.head = HEADS.build(backbone, head)
        if self.head.require_attentions:
            self.backbone.sequence.config.output_attentions = True

    def forward(
        self,
        sequence: NestedTensor | Tensor,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
        labels: NestedTensor | Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_model_output: bool = False,
    ) -> HeadOutput:
        backbone_output = self.backbone(
            sequence,
            discrete,
            continuous,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        neck_output = self.neck(**backbone_output) if self.neck is not None else backbone_output
        head_output = self.head(neck_output, input_ids=sequence, labels=labels)
        if return_model_output:
            head_output["model"] = backbone_output
        return head_output

    def trainable_parameters(
        self,
        lr: float,
        weight_decay: float,
        pretrained_ratio: float = 1e-2,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> list[dict]:
        """
        Prepare parameter groups with specific optimization settings.

        Args:
            lr: Base learning rate.
            weight_decay: Base weight decay.
            pretrained_ratio: Scaling factor for backbone's learning rate and weight decay.
            whitelist: List of parameter name substrings to include in weight decay.
            blacklist: List of parameter name substrings to exclude from weight decay.

        Returns:
            Parameter groups for the optimizer.
        """

        whitelist = whitelist or self.whitelist
        blacklist = blacklist or self.blacklist
        trainable_parameters: list[dict] = []

        def categorize_parameters(
            module: nn.Module, base_lr: float, base_wd: float, lr_ratio: float = 1.0
        ) -> list[dict]:
            decay_params = []
            no_decay_params = []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(w in name for w in whitelist):
                    decay_params.append(param)
                elif any(b in name for b in blacklist):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups = []
            if decay_params:
                param_groups.append(
                    {"params": decay_params, "weight_decay": base_wd * lr_ratio, "lr": base_lr * lr_ratio}
                )
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr * lr_ratio})
            return param_groups

        head_param_groups = categorize_parameters(self.head, lr, weight_decay)
        trainable_parameters.extend(head_param_groups)

        if isinstance(self.backbone, nn.Module):
            backbone_param_groups = categorize_parameters(self.backbone, lr, weight_decay, lr_ratio=pretrained_ratio)
            trainable_parameters.extend(backbone_param_groups)

        if isinstance(self.neck, nn.Module):
            neck_param_groups = categorize_parameters(self.neck, lr, weight_decay)
            trainable_parameters.extend(neck_param_groups)

        return trainable_parameters
