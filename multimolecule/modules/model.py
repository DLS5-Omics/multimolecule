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

from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn

from .backbones import BACKBONES
from .heads import HEADS
from .necks import NECKS
from .registry import MODELS


@MODELS.register(default=True)
class MultiMoleculeModel(nn.Module):

    whitelist: list[str] = ["weight", "conv", "fc"]
    blacklist: list[str] = ["bias", "bn", "norm"]

    def __init__(
        self,
        backbone: dict,
        heads: dict,
        neck: dict | None = None,
        max_length: int = 1024,
        truncation: bool = False,
        probing: bool = False,
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
            embed_dim = self.backbone.sequence.config.hidden_size
            attention_heads = self.backbone.sequence.config.num_attention_heads
            neck.update(
                {
                    "num_discrete": num_discrete,
                    "num_continuous": num_continuous,
                    "embed_dim": embed_dim,
                    "attention_heads": attention_heads,
                    "max_length": max_length,
                    "truncation": truncation,
                }
            )
            self.neck = NECKS.build(**neck)
            out_channels = self.neck.out_channels
        else:
            self.neck = None

        # Heads
        for head in heads.values():
            if "hidden_size" not in head or head["hidden_size"] is None:
                head["hidden_size"] = out_channels
        self.heads = nn.ModuleDict({name: HEADS.build(backbone, head) for name, head in heads.items()})
        if any(getattr(h, "require_attentions", False) for h in self.heads.values()):
            self.backbone.sequence.config.output_attentions = True

        if probing:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        sequence: NestedTensor | Tensor,
        discrete: Tensor | None = None,
        continuous: Tensor | None = None,
        dataset: str | None = None,
        **labels: NestedTensor | Tensor,
    ) -> FlatDict:
        ret = FlatDict()
        output, _ = self.backbone(sequence, discrete, continuous)
        if self.neck is not None:
            output = self.neck(**output)
        for task, label in labels.items():
            ret[task] = self.heads[task](output, input_ids=sequence, labels=label)
        return ret

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

        heads_param_groups = categorize_parameters(self.heads, lr, weight_decay)
        trainable_parameters.extend(heads_param_groups)

        if isinstance(self.backbone, nn.Module):
            backbone_param_groups = categorize_parameters(self.backbone, lr, weight_decay, lr_ratio=pretrained_ratio)
            trainable_parameters.extend(backbone_param_groups)

        if isinstance(self.neck, nn.Module):
            neck_param_groups = categorize_parameters(self.neck, lr, weight_decay)
            trainable_parameters.extend(neck_param_groups)

        return trainable_parameters
