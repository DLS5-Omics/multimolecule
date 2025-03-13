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

import danling as dl
import transformers
from chanfig import Registry as Registry_
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class Registry(Registry_):  # pylint: disable=too-few-public-methods
    def build(
        self,
        type: str | None = None,
        name: str | None = None,
        use_pretrained: bool = True,
        gradient_checkpoint: bool = False,
        checkpoint: str | None = None,
        *args,
        **kwargs,
    ) -> nn.Module:
        if type is not None:
            if type in self:
                sequence_cls = self.lookup(type)
                sequence = self.init(sequence_cls, *args, **kwargs)
                if checkpoint is not None:
                    sequence.load_state_dict(dl.load(checkpoint))
            elif hasattr(transformers, type + "Model"):
                if use_pretrained:
                    sequence_cls: PreTrainedModel = getattr(transformers, type + "Model")  # type: ignore[no-redef]
                    sequence = sequence_cls.from_pretrained(name, *args, **kwargs)
                else:
                    config_cls: PretrainedConfig = getattr(transformers, type + "Config")
                    config, kwargs = config_cls.from_pretrained(name, return_unused_kwargs=True, **kwargs)
                    sequence_cls: PreTrainedModel = getattr(transformers, type + "Model")  # type: ignore[no-redef]
                    sequence = sequence_cls.from_config(config, *args, **kwargs)
            else:
                raise ValueError(f"Sequence {type} not found in registry or transformers")
        else:
            if use_pretrained:
                sequence = AutoModel.from_pretrained(name, *args, **kwargs)
            else:
                config, kwargs = AutoConfig.from_pretrained(name, return_unused_kwargs=True, **kwargs)
                sequence = AutoModel.from_config(config, *args, **kwargs)

        if gradient_checkpoint:
            sequence.gradient_checkpointing_enable()
        return sequence


SEQUENCES = Registry()
