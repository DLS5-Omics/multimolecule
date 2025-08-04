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

import os

import chanfig
import torch

from multimolecule.models import PreTrainedConfig
from multimolecule.models import RnaMoeConfig as Config
from multimolecule.models import RnaMoeForPreTraining
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import load_checkpoint, save_checkpoint
from multimolecule.modules import HEADS

torch.manual_seed(1016)


def convert_checkpoint(convert_config):
    print(f"Converting RnaMoe checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    config.architectures = ["RnaMoeModel"]

    model = RnaMoeForPreTraining(config)
    cfg = chanfig.load(os.path.join(os.path.dirname(convert_config.checkpoint_path), "runner.yaml")).network
    cfg.heads.secondary_structure.setdefault("num_layers", 8)
    for name, head in cfg.heads.items():
        head.transform = "nonlinear"
        setattr(model, name, HEADS.build(PreTrainedConfig(), head))

    ckpt = torch.load(convert_config.checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    ckpt = ckpt.get("ema", ckpt)
    state_dict = _convert_checkpoint(config, ckpt)

    load_checkpoint(model, state_dict)

    model.ss_head = model.secondary_structure
    del model.secondary_structure

    save_checkpoint(convert_config, model)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _convert_checkpoint(config, original_state_dict):
    state_dict = {}
    for key, value in original_state_dict.items():
        if key in ["initted", "step"]:
            continue
        if key.startswith("ema_model."):
            key = key[10:]
        if key.startswith("heads."):
            key = key[6:]
        key = key.replace("backbone.sequence", "rnamoe")
        state_dict[key] = value

    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
