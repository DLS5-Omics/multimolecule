# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from transformers.configuration_utils import PretrainedConfig

from multimolecule.module import BaseHeadConfig, HeadConfig, MaskedLMHeadConfig

__all__ = ["PreTrainedConfig", "BaseHeadConfig", "HeadConfig", "MaskedLMHeadConfig"]


class PreTrainedConfig(PretrainedConfig):
    r"""
    Base class for all model configuration classes.
    """

    head: HeadConfig

    hidden_size: int

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    mask_token_id: int = 4
    null_token_id: int = 5

    def __init__(
        self, pad_token_id=0, bos_token_id=1, eos_token_id=2, unk_token_id=3, mask_token_id=4, null_token_id=5, **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            mask_token_id=mask_token_id,
            null_token_id=null_token_id,
            **kwargs,
        )

    def to_dict(self):
        output = super().to_dict()
        for k, v in output.items():
            if hasattr(v, "to_dict"):
                output[k] = v.to_dict()
            if is_dataclass(v):
                output[k] = asdict(v)
        return output
