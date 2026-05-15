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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from chanfig import Registry as Registry_

if TYPE_CHECKING:
    from .model import ModelBase


class Registry(Registry_):  # pylint: disable=too-few-public-methods
    def build(self, *args: Any, **kwargs: Any) -> ModelBase:
        type_ = kwargs.pop("type", None) or "auto"
        if type_ == "auto":
            type_ = self._select_auto(kwargs)
        return super().build(*args, type=type_, **kwargs)

    @staticmethod
    def _select_auto(config: dict) -> str:
        """Pick `mono` when the config matches every constraint `MonoModel` enforces; otherwise
        `poly` (the universal fallback).

        Constraints: no neck, exactly one head, sequence-only backbone, `backbone.sequence.use_pretrained`
        truthy, and the head's `type` resolves to a kind that has an `AutoModelFor*` counterpart.
        """
        if config.get("neck"):
            return "poly"
        heads = config.get("heads") or {}
        if len(heads) != 1:
            return "poly"
        backbone = config.get("backbone")
        if not isinstance(backbone, Mapping) or set(backbone.keys()) != {"sequence"}:
            return "poly"
        sequence = backbone.get("sequence")
        if not isinstance(sequence, Mapping) or not sequence.get("use_pretrained", True):
            return "poly"
        head_config = next(iter(heads.values()))
        if not isinstance(head_config, Mapping):
            return "poly"
        head_type = head_config.get("type")
        if not head_type:
            return "poly"
        from .model import _MONO_HEAD_TYPES

        return "mono" if head_type.split(".", 1)[0] in _MONO_HEAD_TYPES else "poly"


MODELS = Registry()

__all__ = ["MODELS"]
