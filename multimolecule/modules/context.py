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

from collections.abc import Iterator
from contextlib import contextmanager

from torch import Tensor, nn


@contextmanager
def preserve_batch_norm_stats(module: nn.Module) -> Iterator[None]:
    snapshots: list[tuple[nn.modules.batchnorm._BatchNorm, Tensor | None, Tensor | None, Tensor | None]] = []
    for submodule in module.modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm) and submodule.track_running_stats:
            snapshots.append(
                (
                    submodule,
                    submodule.running_mean.clone() if submodule.running_mean is not None else None,
                    submodule.running_var.clone() if submodule.running_var is not None else None,
                    submodule.num_batches_tracked.clone() if submodule.num_batches_tracked is not None else None,
                )
            )
    try:
        yield
    finally:
        for batch_norm, running_mean, running_var, num_batches_tracked in snapshots:
            if running_mean is not None:
                batch_norm.running_mean.copy_(running_mean)
            if running_var is not None:
                batch_norm.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                batch_norm.num_batches_tracked.copy_(num_batches_tracked)
