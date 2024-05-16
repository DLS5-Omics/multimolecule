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

from torch import nn


def count_parameters(
    model: nn.Module, trainable: bool = True, unique: bool = True, format_spec: str | None = None
) -> int | str:
    """
    Count the number of parameters in a PyTorch model, optionally only counting
    those that require gradients (i.e., are trainable) and/or are unique.

    Args:
        model (torch.nn.Module): The model for which to count the parameters.
        trainable (bool, optional): Whether to count only parameters that require gradients.
            Default to True.
        unique (bool, optional): Whether to count only unique parameters.
            Default to True.
        format_spec (str, optional): A format specifier to format the output.
            If is None, the number of parameters is returned as an int.
            If is not None, the number of parameters is returned as a str formatted according to the format specifier.
            Default to None.

    Returns:
        int | str: The number of parameters in the model, according to the criteria specified by `trainable` and
            `unique`.

    Examples:
        >>> from torchvision import models

        >>> model = models.alexnet()
        >>> count_parameters(model)
        61100840
        >>> count_parameters(model, format_spec=",")
        '61,100,840'

        >>> model = models.vgg16()
        >>> count_parameters(model)
        138357544
        >>> count_parameters(model, format_spec=",")
        '138,357,544'

        >>> model = models.resnet50()
        >>> count_parameters(model)
        25557032
        >>> count_parameters(model, format_spec=",")
        '25,557,032'
    """

    if unique:
        unique_parameters = set()
        num_parameters = 0
        for p in model.parameters():
            if p.data_ptr() in unique_parameters:
                continue
            if (trainable and p.requires_grad) or not trainable:
                unique_parameters.add(p.data_ptr())
                num_parameters += p.numel()
    elif trainable:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters())
    if format_spec is not None:
        return format(num_parameters, format_spec)
    return num_parameters
