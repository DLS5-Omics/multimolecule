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

from collections.abc import Mapping
from functools import wraps
from math import prod
from typing import Callable, Tuple, Type

import torch
from chanfig import FlatDict
from torch import nn

MODULE_FLOPS = FlatDict()


def register_MODULE_FLOPS(module_type):

    @wraps(module_type)
    def decorator(func):
        MODULE_FLOPS[module_type] = func
        return func

    return decorator


MODULE_FLOPS.register = register_MODULE_FLOPS

ignored_modules = (nn.Identity, nn.Flatten, nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Parameter)


def calculate_flops(
    model: nn.Module,
    input: torch.Tensor,
    module_flops: Mapping[str, Callable] | None = None,
    excluded_modules: Type | Tuple[Type] | None = None,
    format_spec: str | None = None,
) -> int | str:
    """
    Calculate the number of FLOPs in a PyTorch model.

    Args:
        model (torch.nn.Module): The model for which to calculate the FLOPs.
        input (torch.Tensor): The input tensor to the model.
        module_flops (Mapping[str, Callable]): A mapping of module types to functions that calculate
            the FLOPs for that module.
        excluded_modules (Type | Tuple[Type]): A module type or a tuple of module types to exclude from the calculation.
        format_spec (str, optional): A format specifier to format the output.
            If is None, the number of parameters is returned as an int.
            If is not None, the number of parameters is returned as a str formatted according to the format specifier.
            Default to None.

    Returns:
        int | str: The number of FLOPs in the model.

    Examples:
        >>> from torchvision import models
        >>> model = models.vgg16()
        >>> input = torch.randn(1, 3, 224, 224)
        >>> calculate_flops(model, input, excluded_modules=models.VGG)
        138357544
        >>> calculate_flops(model, input, format_spec=",")
        '138,357,544'

        >>> model = models.resnet50()
        >>> input = torch.randn(1, 3, 224, 224)
        >>> calculate_flops(model, input)
        25557032
        >>> calculate_flops(model, input, format_spec=",")
        '25,557,032'
    """

    module_flops = MODULE_FLOPS.update(module_flops or {})

    hooks = []
    flops = 0

    for module in model.modules():

        if isinstance(module, ignored_modules):
            continue

        if excluded_modules and isinstance(module, excluded_modules):
            continue

        def hook(module, input, output):
            nonlocal flops
            flops += MODULE_FLOPS[type(module)](module, input, output)

        hooks.append(module.register_forward_hook(hook))

    # Run the input through the model to trigger the hooks
    model(input)

    # Remove the hooks after calculation
    for hook in hooks:
        hook.remove()  # type: ignore[attr-defined]

    if format_spec is not None:
        return format(flops, format_spec)
    return flops


@MODULE_FLOPS.register(nn.Linear)
def linear_hook(module, input, output):
    # For linear layer, FLOPs is input dimension times output dimension
    return input[0].nelement() * output[0].nelement()


@MODULE_FLOPS.register(nn.Conv1d)
@MODULE_FLOPS.register(nn.Conv2d)
@MODULE_FLOPS.register(nn.Conv3d)
def conv_hook(module, input, output):
    # For Conv, FLOPs is the number of output elements times the kernel size
    return prod(module.kernel_size) * output[0].nelement()


@MODULE_FLOPS.register(nn.ReLU)
def relu_hook(module, input, output):
    # For ReLU, FLOPs is the number of output elements
    # (each output is the result of a comparison operation)
    return output[0].nelement()


@MODULE_FLOPS.register(nn.GELU)
def gelu_hook(module, input, output):
    # For GELU, FLOPs is 3 times the number of output elements
    # (each output is the result of an addition, a multiplication, and a division)
    return 3 * output[0].nelement()


@MODULE_FLOPS.register(nn.Sigmoid)
def sigmoid_hook(module, input, output):
    # For sigmoid, FLOPs is 4 times the number of output elements
    # (each output is the result of an exponentiation, an addition, and a division)
    return 4 * output[0].nelement()


@MODULE_FLOPS.register(nn.Softmax)
def softmax_hook(module, input, output):
    # For softmax, FLOPs is 2 times the number of output elements times the input dimension
    # (each output is the result of an exponentiation, a sum, and a division)
    return 2 * output[0].nelement() * input[0].size(-1)


@MODULE_FLOPS.register(nn.Dropout)
def dropout_hook(module, input, output):
    # For dropout, FLOPs is the number of output elements
    # (each output is the result of a comparison operation)
    return output[0].nelement()


@MODULE_FLOPS.register(nn.MultiheadAttention)
def multihead_attention_hook(module, input, output):
    # For MultiheadAttention, FLOPs is the sum of FLOPs of query, key, value projections and
    # scaled dot-product attention
    return 3 * module.in_proj_weight.nelement() + module.out_proj.nelement() + module.nhead * input[0].size(-1) ** 2


@MODULE_FLOPS.register(nn.LayerNorm)
def layernorm_hook(module, input, output):
    # For LayerNorm, FLOPs is 4 times the number of input elements
    return 4 * input[0].nelement()


@MODULE_FLOPS.register(nn.BatchNorm1d)
@MODULE_FLOPS.register(nn.BatchNorm2d)
@MODULE_FLOPS.register(nn.BatchNorm3d)
def batchnorm_hook(module, input, output):
    # For BatchNorm, FLOPs is 2 times the number of output elements
    # (each output is the result of a subtraction and a division)
    return 2 * output[0].nelement()


@MODULE_FLOPS.register(nn.MaxPool1d)
@MODULE_FLOPS.register(nn.MaxPool2d)
@MODULE_FLOPS.register(nn.MaxPool3d)
def maxpool_hook(module, input, output):
    # For MaxPool, FLOPs is the number of output elements
    # (each output is the result of a comparison operation)
    return output[0].nelement()


@MODULE_FLOPS.register(nn.AvgPool1d)
@MODULE_FLOPS.register(nn.AvgPool2d)
@MODULE_FLOPS.register(nn.AvgPool3d)
def avgpool3d_hook(module, input, output):
    # For AvgPool, FLOPs is the number of output elements times the kernel size
    # (each output is the result of a sum and a division)
    return prod(module.kernel_size) * output[0].nelement()


@MODULE_FLOPS.register(nn.AdaptiveAvgPool1d)
@MODULE_FLOPS.register(nn.AdaptiveAvgPool2d)
@MODULE_FLOPS.register(nn.AdaptiveAvgPool3d)
def adaptive_avgpool3d_hook(module, input, output):
    # For AdaptiveAvgPool, FLOPs is the number of output elements times the input shape
    # (each output is the result of a sum and a division)
    return prod(input[0][2:]) * output[0].nelement()
