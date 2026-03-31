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

from collections.abc import Callable, Mapping
from math import prod
from typing import Tuple, Type

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

# Registry mapping nn.Module subclasses to functions computing (flops, macs).
# Hook signature: (module, input, output) -> tuple[int, int]
_MODULE_OPS: dict[type, Callable] = {}

_IGNORED_MODULES: tuple[type, ...] = (
    nn.Identity,
    nn.Flatten,
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
)


def _register(*module_types: type):
    def decorator(func: Callable) -> Callable:
        for t in module_types:
            _MODULE_OPS[t] = func
        return func

    return decorator


def _get_output_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, PackedSequence):
        return output.data
    if isinstance(output, (tuple, list)):
        for o in output:
            try:
                return _get_output_tensor(o)
            except TypeError:
                continue
    # HuggingFace ModelOutput
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Cannot extract tensor from output of type {type(output)}")


def _resolve_module_ops(module: nn.Module, ops_registry: Mapping[type, Callable]) -> Callable | None:
    for module_type in type(module).mro():
        if module_type in ops_registry:
            return ops_registry[module_type]
    return None


def _estimate_attention_ops(model: nn.Module, seq_length: int) -> Tuple[int, int]:
    """Estimate FLOPs/MACs for attention score computation not captured by hooks.

    HuggingFace models compute Q@K^T and attn@V via torch.matmul or
    F.scaled_dot_product_attention, which are not nn.Module subclasses
    and cannot be captured by forward hooks. This function estimates
    those operations from the model config.
    """

    config = getattr(model, "config", None)
    if config is None:
        return 0, 0

    num_layers = getattr(config, "num_hidden_layers", 0)
    num_heads = getattr(config, "num_attention_heads", 0)
    hidden_size = getattr(config, "hidden_size", 0)

    if not all((num_layers, num_heads, hidden_size)):
        return 0, 0

    head_dim = hidden_size // num_heads

    # Per layer:
    #   Q@K^T: seq * seq * head_dim * num_heads MACs
    #   attn@V: seq * seq * head_dim * num_heads MACs
    #   softmax: ~3 * num_heads * seq^2 FLOPs (no MACs)
    attn_macs_per_layer = 2 * num_heads * seq_length * seq_length * head_dim
    attn_flops_per_layer = 2 * attn_macs_per_layer
    softmax_flops_per_layer = 3 * num_heads * seq_length * seq_length

    total_macs = attn_macs_per_layer * num_layers
    total_flops = (attn_flops_per_layer + softmax_flops_per_layer) * num_layers
    return total_flops, total_macs


def _calculate_ops(
    model: nn.Module,
    *args,
    module_ops: Mapping[type, Callable] | None = None,
    excluded_modules: Type | Tuple[Type, ...] | None = None,
    **kwargs,
) -> Tuple[int, int]:
    """Run a forward pass with hooks and return (total_flops, total_macs)."""

    ops_registry = {**_MODULE_OPS, **(module_ops or {})}
    total_flops = 0
    total_macs = 0
    hooks = []

    def _make_hook(ops_fn: Callable):
        def hook(module, input, output):
            nonlocal total_flops, total_macs
            f, m = ops_fn(module, input, output)
            total_flops += f
            total_macs += m

        return hook

    for module in model.modules():
        if isinstance(module, _IGNORED_MODULES):
            continue
        if excluded_modules and isinstance(module, excluded_modules):
            continue
        ops_fn = _resolve_module_ops(module, ops_registry)
        if ops_fn is not None:
            hooks.append(module.register_forward_hook(_make_hook(ops_fn)))

    training = model.training
    model.eval()
    with torch.no_grad():
        output = model(*args, **kwargs)  # noqa: F841
    model.train(training)

    for h in hooks:
        h.remove()

    # Estimate attention ops not captured by hooks
    seq_length = 0
    if args:
        inp = args[0]
        if isinstance(inp, torch.Tensor) and inp.dim() >= 2:
            seq_length = inp.shape[1]
    elif "input_ids" in kwargs:
        inp = kwargs["input_ids"]
        if isinstance(inp, torch.Tensor) and inp.dim() >= 2:
            seq_length = inp.shape[1]

    if seq_length > 0:
        attn_flops, attn_macs = _estimate_attention_ops(model, seq_length)
        total_flops += attn_flops
        total_macs += attn_macs

    return total_flops, total_macs


def calculate_flops(
    model: nn.Module,
    *args,
    module_ops: Mapping[type, Callable] | None = None,
    excluded_modules: Type | Tuple[Type, ...] | None = None,
    format_spec: str | None = None,
    **kwargs,
) -> int | str:
    """
    Calculate the number of FLOPs (floating point operations) in a PyTorch model.

    This performs a single forward pass with hooks attached to count operations
    per module. For HuggingFace transformer models, attention matrix operations
    (Q@K^T and attn@V) that use raw torch.matmul are automatically estimated
    from the model config.

    Args:
        model (torch.nn.Module): The model for which to calculate the FLOPs.
        *args: Positional arguments forwarded to ``model.forward()``.
        module_ops (Mapping[type, Callable], optional): Custom per-module-type
            hooks. Each callable has signature ``(module, input, output) -> (flops, macs)``.
            Overrides built-in hooks for the same type.
        excluded_modules (type | tuple[type, ...], optional): Module types to
            exclude from the calculation.
        format_spec (str, optional): A format specifier to format the output.
            If is None, the number of FLOPs is returned as an int.
            If is not None, the number of FLOPs is returned as a str formatted
            according to the format specifier. Default to None.
        **kwargs: Keyword arguments forwarded to ``model.forward()``.

    Returns:
        int | str: The number of FLOPs in the model.

    Examples:
        >>> model = nn.Linear(768, 3072)
        >>> input = torch.randn(1, 128, 768)
        >>> calculate_flops(model, input)
        604372992
        >>> calculate_flops(model, input, format_spec=",")
        '604,372,992'
    """

    flops, _ = _calculate_ops(model, *args, module_ops=module_ops, excluded_modules=excluded_modules, **kwargs)
    if format_spec is not None:
        return format(flops, format_spec)
    return flops


def calculate_macs(
    model: nn.Module,
    *args,
    module_ops: Mapping[type, Callable] | None = None,
    excluded_modules: Type | Tuple[Type, ...] | None = None,
    format_spec: str | None = None,
    **kwargs,
) -> int | str:
    """
    Calculate the number of MACs (multiply-accumulate operations) in a PyTorch model.

    This performs a single forward pass with hooks attached to count operations
    per module. For HuggingFace transformer models, attention matrix operations
    (Q@K^T and attn@V) that use raw torch.matmul are automatically estimated
    from the model config.

    Args:
        model (torch.nn.Module): The model for which to calculate the MACs.
        *args: Positional arguments forwarded to ``model.forward()``.
        module_ops (Mapping[type, Callable], optional): Custom per-module-type
            hooks. Each callable has signature ``(module, input, output) -> (flops, macs)``.
            Overrides built-in hooks for the same type.
        excluded_modules (type | tuple[type, ...], optional): Module types to
            exclude from the calculation.
        format_spec (str, optional): A format specifier to format the output.
            If is None, the number of MACs is returned as an int.
            If is not None, the number of MACs is returned as a str formatted
            according to the format specifier. Default to None.
        **kwargs: Keyword arguments forwarded to ``model.forward()``.

    Returns:
        int | str: The number of MACs in the model.

    Examples:
        >>> model = nn.Linear(768, 3072)
        >>> input = torch.randn(1, 128, 768)
        >>> calculate_macs(model, input)
        301989888
        >>> calculate_macs(model, input, format_spec=",")
        '301,989,888'
    """

    _, macs = _calculate_ops(model, *args, module_ops=module_ops, excluded_modules=excluded_modules, **kwargs)
    if format_spec is not None:
        return format(macs, format_spec)
    return macs


# ---------------------------------------------------------------------------
# Per-module operation hooks
# Each returns (flops, macs)
# ---------------------------------------------------------------------------


@_register(nn.Linear)
def _linear_ops(module, input, output):
    out = _get_output_tensor(output)
    macs = out.numel() * module.in_features
    flops = 2 * macs
    if module.bias is not None:
        flops += out.numel()
    return flops, macs


@_register(nn.Conv1d, nn.Conv2d, nn.Conv3d)
def _conv_ops(module, input, output):
    out = _get_output_tensor(output)
    kernel_size = prod(module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size
    macs = out.numel() * kernel_size * (module.in_channels // module.groups)
    flops = 2 * macs
    if module.bias is not None:
        flops += out.numel()
    return flops, macs


@_register(nn.Embedding)
def _embedding_ops(module, input, output):
    return 0, 0


@_register(nn.LayerNorm)
def _layernorm_ops(module, input, output):
    inp = input[0] if isinstance(input, tuple) else input
    n = inp.numel()
    flops = 5 * n
    if module.elementwise_affine:
        flops += 2 * n
    return flops, 0


@_register(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
def _batchnorm_ops(module, input, output):
    out = _get_output_tensor(output)
    return 4 * out.numel(), 0


@_register(nn.ReLU, nn.ReLU6, nn.LeakyReLU)
def _relu_ops(module, input, output):
    out = _get_output_tensor(output)
    return out.numel(), 0


@_register(nn.GELU)
def _gelu_ops(module, input, output):
    out = _get_output_tensor(output)
    return 8 * out.numel(), 0


@_register(nn.Sigmoid)
def _sigmoid_ops(module, input, output):
    out = _get_output_tensor(output)
    return 4 * out.numel(), 0


@_register(nn.Tanh)
def _tanh_ops(module, input, output):
    out = _get_output_tensor(output)
    return 5 * out.numel(), 0


@_register(nn.SiLU)
def _silu_ops(module, input, output):
    out = _get_output_tensor(output)
    return 5 * out.numel(), 0


@_register(nn.Softmax)
def _softmax_ops(module, input, output):
    inp = input[0] if isinstance(input, tuple) else input
    return 3 * inp.numel(), 0


@_register(nn.Dropout)
def _dropout_ops(module, input, output):
    return 0, 0


@_register(nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
def _maxpool_ops(module, input, output):
    out = _get_output_tensor(output)
    kernel = module.kernel_size
    kernel_size = prod(kernel) if isinstance(kernel, tuple) else kernel
    return out.numel() * (kernel_size - 1), 0


@_register(nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
def _avgpool_ops(module, input, output):
    out = _get_output_tensor(output)
    kernel = module.kernel_size
    kernel_size = prod(kernel) if isinstance(kernel, tuple) else kernel
    return out.numel() * (kernel_size + 1), 0


@_register(nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
def _adaptive_avgpool_ops(module, input, output):
    inp = input[0] if isinstance(input, tuple) else input
    out = _get_output_tensor(output)
    return inp.numel() + out.numel(), 0


@_register(nn.MultiheadAttention)
def _mha_ops(module, input, output):
    q = input[0] if isinstance(input, tuple) else input
    batch_seq = q.numel() // module.embed_dim
    embed_dim = module.embed_dim
    # Q, K, V projections: 3 linear transforms
    proj_macs = 3 * batch_seq * embed_dim * embed_dim
    # Attention: Q@K^T and attn@V
    head_dim = embed_dim // module.num_heads
    seq_len = q.shape[0] if q.dim() == 3 else q.shape[1]
    attn_macs = 2 * module.num_heads * seq_len * seq_len * head_dim
    # Output projection
    out_proj_macs = batch_seq * embed_dim * embed_dim
    macs = proj_macs + attn_macs + out_proj_macs
    flops = 2 * macs
    return flops, macs


@_register(nn.LSTM)
def _lstm_ops(module, input, output):
    sequence = input[0] if isinstance(input, tuple) else input
    if isinstance(sequence, PackedSequence):
        total_steps = int(sequence.batch_sizes.sum().item())
    elif isinstance(sequence, torch.Tensor):
        if sequence.dim() == 2:
            total_steps = sequence.shape[0]
        elif sequence.dim() >= 3:
            total_steps = sequence.shape[0] * sequence.shape[1]
        else:
            raise ValueError(f"Unsupported LSTM input shape {tuple(sequence.shape)}")
    else:
        raise TypeError(f"Unsupported LSTM input type {type(sequence)}")

    hidden_size = module.hidden_size
    proj_size = getattr(module, "proj_size", 0)
    recurrent_size = proj_size or hidden_size
    num_directions = 2 if module.bidirectional else 1
    layer_input_size = module.input_size

    flops = 0
    macs = 0
    pointwise_flops_per_step = hidden_size * (3 * 4 + 2 * 5 + 4)
    bias_flops_per_step = 8 * hidden_size if module.bias else 0

    for _ in range(module.num_layers):
        for _ in range(num_directions):
            gate_macs = total_steps * 4 * hidden_size * (layer_input_size + recurrent_size)
            macs += gate_macs
            flops += 2 * gate_macs
            flops += total_steps * (bias_flops_per_step + pointwise_flops_per_step)

            if proj_size > 0:
                proj_macs = total_steps * hidden_size * proj_size
                macs += proj_macs
                flops += 2 * proj_macs

        layer_input_size = recurrent_size * num_directions

    return flops, macs
