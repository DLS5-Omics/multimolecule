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

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from chanfig import FlatDict
from danling import NestedTensor
from torch import Tensor, nn

from .backbones import BACKBONES
from .heads import HEADS
from .necks import NECKS
from .registry import MODELS

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

_DECAY_PATTERNS: list[str] = ["weight", "conv", "fc"]
_NO_DECAY_PATTERNS: list[str] = ["bias", "bn", "norm"]

_MONO_HEAD_TYPES: frozenset[str] = frozenset({"sequence", "token", "contact"})


class ModelBase(nn.Module, ABC):
    r"""
    Abstract base for all multimolecule models.

    Defines the contract that the runner expects: `forward` returns a per-task mapping (one
    [`HeadOutput`][multimolecule.modules.HeadOutput] per task), and `trainable_parameters` produces optimizer
    parameter groups with separate learning-rate scaling for the pretrained backbone.

    Subclass to expose new model topologies through [`MODELS`][multimolecule.modules.MODELS]; the runner
    discriminates models with `isinstance(model, ModelBase)` rather than against any concrete subclass.
    """

    decay_patterns: list[str] = _DECAY_PATTERNS
    no_decay_patterns: list[str] = _NO_DECAY_PATTERNS

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        r"""
        Run the model and return a per-task mapping of outputs.

        Returns:
            A mapping from task name to a `HeadOutput`-like object exposing `.loss` and `.logits`.
        """

    @abstractmethod
    def trainable_parameters(
        self,
        lr: float,
        weight_decay: float,
        pretrained_ratio: float = 1e-2,
    ) -> list[dict]:
        r"""
        Build parameter groups for the optimizer.

        Args:
            lr: Base learning rate for newly initialized parameters.
            weight_decay: Base weight decay for newly initialized parameters.
            pretrained_ratio: Multiplier applied to the backbone learning rate and weight decay.

        Returns:
            A list of parameter group dicts compatible with [`torch.optim.Optimizer`][torch.optim.Optimizer].
        """


def _make_param_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
    lr_ratio: float = 1.0,
    decay_patterns: list[str] | None = None,
    no_decay_patterns: list[str] | None = None,
) -> list[dict]:
    decay_patterns = decay_patterns if decay_patterns is not None else _DECAY_PATTERNS
    no_decay_patterns = no_decay_patterns if no_decay_patterns is not None else _NO_DECAY_PATTERNS
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if any(w in name for w in decay_patterns):
            decay_params.append(param)
        elif any(b in name for b in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups: list[dict] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay * lr_ratio, "lr": lr * lr_ratio})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": lr * lr_ratio})
    return param_groups


@MODELS.register("poly")
class PolyModel(ModelBase):
    r"""
    Compose a backbone, optional neck, and one head per task into a single trainable model.

    Use when the task graph involves multiple labels, extra non-sequence features, or a neck transform.
    For the single-task / single-input case, prefer [`MonoModel`][multimolecule.MonoModel].
    """

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
        decay_patterns: list[str] | None = None,
        no_decay_patterns: list[str] | None = None,
    ) -> list[dict]:
        decay_patterns = decay_patterns if decay_patterns is not None else self.decay_patterns
        no_decay_patterns = no_decay_patterns if no_decay_patterns is not None else self.no_decay_patterns
        param_groups: list[dict] = []
        param_groups.extend(
            _make_param_groups(
                self.heads.named_parameters(),
                lr,
                weight_decay,
                decay_patterns=decay_patterns,
                no_decay_patterns=no_decay_patterns,
            )
        )
        if isinstance(self.backbone, nn.Module):
            param_groups.extend(
                _make_param_groups(
                    self.backbone.named_parameters(),
                    lr,
                    weight_decay,
                    lr_ratio=pretrained_ratio,
                    decay_patterns=decay_patterns,
                    no_decay_patterns=no_decay_patterns,
                )
            )
        if isinstance(self.neck, nn.Module):
            param_groups.extend(
                _make_param_groups(
                    self.neck.named_parameters(),
                    lr,
                    weight_decay,
                    decay_patterns=decay_patterns,
                    no_decay_patterns=no_decay_patterns,
                )
            )
        return param_groups


@MODELS.register("mono")
class MonoModel(ModelBase):
    r"""
    Single-task wrapper around a multimolecule `AutoModelFor*` prediction model.

    Use when the task graph is a single sequence-, token-, or contact-level prediction with no neck and a
    sequence-only backbone — i.e. when the underlying HF prediction model already does what `PolyModel`
    would assemble. The wrapper makes the HF model invisible at the `state_dict` layer, so checkpoints saved
    here are byte-identical to checkpoints from the bare `AutoModelFor*` and vice versa.

    Args:
        backbone: Backbone configuration. Must contain a single `sequence` sub-dict whose `name` resolves to
            a Hugging Face model identifier.
        heads: Per-task head configuration; must contain exactly one entry whose `type` is `"sequence"`,
            `"token"`, or `"contact"`.
        neck: Must be unset; rejected if provided.
        probing: When `True`, freeze the encoder (`base_model`) parameters so only the head trains.
    """

    def __init__(
        self,
        backbone: dict,
        heads: dict,
        neck: dict | None = None,
        max_length: int = 1024,  # noqa: ARG002 — accepted for API parity with PolyModel
        truncation: bool = False,  # noqa: ARG002
        probing: bool = False,
        **_: Any,
    ):
        super().__init__()
        if neck:
            raise ValueError("MonoModel does not support a neck; use PolyModel.")
        if len(heads) != 1:
            raise ValueError(f"MonoModel supports exactly one head, got {len(heads)}.")
        if set(backbone.keys()) != {"sequence"}:
            raise ValueError(f"MonoModel only supports a sequence-only backbone, got keys {sorted(backbone.keys())}.")

        self.task = next(iter(heads))
        head_config = next(iter(heads.values()))
        sequence_config = backbone["sequence"]
        pretrained_name = sequence_config.get("name")
        if not pretrained_name:
            raise ValueError("MonoModel requires `backbone.sequence.name` to be set.")

        auto_classes = _auto_classes_for(head_config.get("type"))
        from_pretrained_kwargs: dict[str, Any] = {}
        if head_config.get("num_labels") is not None:
            from_pretrained_kwargs["num_labels"] = head_config["num_labels"]
        # `problem_type` is intentionally not forwarded: HF's `PretrainedConfig.problem_type` is a strict
        # dataclass field that only accepts {"regression", "single_label_classification",
        # "multi_label_classification"}, while multimolecule's `TaskType` uses {"binary", "multiclass",
        # "multilabel", "regression"}. The multimolecule head reads its own `HeadConfig.problem_type` and
        # HF's stock heads auto-detect from labels at training time, so leaving the HF-side field unset is
        # safe for both backends.

        self.module = _build_module(
            auto_classes,
            pretrained_name,
            use_pretrained=sequence_config.get("use_pretrained", True),
            kwargs=from_pretrained_kwargs,
        )

        if probing:
            for param in self.module.base_model.parameters():
                param.requires_grad = False

    def forward(self, **inputs: Any) -> Mapping[str, Any]:
        labels = inputs.pop(self.task, None)
        if "input_ids" not in inputs and "sequence" in inputs:
            inputs["input_ids"] = inputs.pop("sequence")
        for unused in ("discrete", "continuous", "dataset"):
            inputs.pop(unused, None)
        if labels is not None:
            output = self.module(**inputs, labels=labels)
        else:
            output = self.module(**inputs)
        return FlatDict({self.task: output})

    def trainable_parameters(
        self,
        lr: float,
        weight_decay: float,
        pretrained_ratio: float = 1e-2,
        decay_patterns: list[str] | None = None,
        no_decay_patterns: list[str] | None = None,
    ) -> list[dict]:
        decay_patterns = decay_patterns if decay_patterns is not None else self.decay_patterns
        no_decay_patterns = no_decay_patterns if no_decay_patterns is not None else self.no_decay_patterns
        backbone_param_ids = {id(p) for p in self.module.base_model.parameters()}
        backbone_named = ((n, p) for n, p in self.module.named_parameters() if id(p) in backbone_param_ids)
        head_named = ((n, p) for n, p in self.module.named_parameters() if id(p) not in backbone_param_ids)
        return _make_param_groups(
            head_named, lr, weight_decay, decay_patterns=decay_patterns, no_decay_patterns=no_decay_patterns
        ) + _make_param_groups(
            backbone_named,
            lr,
            weight_decay,
            lr_ratio=pretrained_ratio,
            decay_patterns=decay_patterns,
            no_decay_patterns=no_decay_patterns,
        )

    # The wrapper is intentionally invisible at the state_dict layer: checkpoints round-trip with the bare
    # HF model.
    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False):
        return self.module.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.module.load_state_dict(state_dict, strict=strict, assign=assign)


def _auto_classes_for(level: str | None) -> list[type[_BaseAutoModelClass]]:
    """Resolve the candidate `AutoModelFor*` classes to try for a given head type/task level.

    Returns the multimolecule class first (preferred — matches multimolecule head semantics) followed by
    the HuggingFace stock equivalent when one exists, so backbones registered only with `transformers` (e.g.
    `bert-base-uncased`) still load through `MonoModel`.
    """
    if not level:
        raise ValueError("MonoModel requires the head config to specify a `type`.")
    # Tolerate richer head types like "contact.logits.linear" by taking the first component.
    head_kind = level.split(".", 1)[0]

    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

    from multimolecule.models.modeling_auto import (
        AutoModelForContactPrediction,
        AutoModelForRnaSecondaryStructurePrediction,
        AutoModelForSequencePrediction,
        AutoModelForTokenPrediction,
    )

    mapping: dict[str, list[type]] = {
        "sequence": [AutoModelForSequencePrediction, AutoModelForSequenceClassification],
        "token": [AutoModelForTokenPrediction, AutoModelForTokenClassification],
        "contact": [AutoModelForContactPrediction, AutoModelForRnaSecondaryStructurePrediction],
    }
    if head_kind not in mapping:
        raise ValueError(
            f"MonoModel does not support head type {level!r}; expected one of "
            f"{sorted(_MONO_HEAD_TYPES)} or use PolyModel."
        )
    return mapping[head_kind]


def _build_module(
    auto_classes: list[type[_BaseAutoModelClass]],
    pretrained_name: str,
    use_pretrained: bool,
    kwargs: dict[str, Any],
) -> nn.Module:
    """Try each `AutoModelFor*` class in order; return the first one that loads `pretrained_name`."""
    last_err: Exception | None = None
    for cls in auto_classes:
        try:
            if use_pretrained:
                return cls.from_pretrained(pretrained_name, **kwargs)
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(pretrained_name, **kwargs)
            return cls.from_config(cfg)
        except (KeyError, ValueError) as err:
            # KeyError: config type not in this AutoModel's mapping. ValueError: HF wraps the same condition.
            last_err = err
            continue
    raise ValueError(
        f"MonoModel could not load {pretrained_name!r} with any of "
        f"{[c.__name__ for c in auto_classes]}; last error: {last_err}"
    )
