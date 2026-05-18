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

from ...modules import HeadConfig
from ..configuration_utils import PreTrainedConfig

MODULE_ORDER = ("acceptor_intron", "acceptor", "exon", "donor", "donor_intron")


class MmSpliceModuleConfig(FlatDict):
    r"""
    Configuration for a single MMSplice sub-module.

    MMSplice is a *modular* model: each genomic region (the acceptor and donor
    splice sites, the exon body, and the two flanking intron stubs) is scored by
    an independent small network. The five upstream sub-networks
    ([gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice))
    do **not** share an architecture, so each is described by its own
    `MmSpliceModuleConfig`.

    Two architecture families exist:

    - `conv` (``acceptor_intron``, ``exon``, ``donor_intron``): a single
      length-preserving 1D convolution, optional batch-norm, ReLU, global
      average pooling over positions, then a linear projection to a scalar.
      Length-independent.
    - `dense` (``acceptor``, ``donor``): a fixed-length splice-site network.
      The region is one-hot encoded, optionally passed through one or more
      convolutions, flattened, and projected to a scalar with a stack of
      dense + batch-norm + ReLU blocks. A final sigmoid produces a probability
      that is converted to a logit score.

    Args:
        architecture:
            Either `conv` or `dense` (see above).
        region_length:
            Fixed input length the module consumes. `0` for the length-
            independent `conv` modules (``acceptor_intron`` / ``donor_intron``).
        conv_channels:
            Output channels of the (first) convolution.
        conv_kernel_size:
            Kernel size of the (first) convolution.
        conv_activation:
            Activation applied to the (first) convolution.
        conv_batch_norm:
            Whether a batch-norm follows the (first) convolution.
        pool_mask_zeros:
            Whether global average pooling ignores all-zero input positions.
            Upstream `exon` uses masked pooling to ignore `N` padding.
        pointwise_channels:
            Output channels of the `dense`-family `1x1` convolution. `0`
            disables it. Followed by a batch-norm.
        hidden_sizes:
            Output sizes of the dense blocks of a `dense`-family head.
        flatten_dropout:
            Whether a dropout is applied right after the flatten (before the
            dense blocks). Upstream `acceptor` uses this; `donor` instead applies
            dropout inside each dense block.
        dropout:
            Dropout probability used by `dense`-family heads.
        batch_norm_eps:
            Epsilon of every batch-norm (upstream Keras default `1e-3`).
    """

    architecture: str = "conv"
    region_length: int = 0
    conv_channels: int = 0
    conv_kernel_size: int = 0
    conv_activation: str = "linear"
    conv_batch_norm: bool = False
    pool_mask_zeros: bool = False
    pointwise_channels: int = 0
    hidden_sizes: list = []  # noqa: RUF012
    flatten_dropout: bool = False
    dropout: float = 0.2
    batch_norm_eps: float = 1e-3


class MmSpliceConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`MmSpliceModel`][multimolecule.models.MmSpliceModel]. It is used to instantiate a MMSplice model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a
    configuration to that of the MMSplice
    [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    MMSplice (Cheng et al. 2019, *Genome Biology*) is a *modular* model. Five
    independent sub-networks (``acceptor_intron``, ``acceptor``, ``exon``,
    ``donor``, ``donor_intron``) each score one region of the exon-with-flanking-
    introns sequence. The five scalar scores form the module score vector. For
    variant-effect estimation the model is run on the reference and the
    alternative sequence and the per-module score deltas are combined by the
    fixed upstream linear model into a delta-logit-PSI splicing-effect score.

    The default module configurations replicate the upstream pretrained weights
    exactly (see [`MmSpliceModuleConfig`]).

    Args:
        vocab_size:
            Vocabulary size of the MMSplice model. Defines the number of feature
            channels derived from the one-hot encoded `input_ids`. MMSplice uses
            four `A/C/G/U` channels; `N`, padding, special, and unknown tokens are
            encoded as all-zero columns by the embedding layer.
            Defaults to 4 (the `ACGU` nucleobase alphabet).
        modules:
            Per sub-module architecture configuration. A mapping from module name
            to a [`MmSpliceModuleConfig`]. The default defines the five canonical
            MMSplice modules with their upstream architectures.
        modules_config:
            Alias used when loading serialized configs. Prefer `modules` when
            constructing configs directly.
        acceptor_intron_cut:
            Number of bp removed from the 3' end of the acceptor intron (the part
            considered the acceptor site).
        donor_intron_cut:
            Number of bp removed from the 5' end of the donor intron (the part
            considered the donor site).
        acceptor_intron_length:
            Intron length consumed by the acceptor splice-site module.
        acceptor_exon_length:
            Exon flank length consumed by the acceptor splice-site module.
        donor_exon_length:
            Exon flank length consumed by the donor splice-site module.
        donor_intron_length:
            Intron length consumed by the donor splice-site module.
        num_labels:
            Number of sequence-prediction labels. MMSplice emits one scalar
            delta-logit-PSI score, so this must be 1.
        head:
            Loss configuration for [`MmSpliceForSequencePrediction`]. The
            upstream variant-effect combiner emits one scalar delta-logit-PSI
            score, so the default head config has `num_labels=1`.

    Examples:
        >>> from multimolecule import MmSpliceConfig, MmSpliceModel
        >>> # Initializing a MMSplice multimolecule/mmsplice style configuration
        >>> configuration = MmSpliceConfig()
        >>> # Initializing a model (with random weights) from the multimolecule/mmsplice style configuration
        >>> model = MmSpliceModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "mmsplice"

    def __init__(
        self,
        vocab_size: int = 4,
        modules: dict | None = None,
        modules_config: dict | None = None,
        acceptor_intron_cut: int = 6,
        donor_intron_cut: int = 6,
        acceptor_intron_length: int = 50,
        acceptor_exon_length: int = 3,
        donor_exon_length: int = 5,
        donor_intron_length: int = 13,
        num_labels: int = 1,
        head: HeadConfig | None = None,
        problem_type: str | None = "regression",
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        if num_labels != 1:
            raise ValueError(f"MMSplice emits one delta-logit-PSI score; `num_labels` must be 1, got {num_labels}")
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        if modules is None:
            modules = modules_config
        if modules is None:
            acceptor_region = acceptor_intron_length + acceptor_exon_length
            donor_region = donor_exon_length + donor_intron_length
            modules = {
                "acceptor_intron": MmSpliceModuleConfig(
                    architecture="conv",
                    conv_channels=256,
                    conv_kernel_size=13,
                    conv_activation="relu",
                ),
                "acceptor": MmSpliceModuleConfig(
                    architecture="dense",
                    region_length=acceptor_region,
                    conv_channels=32,
                    conv_kernel_size=15,
                    conv_activation="relu",
                    conv_batch_norm=True,
                    pointwise_channels=32,
                    hidden_sizes=[],
                    flatten_dropout=True,
                ),
                "exon": MmSpliceModuleConfig(
                    architecture="conv",
                    conv_channels=128,
                    conv_kernel_size=11,
                    conv_activation="relu",
                    conv_batch_norm=True,
                    pool_mask_zeros=True,
                ),
                "donor": MmSpliceModuleConfig(
                    architecture="dense",
                    region_length=donor_region,
                    hidden_sizes=[128, 64],
                ),
                "donor_intron": MmSpliceModuleConfig(
                    architecture="conv",
                    conv_channels=256,
                    conv_kernel_size=13,
                    conv_activation="relu",
                ),
            }
        self.modules_config = {
            name: cfg if isinstance(cfg, MmSpliceModuleConfig) else MmSpliceModuleConfig(**cfg)
            for name, cfg in modules.items()
        }
        self.acceptor_intron_cut = acceptor_intron_cut
        self.donor_intron_cut = donor_intron_cut
        self.acceptor_intron_length = acceptor_intron_length
        self.acceptor_exon_length = acceptor_exon_length
        self.donor_exon_length = donor_exon_length
        self.donor_intron_length = donor_intron_length
        # The backbone hidden representation is the per-module score vector.
        self.hidden_size = len(MODULE_ORDER)
        self.problem_type = problem_type
        if head is None:
            head = HeadConfig(num_labels=num_labels, hidden_size=1, problem_type=problem_type)
        elif not isinstance(head, HeadConfig):
            head = HeadConfig(**head)
        self.head = head

        missing = sorted(set(MODULE_ORDER) - set(self.modules_config))
        if missing:
            raise ValueError(f"Missing required MMSplice modules in modules config: {missing}.")
        unexpected = sorted(set(self.modules_config) - set(MODULE_ORDER))
        if unexpected:
            raise ValueError(f"Unexpected MMSplice modules in modules config: {unexpected}.")
        if min(acceptor_intron_length, donor_intron_length) <= 0:
            raise ValueError("Intron region lengths must be positive.")
        if min(acceptor_exon_length, donor_exon_length) < 0:
            raise ValueError("Exon flank lengths must be non-negative.")
