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

from ..configuration_utils import PreTrainedConfig


class SpotRnaNetworkConfig(FlatDict):
    r"""
    Configuration for a single SPOT-RNA network in the ensemble.

    Args:
        num_conv_blocks:
            Number of convolutional blocks (N_A in the paper).
        num_blstm_blocks:
            Number of 2D bidirectional LSTM blocks. Set to 0 to disable.
        num_fc_blocks:
            Number of fully connected blocks. Set to 0 to disable.
        conv_channels:
            Number of channels in the convolutional blocks.
        blstm_hidden_size:
            Hidden size per direction in the 2D-BLSTM. Ignored if num_blstm_blocks is 0.
        fc_hidden_size:
            Hidden size of the fully connected blocks. Ignored if num_fc_blocks is 0.
        hidden_act:
            Activation used in the convolutional residual blocks.
        fc_act:
            Optional activation used in the fully connected blocks. Falls back to `hidden_act` when unset.
        output_act:
            Activation applied before the final normalization stage.
        use_dilation:
            Whether to use dilated convolutions.
        dilation_cycle:
            The cycle length for the dilation factor.
    """

    num_conv_blocks: int = 20
    num_blstm_blocks: int = 0
    num_fc_blocks: int = 1
    conv_channels: int = 64
    blstm_hidden_size: int = 200
    fc_hidden_size: int = 512
    hidden_act: str = "relu"
    fc_act: str | None = None
    output_act: str = "relu"
    use_dilation: bool = False
    dilation_cycle: int = 5


class SpotRnaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`SpotRnaModel`][multimolecule.models.SpotRnaModel]. It is used to instantiate a SPOT-RNA model according to
    the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the SPOT-RNA
    [jaswindersingh2/SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA) architecture.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Token vocabulary size of the SPOT-RNA model. Defaults to 5 for the `A/C/G/U/N` tokenizer vocabulary.
        networks:
            List of configurations for each network in the ensemble. Each entry is a
            [`SpotRnaNetworkConfig`] object. If None, defaults to the 5 networks from Supplementary Table 9.
        input_channels:
            Number of input feature channels after outer concatenation. Defaults to 8 for the original `A/C/G/U`
            pairwise representation.
        hidden_act:
            The non-linear activation function in the convolutional and fully connected blocks.
        conv_dropout:
            Dropout rate in the convolutional blocks.
        fc_dropout:
            Dropout rate in the fully connected blocks.
        threshold:
            Probability threshold for predicting base pairs during post-processing.

    Examples:
        >>> from multimolecule import SpotRnaConfig, SpotRnaModel
        >>> configuration = SpotRnaConfig()
        >>> model = SpotRnaModel(configuration)
        >>> configuration = model.config
    """

    model_type = "spotrna"

    def __init__(
        self,
        vocab_size: int = 5,
        networks: list[SpotRnaNetworkConfig] | None = None,
        input_channels: int = 8,
        hidden_act: str = "relu",
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.5,
        threshold: float = 0.335,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if input_channels % 2 != 0:
            raise ValueError(f"SpotRnaConfig.input_channels must be even, but got {input_channels}.")
        self.vocab_size = vocab_size
        if networks is None:
            networks = [
                SpotRnaNetworkConfig(num_conv_blocks=16, conv_channels=48, num_fc_blocks=2, fc_hidden_size=512),
                SpotRnaNetworkConfig(num_conv_blocks=20, conv_channels=64, num_fc_blocks=1, fc_hidden_size=512),
                SpotRnaNetworkConfig(num_conv_blocks=30, conv_channels=64, num_fc_blocks=1, fc_hidden_size=512),
                SpotRnaNetworkConfig(
                    num_conv_blocks=30,
                    conv_channels=64,
                    num_blstm_blocks=1,
                    blstm_hidden_size=200,
                    num_fc_blocks=0,
                    hidden_act="elu",
                ),
                SpotRnaNetworkConfig(
                    num_conv_blocks=30,
                    conv_channels=64,
                    num_fc_blocks=1,
                    fc_hidden_size=512,
                    fc_act="elu",
                    output_act="elu",
                    use_dilation=True,
                    dilation_cycle=5,
                ),
            ]
        self.networks = networks
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.threshold = threshold
