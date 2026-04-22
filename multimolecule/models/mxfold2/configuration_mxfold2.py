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

from ..configuration_utils import PreTrainedConfig


class Mxfold2Config(PreTrainedConfig):
    r"""
    Configuration for [`Mxfold2Model`][multimolecule.models.Mxfold2Model].

    The defaults match the published `TrainSetAB` checkpoint from MXfold2.
    """

    model_type = "mxfold2"

    def __init__(
        self,
        vocab_size: int = 5,
        folding_model: str = "MixC",
        max_internal_length: int = 30,
        max_helix_length: int = 30,
        embed_size: int = 64,
        num_filters: list[int] | None = None,
        filter_size: list[int] | None = None,
        pool_size: list[int] | None = None,
        dilation: int = 0,
        num_lstm_layers: int = 2,
        num_lstm_units: int = 32,
        num_att: int = 8,
        num_transformer_layers: int = 0,
        num_transformer_hidden_units: int = 2048,
        num_transformer_att: int = 8,
        no_split_lr: bool = False,
        pair_join: str = "cat",
        num_paired_filters: list[int] | None = None,
        paired_filter_size: list[int] | None = None,
        num_hidden_units: list[int] | None = None,
        dropout_rate: float = 0.5,
        fc_dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.folding_model = folding_model
        self.max_internal_length = max_internal_length
        self.max_helix_length = max_helix_length
        self.embed_size = embed_size
        self.num_filters = [64] * 8 if num_filters is None else list(num_filters)
        self.filter_size = [5, 3, 5, 3, 5, 3, 5, 3] if filter_size is None else list(filter_size)
        self.pool_size = [1] if pool_size is None else list(pool_size)
        self.dilation = dilation
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.num_att = num_att
        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_hidden_units = num_transformer_hidden_units
        self.num_transformer_att = num_transformer_att
        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.num_paired_filters = [64] * 8 if num_paired_filters is None else list(num_paired_filters)
        self.paired_filter_size = [5, 3, 5, 3, 5, 3, 5, 3] if paired_filter_size is None else list(paired_filter_size)
        self.num_hidden_units = [32] if num_hidden_units is None else list(num_hidden_units)
        self.dropout_rate = dropout_rate
        self.fc_dropout_rate = fc_dropout_rate
