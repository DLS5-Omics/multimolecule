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

from ..configuration_utils import HeadConfig, PreTrainedConfig

# Window lengths defined by Yeo & Burge (2004): the 5' splice-site scorer consumes
# 3 exonic + 6 intronic = 9 nucleotides, the 3' splice-site scorer consumes 23 nucleotides.
SCORE5_WINDOW = 9
SCORE3_WINDOW = 23
WINDOW_FOR_MODE = {"score5": SCORE5_WINDOW, "score3": SCORE3_WINDOW}


class MaxEntScanConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`MaxEntScanModel`][multimolecule.models.MaxEntScanModel]. It is used to instantiate a MaxEntScan scorer according
    to the specified arguments, defining the model behavior. Instantiating a configuration with the defaults will yield
    a configuration equivalent to the 5' splice-site scorer (`score5`) of the original MaxEntScan tool.

    MaxEntScan is a maximum-entropy model and has no trainable weights. The score tables are fixed maximum-entropy
    probability tables published with the original tool and are registered as buffers on the model.

    Configuration objects inherit from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig] and can be used to
    control the model outputs. Read the documentation from [`PreTrainedConfig`][multimolecule.models.PreTrainedConfig]
    for more information.

    Args:
        vocab_size:
            Vocabulary size of the MaxEntScan model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`MaxEntScanModel`].
            Defaults to 5 (the streamline RNA alphabet `ACGUN`).
        mode:
            Which splice-site scorer to use. `"score5"` scores 5' (donor) splice sites, `"score3"` scores 3' (acceptor)
            splice sites.
        window:
            The fixed length of the input window. Must match `mode`: 9 for `score5`, 23 for `score3`. If `None`, it is
            derived from `mode`.
        num_labels:
            Number of output labels. MaxEntScan emits a single maximum-entropy score, so this must be 1.

    Examples:
        >>> from multimolecule import MaxEntScanConfig, MaxEntScanModel
        >>> # Initializing a MaxEntScan multimolecule/maxentscan-score5 style configuration
        >>> configuration = MaxEntScanConfig()
        >>> # Initializing a model (with random buffers) from the multimolecule/maxentscan-score5 style configuration
        >>> model = MaxEntScanModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "maxentscan"

    def __init__(
        self,
        vocab_size: int = 5,
        mode: str = "score5",
        window: int | None = None,
        hidden_size: int = 1,
        head: HeadConfig | None = None,
        num_labels: int = 1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, pad_token_id=pad_token_id, **kwargs)
        if mode not in WINDOW_FOR_MODE:
            raise ValueError(f"`mode` must be one of {sorted(WINDOW_FOR_MODE)}, got {mode!r}")
        expected_window = WINDOW_FOR_MODE[mode]
        if window is None:
            window = expected_window
        if window != expected_window:
            raise ValueError(f"`window` ({window}) does not match `mode` ({mode!r}); expected window {expected_window}")
        if num_labels != 1:
            raise ValueError(f"MaxEntScan emits a single score; `num_labels` must be 1, got {num_labels}")
        if hidden_size != 1:
            raise ValueError(f"MaxEntScan emits a single scalar feature; `hidden_size` must be 1, got {hidden_size}")
        self.bos_token_id = bos_token_id  # type: ignore[assignment]
        self.eos_token_id = eos_token_id  # type: ignore[assignment]
        self.vocab_size = vocab_size
        self.mode = mode
        self.window = window
        # The maximum-entropy score is a single scalar feature; the downstream regression head projects from it.
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.problem_type = "regression"
        self.head = HeadConfig(head) if head is not None else HeadConfig(num_labels=1, problem_type="regression")
