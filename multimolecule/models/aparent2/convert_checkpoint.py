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

import os
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import Aparent2Config as Config
from multimolecule.models import Aparent2ForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.dna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream APARENT2 one-hot encoding is ordered as ["A", "C", "G", "T"] (see
# examples/aparent2_score_variants.ipynb `one_hot_encode`). The upstream encoder represents
# "N" as 0.25 in each nucleotide channel, so the converted MultiMolecule "N" projection
# channel is the mean of the four upstream channels.
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "T"]


def convert_checkpoint(convert_config):
    print(f"Converting APARENT2 checkpoint at {convert_config.checkpoint_path}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    state_dict = _convert_checkpoint(convert_config.checkpoint_path, config)
    key = "encoder.projection.weight"
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError("encoder.projection.weight missing; cannot apply vocabulary channel conversion")
    state_dict[key] = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST,
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    reference_state = model.model.state_dict()
    for key, value in reference_state.items():
        if key.endswith("num_batches_tracked") and key not in state_dict:
            state_dict[key] = value
    state_dict = OrderedDict((f"model.{key}", value) for key, value in state_dict.items())

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


def _h5_weights(path: str):
    """Read raw Keras layer weights from the legacy .h5 file.

    Keras 3 refuses to deserialize the upstream graph (it contains Python-lambda `Lambda` layers,
    blocked for security). h5py reads the saved weight tensors directly, which is the proven pattern
    for legacy Keras ports in this repository.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "No upstream APARENT2 checkpoint found. Download the APARENT2 saved models from the official "
            "repository and pass `aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode.h5` "
            "via `--checkpoint_path`."
        )

    import h5py

    f = h5py.File(path, "r")
    model_weights = f["model_weights"]

    def get(layer: str, name: str):
        return torch.from_numpy(model_weights[layer][f"{layer}/{name}:0"][()].copy())

    return f, get


def _convert_conv_kernel(kernel: torch.Tensor) -> torch.Tensor:
    """Keras Conv2D kernel `(1, K, Cin, Cout)` -> torch Conv1d weight `(Cout, Cin, K)`."""
    kernel = kernel.squeeze(0)  # (K, Cin, Cout)
    return kernel.permute(2, 1, 0).contiguous()


def _convert_checkpoint(path: str, config: Config) -> OrderedDict:
    f, get = _h5_weights(path)
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    # Input projection (first conv that sees vocabulary channels).
    state_dict["encoder.projection.weight"] = _convert_conv_kernel(get("aparent_conv_0", "kernel"))
    state_dict["encoder.projection.bias"] = get("aparent_conv_0", "bias")

    for group_idx, _ in enumerate(config.dilations):
        prefix = f"encoder.groups.{group_idx}"
        state_dict[f"{prefix}.skip.weight"] = _convert_conv_kernel(get(f"aparent_skip_conv_{group_idx}", "kernel"))
        state_dict[f"{prefix}.skip.bias"] = get(f"aparent_skip_conv_{group_idx}", "bias")
        for block_idx in range(config.num_blocks):
            up = f"aparent_resblock_{group_idx}_{block_idx}"
            bp = f"{prefix}.blocks.{block_idx}"
            for norm_idx in (0, 1):
                src = f"{up}_batch_norm_{norm_idx}"
                dst = f"{bp}.norm{norm_idx + 1}"
                state_dict[f"{dst}.weight"] = get(src, "gamma")
                state_dict[f"{dst}.bias"] = get(src, "beta")
                state_dict[f"{dst}.running_mean"] = get(src, "moving_mean")
                state_dict[f"{dst}.running_var"] = get(src, "moving_variance")
            for conv_idx in (0, 1):
                src = f"{up}_conv_{conv_idx}"
                dst = f"{bp}.conv{conv_idx + 1}"
                state_dict[f"{dst}.weight"] = _convert_conv_kernel(get(src, "kernel"))
                state_dict[f"{dst}.bias"] = get(src, "bias")

    # Upstream `last_block_conv` is applied to the final residual-network output, then `skip_add`
    # sums it with every per-group skip conv. MultiMolecule keeps it as `encoder.conv`.
    state_dict["encoder.conv.weight"] = _convert_conv_kernel(get("aparent_last_block_conv", "kernel"))
    state_dict["encoder.conv.bias"] = get("aparent_last_block_conv", "bias")

    # Final 1x1 cleavage projection.
    state_dict["prediction.weight"] = _convert_conv_kernel(get("aparent_final_conv", "kernel"))
    state_dict["prediction.bias"] = get("aparent_final_conv", "bias")

    # LocallyConnected2D library bias: kernel `(num_positions, num_libraries, 1)`,
    # bias `(1, num_positions, 1)`. There is NO flatten->dense layer anywhere in APARENT2,
    # so the Keras row-major Flatten vs torch (C, L) reconciliation does not apply here.
    lib_kernel = get("aparent_lib_conv", "kernel")  # (206, 13, 1)
    state_dict["library_bias.weight"] = lib_kernel[:, :, 0].contiguous()  # (206, 13)
    lib_bias = get("aparent_lib_conv", "bias")  # (1, 206, 1)
    state_dict["library_bias.bias"] = lib_bias.reshape(-1).contiguous()  # (206,)

    f.close()
    return state_dict


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
