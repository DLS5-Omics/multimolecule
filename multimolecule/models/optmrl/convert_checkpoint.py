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

import hashlib
import os
import urllib.request
from collections import OrderedDict

import chanfig
import torch

from multimolecule.models import OptMrlConfig as Config
from multimolecule.models import OptMrlForSequencePrediction as Model
from multimolecule.models.conversion_utils import ConvertConfig as ConvertConfig_
from multimolecule.models.conversion_utils import convert_one_hot_embeddings, load_checkpoint, save_checkpoint
from multimolecule.tokenisers.rna.utils import convert_word_embeddings, get_alphabet, get_tokenizer_config

torch.manual_seed(1016)


# Upstream OptMRL one-hot order is ("A", "C", "G", "T") (kipoiseq.utils.DNA used by the
# OptMRL Kipoi dataloader). The 5'UTR alphabet is RNA so "T" is treated as "U".
ORIGINAL_VOCAB_LIST = ["A", "C", "G", "U"]

# Published OptMRL artifact hosted at https://kipoi.org/models/OptMRL/ and archived on Zenodo
# (https://zenodo.org/records/11258762). The MD5 values come from the OptMRL ``model.yaml`` in
# https://github.com/kipoi/models/tree/master/OptMRL.
OPTMRL_ARCH_URL = "https://zenodo.org/records/11258762/files/OptMRL-arch.json"
OPTMRL_ARCH_MD5 = "ecd4879e2d9810f220732e59015d4e8f"
OPTMRL_WEIGHTS_URL = "https://zenodo.org/records/11258762/files/OptMRL-weights.h5"
OPTMRL_WEIGHTS_MD5 = "7ee6242c16a5f718179a084cacc4caea"

# Maps the published Keras layer name to the MultiMolecule parameter prefix.
NAME_MAPPING = {
    "conv1d_1": "model.encoder.conv_layers.0",
    "conv1d_2": "model.encoder.conv_layers.1",
    "conv1d_3": "model.encoder.conv_layers.2",
    "dense_1": "model.encoder.dense",
    "dense_2": "sequence_head.decoder",
}


def _download(url: str, expected_md5: str, cache_path: str | None = None) -> bytes:
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            raw = file.read()
    else:
        print(f"Downloading {url}")
        with urllib.request.urlopen(url) as response:  # noqa: S310
            raw = response.read()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as file:
                file.write(raw)
    digest = hashlib.md5(raw).hexdigest()
    if digest != expected_md5:
        raise ValueError(
            f"Downloaded artifact at {url} has MD5 {digest} but the published value is {expected_md5}; "
            "refusing to convert untrusted weights."
        )
    return raw


def _convert_checkpoint(weights_path: str) -> OrderedDict[str, torch.Tensor]:
    """Read the published OptMRL Keras (HDF5) weights and convert to MultiMolecule torch layout.

    Keras stores Conv1D kernels as `(kernel, in_channels, out_channels)` and Dense kernels as
    `(in, out)`; both are transformed to the torch layout here.
    """
    try:
        import h5py  # noqa: PLC0415  conversion-only dependency
    except ImportError as e:
        raise ImportError(
            "h5py is required to convert an OptMRL checkpoint. " "Install it with: pip install h5py"
        ) from e
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    with h5py.File(weights_path, "r") as f:
        for keras_name, torch_prefix in NAME_MAPPING.items():
            group = f[keras_name][keras_name]
            kernel = torch.from_numpy(group["kernel:0"][()]).float()
            bias = torch.from_numpy(group["bias:0"][()]).float()
            if keras_name.startswith("conv1d"):
                # Keras Conv1D kernel: (kernel, in_channels, out_channels) ->
                # torch Conv1d (out_channels, in_channels, kernel).
                kernel = kernel.permute(2, 1, 0).contiguous()
            else:
                # Keras Dense kernel: (in, out) -> torch Linear (out, in).
                kernel = kernel.t().contiguous()
            state_dict[f"{torch_prefix}.weight"] = kernel
            state_dict[f"{torch_prefix}.bias"] = bias
    return state_dict


def convert_checkpoint(convert_config):
    print(f"Converting OptMRL checkpoint at {convert_config.checkpoint_path or '<download>'}")
    config = Config()
    model = Model(config)

    alphabet = get_alphabet("streamline", prepend_tokens=[])
    tokenizer_config = chanfig.NestedDict(get_tokenizer_config())
    tokenizer_config["alphabet"] = alphabet
    tokenizer_config["model_max_length"] = config.sequence_length
    tokenizer_config["unk_token"] = tokenizer_config["pad_token"] = "N"

    new_vocab_list = list(alphabet.vocabulary)

    weights_path = convert_config.checkpoint_path
    if not weights_path or not os.path.isfile(weights_path):
        cache_dir = os.path.join(os.path.dirname(__file__), "saved_models")
        os.makedirs(cache_dir, exist_ok=True)
        weights_path = os.path.join(cache_dir, "OptMRL-weights.h5")
        _download(OPTMRL_WEIGHTS_URL, OPTMRL_WEIGHTS_MD5, cache_path=weights_path)
        # The arch JSON is fetched only to verify provenance; the architecture itself is
        # described by ``OptMrlConfig`` and the layout transforms in ``_convert_checkpoint``.
        arch_path = os.path.join(cache_dir, "OptMRL-arch.json")
        _download(OPTMRL_ARCH_URL, OPTMRL_ARCH_MD5, cache_path=arch_path)
    else:
        with open(weights_path, "rb") as file:
            digest = hashlib.md5(file.read()).hexdigest()
        if digest != OPTMRL_WEIGHTS_MD5:
            raise ValueError(
                f"Local OptMRL weights at {weights_path} have MD5 {digest} but the published value is "
                f"{OPTMRL_WEIGHTS_MD5}; refusing to convert untrusted weights."
            )

    state_dict = _convert_checkpoint(weights_path)
    key = "model.encoder.conv_layers.0.weight"
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError(f"Converted state dict is missing {key!r}.")
    # Upstream encodes unknown bases as an all-zero vector; make that channel explicit
    # before using tokenizer-aware conversion so ``N`` is copied as zeros, not averaged.
    weight = torch.cat([weight, torch.zeros_like(weight[:, :1, :])], dim=1)
    state_dict[key] = convert_one_hot_embeddings(
        weight,
        old_vocab=ORIGINAL_VOCAB_LIST + ["N"],
        new_vocab=new_vocab_list,
        convert_word_embeddings=convert_word_embeddings,
    )

    model_state = model.state_dict()
    for key in model_state:
        if key not in state_dict and not key.startswith(("model.embeddings.",)):
            raise KeyError(f"Converted state dict is missing the learned parameter {key!r}.")

    load_checkpoint(model, state_dict)
    save_checkpoint(convert_config, model, tokenizer_config=tokenizer_config)
    print(f"Checkpoint saved to {convert_config.output_path}")


class ConvertConfig(ConvertConfig_):
    root: str = os.path.dirname(__file__)
    output_path: str = Config.model_type
    checkpoint_path: str = ""


if __name__ == "__main__":
    config = ConvertConfig()
    config.parse()  # type: ignore[attr-defined]
    convert_checkpoint(config)
