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

import os
from typing import List, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Base tokenizer.

    Examples:
        >>> from multimolecule.tokenisers import Tokenizer
        >>> tokenizer = Tokenizer(["A", "C", "G", "T"])
        >>> tokenizer('ACGTN')["input_ids"]
        [4, 0, 1, 2, 3, 6, 5]
        >>> tokenizer('acgtn')["input_ids"]
        [4, 0, 1, 2, 3, 6, 5]
        >>> tokenizer('ac<mask>gt')["input_ids"]
        [4, 0, 1, 8, 2, 3, 5]
        >>> len(tokenizer)
        9
        >>> tokenizer.all_tokens
        ['A', 'C', 'G', 'T', '<cls>', '<eos>', '<unk>', '<pad>', '<mask>']
        >>> tokenizer = Tokenizer(["A", "C", "G", "T"], do_upper_case=False)
        >>> tokenizer('ACGTN')["input_ids"]
        [4, 0, 1, 2, 3, 6, 5]
        >>> tokenizer('acgtn')["input_ids"]
        [4, 6, 6, 6, 6, 6, 5]
        >>> tokenizer('AC<mask>gt')["input_ids"]
        [4, 0, 1, 8, 6, 6, 5]
    """

    model_input_names = ["input_ids", "attention_mask"]
    do_upper_case: bool = True

    def __init__(
        self,
        alphabet: List[str],
        bos_token: str = "<cls>",
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        sep_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        additional_special_tokens: List | Tuple | None = None,
        do_upper_case: bool = True,
        **kwargs,
    ):
        self._id_to_token = dict(enumerate(alphabet))
        self._token_to_id = {tok: ind for ind, tok in enumerate(alphabet)}
        if additional_special_tokens is None:
            additional_special_tokens = []
        if "<null>" in alphabet and "<null>" not in additional_special_tokens:
            additional_special_tokens = list(additional_special_tokens)
            additional_special_tokens.append("<null>")
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.do_upper_case = do_upper_case
        self._id_to_token.update(self.added_tokens_decoder)
        self._token_to_id.update(self.added_tokens_encoder)

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        # self.unique_no_split_tokens = self.all_tokens
        # self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))  # type: ignore[arg-type]

    def _tokenize(self, text: str, **kwargs):
        if self.do_upper_case:
            text = text.upper()
        return list(text)

    def get_vocab(self):
        return self._token_to_id.copy()

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))  # type: ignore[arg-type]

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            return cls + token_ids_0 + eos
        if self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + eos

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def all_tokens(self) -> List[str]:
        return list(self.get_vocab().keys())

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)
