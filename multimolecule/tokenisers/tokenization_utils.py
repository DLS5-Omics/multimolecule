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

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from .alphabet import Alphabet

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Base tokenizer.

    Args:
        alphabet: List of tokens or an Alphabet object to use in tokenization.
            Either alphabet or vocab_file must be specified.
        bos_token: A special token representing the beginning of a sequence.
        cls_token: A special token representing the classification token.
        pad_token: A special token representing padding.
        eos_token: A special token representing the end of a sequence.
        sep_token: A special token representing the separator token.
        unk_token: A special token representing unknown tokens.
        mask_token: A special token representing the mask token.
        null_token: A special token representing the null token.
        additional_special_tokens: Additional special tokens to add to the vocabulary.
        do_upper_case: Whether to convert input to uppercase.
        vocab_file: Path to a vocabulary file.
            Either alphabet or vocab_file must be specified.

    Examples:
        >>> from multimolecule.tokenisers import Tokenizer
        >>> tokenizer = Tokenizer(["A", "C", "G", "T", "N"], unk_token="N")
        >>> tokenizer('ACGTN')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> tokenizer('acgtn')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> len(tokenizer)
        5
        >>> tokenizer = Tokenizer(["A", "C", "G", "T", "N"], unk_token="N", do_upper_case=False)
        >>> tokenizer('ACGTN')["input_ids"]
        [0, 1, 2, 3, 4]
        >>> tokenizer('acgtn')["input_ids"]
        [4, 4, 4, 4, 4]
        >>> tokenizer('ACgtN')["input_ids"]
        [0, 1, 4, 4, 4]
        >>> tokenizer = Tokenizer(["<pad>", "<cls>", "A", "C", "G", "T", "N", "<mask>", "<eos>"])
        >>> tokenizer('ACGTN')["input_ids"]
        [1, 2, 3, 4, 5, 6, 8]
        >>> tokenizer('AC<mask>GTN')["input_ids"]
        [1, 2, 3, 7, 4, 5, 6, 8]
        >>> tokenizer(['TATATAT', 'ATCGN'], padding=True)["input_ids"]
        [[1, 5, 2, 5, 2, 5, 2, 5, 8], [1, 2, 5, 3, 4, 6, 8, 0, 0]]
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES
    do_upper_case: bool = True

    def __init__(
        self,
        alphabet: Alphabet | List[str] | None = None,
        bos_token: str | None = ...,  # type: ignore[assignment]
        cls_token: str | None = ...,  # type: ignore[assignment]
        pad_token: str | None = ...,  # type: ignore[assignment]
        eos_token: str | None = ...,  # type: ignore[assignment]
        sep_token: str | None = ...,  # type: ignore[assignment]
        unk_token: str | None = ...,  # type: ignore[assignment]
        mask_token: str | None = ...,  # type: ignore[assignment]
        null_token: str | None = ...,  # type: ignore[assignment]
        additional_special_tokens: List | Tuple | None = None,
        do_upper_case: bool = True,
        vocab_file: str | None = None,
        **kwargs,
    ):
        if alphabet is None and vocab_file is None:
            raise ValueError("You must specify either alphabet or vocab_file")

        if vocab_file is not None:
            alphabet = self.load_vocabulary(vocab_file)

        self._id_to_token = OrderedDict(enumerate(alphabet))
        self._token_to_id = OrderedDict({tok: ind for ind, tok in enumerate(alphabet)})

        if cls_token is ...:
            cls_token = self.identify_special_token(alphabet, "cls")
        if bos_token is ...:
            bos_token = cls_token
        if pad_token is ...:
            pad_token = self.identify_special_token(alphabet, "pad")
        if eos_token is ...:
            eos_token = self.identify_special_token(alphabet, "eos")
        if sep_token is ...:
            sep_token = self.identify_special_token(alphabet, "sep") or self.identify_special_token(alphabet, "eos")
        if unk_token is ...:
            unk_token = self.identify_special_token(alphabet, "unk")
        if mask_token is ...:
            mask_token = self.identify_special_token(alphabet, "mask")
        if null_token is ...:
            null_token = self.identify_special_token(alphabet, "null")
        if additional_special_tokens is None:
            additional_special_tokens = []
        if null_token in alphabet and null_token not in additional_special_tokens:  # type: ignore[operator]
            additional_special_tokens = list(additional_special_tokens)
            additional_special_tokens.append(null_token)

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

    def _tokenize(self, text: str, **kwargs):
        if self.do_upper_case:
            text = text.upper()
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def token_to_id(self, token: str) -> int:
        return self._convert_token_to_id(token)

    def id_to_token(self, index: int) -> str:
        return self._convert_id_to_token(index)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        bos = [self.bos_token_id]  # points to cls
        sep = [self.sep_token_id]  # points to eos
        eos = [self.eos_token_id]  # eos is eos
        if token_ids_1 is None:
            if self.bos_token_id is None:
                if self.eos_token_id is None:
                    return token_ids_0
                return token_ids_0 + eos
            if self.eos_token_id is None:
                return bos + token_ids_0
            return bos + token_ids_0 + eos
        if self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return bos + token_ids_0 + sep + token_ids_1 + eos

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
        mask = [0] * len(token_ids_0)
        if self.bos_token_id is not None:
            mask = [1] + mask
        if self.sep_token_id is not None:
            mask += [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1)
            if self.eos_token_id is not None:
                mask += [1]
        return mask

    @staticmethod
    def load_vocabulary(vocab_file: str | Path) -> List[str]:
        with open(vocab_file, encoding="utf-8") as reader:
            vocabulary = reader.read().splitlines()
        return vocabulary

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @staticmethod
    def identify_special_token(alphabet: Alphabet | List[str], token) -> str | None:
        tokens = [i for i in alphabet if token in i.lower()]
        if len(tokens) == 1:
            return tokens[0]
        if len(tokens) == 0:
            return None
        raise ValueError(f"Token {token} is ambiguous, could be {tokens}")

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    @property
    def vocab(self) -> OrderedDict[str, int]:
        return self._token_to_id.copy()

    @property
    def all_tokens(self) -> List[str]:
        return list(self.get_vocab().keys())

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)
