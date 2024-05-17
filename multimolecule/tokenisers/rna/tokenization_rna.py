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
import re
from typing import List, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from .utils import get_vocab_list

logger = logging.get_logger(__name__)


class RnaTokenizer(PreTrainedTokenizer):
    """
    Constructs an RnaBert tokenizer.

    Examples:
        >>> from multimolecule import RnaTokenizer
        >>> tokenizer = RnaTokenizer()
        >>> tokenizer('<pad><cls><eos><unk><mask><null>ACGUNXVHDBMRWSYK.*-')["input_ids"]
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2]
        >>> tokenizer('acgu')["input_ids"]
        [1, 6, 7, 8, 9, 2]
        >>> tokenizer('acgt')["input_ids"]
        [1, 6, 7, 8, 9, 2]
        >>> tokenizer = RnaTokenizer(convert_T_to_U=False)
        >>> tokenizer('acgt')["input_ids"]
        [1, 6, 7, 8, 3, 2]
        >>> tokenizer = RnaTokenizer(convert_to_uppercase=False)
        >>> tokenizer('acgu')["input_ids"]
        [1, 3, 3, 3, 3, 2]
        >>> tokenizer('ACGU')["input_ids"]
        [1, 6, 7, 8, 9, 2]
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bos_token: str = "<cls>",
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        sep_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        additional_special_tokens: List | Tuple | None = None,
        do_upper_case: bool = True,
        convert_T_to_U: bool = True,
        nmers: int = 1,
        strameline: bool | None = None,
        **kwargs,
    ):
        self.nmers = nmers
        self.strameline = strameline if strameline is not None else nmers > 1
        self.all_tokens = get_vocab_list(nmers, self.strameline)
        self._id_to_token = dict(enumerate(self.all_tokens))
        if additional_special_tokens is None:
            additional_special_tokens = []
        if "<null>" in self.all_tokens and "<null>" not in additional_special_tokens:
            additional_special_tokens = list(additional_special_tokens)
            additional_special_tokens.append("<null>")
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        self.do_upper_case = do_upper_case
        self.convert_T_to_U = convert_T_to_U
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

        self._update_pattern()

    def _add_tokens(self, new_tokens: List, special_tokens: bool = False) -> int:
        ret = super()._add_tokens(new_tokens, special_tokens=special_tokens)
        self._update_pattern()
        return ret

    def _update_pattern(self):
        escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
        escaped_special_toks += [
            re.escape(s_tok.content)
            for s_tok in self._added_tokens_decoder.values()
            if not s_tok.special and s_tok.normalized
        ]
        self.pattern = re.compile(
            "(" + "|".join(escaped_special_toks) + "|.{" + str(self.nmers) + "})", flags=re.DOTALL
        )

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))  # type: ignore[arg-type]

    def _tokenize(self, text: str, **kwargs):
        if self.convert_T_to_U:
            text = text.replace("T", "U")

        if self.do_upper_case:
            text = text.upper()

        tokens = []
        special_tokens_set = set(self.all_special_tokens)

        # Define a function to process matched tokens
        def process_token(token):
            if token not in special_tokens_set:
                tokens.append(token)

        # Tokenize using regular expression pattern
        re_pattern = self.pattern.pattern
        start_index = 0
        for match in re.finditer(re_pattern, text):
            start, end = match.span()
            if start > start_index:
                process_token(text[start_index:start])
            process_token(text[start:end])
            start_index = end

        # Process any remaining text
        if start_index < len(text):
            process_token(text[start_index:])

        return tokens

    def get_vocab(self):
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))  # type: ignore[arg-type]

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # No sep token in RnaBert vocabulary
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # Multiple inputs always have an EOS token

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
    def vocab_size(self) -> int:
        return len(self.all_tokens)
