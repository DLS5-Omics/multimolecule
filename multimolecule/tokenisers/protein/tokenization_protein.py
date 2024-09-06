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

from typing import List, Tuple

from transformers.utils import logging

from ..alphabet import Alphabet
from ..tokenization_utils import Tokenizer
from .utils import get_alphabet

logger = logging.get_logger(__name__)


class ProteinTokenizer(Tokenizer):
    """
    Tokenizer for Protein sequences.

    Args:
        alphabet: alphabet to use for tokenization.

            - If is `None`, the standard RNA alphabet will be used.
            - If is a `string`, it should correspond to the name of a predefined alphabet. The options include
                + `standard`
                + `iupac`
                + `streamline`
            - If is an alphabet or a list of characters, that specific alphabet will be used.
        do_upper_case: Whether to convert input to uppercase.

    Examples:
        >>> from multimolecule import ProteinTokenizer
        >>> tokenizer = ProteinTokenizer()
        >>> tokenizer('ACDEFGHIKLMNPQRSTVWYXZBJUO')["input_ids"]
        [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 2]
        >>> tokenizer('<pad><cls><eos><unk><mask><null>.*-')["input_ids"]
        [1, 0, 1, 2, 3, 4, 5, 32, 33, 34, 2]
        >>> tokenizer('manlgcwmlv')["input_ids"]
        [1, 16, 6, 17, 15, 11, 7, 24, 16, 15, 23, 2]
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        alphabet: Alphabet | str | List[str] | None = None,
        do_upper_case: bool = True,
        additional_special_tokens: List | Tuple | None = None,
        **kwargs,
    ):
        if not isinstance(alphabet, Alphabet):
            alphabet = get_alphabet(alphabet)
        super().__init__(
            alphabet=alphabet,
            additional_special_tokens=additional_special_tokens,
            do_upper_case=do_upper_case,
            **kwargs,
        )

    def _tokenize(self, text: str, **kwargs):
        if self.do_upper_case:
            text = text.upper()
        return list(text)
