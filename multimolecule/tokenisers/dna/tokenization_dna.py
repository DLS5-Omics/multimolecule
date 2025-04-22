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

from typing import List, Tuple

from ..alphabet import Alphabet
from ..tokenization_utils import Tokenizer
from .utils import get_alphabet


class DnaTokenizer(Tokenizer):
    """
    Tokenizer for DNA sequences.

    Args:
        alphabet: alphabet to use for tokenization.

            - If is `None`, the standard RNA alphabet will be used.
            - If is a `string`, it should correspond to the name of a predefined alphabet. The options include
                + `standard`
                + `iupac`
                + `streamline`
                + `nucleobase`
            - If is an alphabet or a list of characters, that specific alphabet will be used.
        nmers: Size of kmer to tokenize.
        codon: Whether to tokenize into codons.
        replace_U_with_T: Whether to replace U with T.
        do_upper_case: Whether to convert input to uppercase.

    Examples:
        >>> from multimolecule import DnaTokenizer
        >>> tokenizer = DnaTokenizer()
        >>> tokenizer('<pad><cls><eos><unk><mask><null>ACGTNRYSWKMBDHV.X*-')["input_ids"]
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2]
        >>> tokenizer('acgt')["input_ids"]
        [1, 6, 7, 8, 9, 2]
        >>> tokenizer('acgu')["input_ids"]
        [1, 6, 7, 8, 9, 2]
        >>> tokenizer = DnaTokenizer(replace_U_with_T=False)
        >>> tokenizer('acgu')["input_ids"]
        [1, 6, 7, 8, 3, 2]
        >>> tokenizer = DnaTokenizer(nmers=3)
        >>> tokenizer('tataaagta')["input_ids"]
        [1, 84, 21, 81, 6, 8, 19, 71, 2]
        >>> tokenizer = DnaTokenizer(codon=True)
        >>> tokenizer('tataaagta')["input_ids"]
        [1, 84, 6, 71, 2]
        >>> tokenizer('tataaagtaa')["input_ids"]
        Traceback (most recent call last):
        ValueError: length of input sequence must be a multiple of 3 for codon tokenization, but got 10
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        alphabet: Alphabet | str | List[str] | None = None,
        nmers: int = 1,
        codon: bool = False,
        replace_U_with_T: bool = True,
        do_upper_case: bool = True,
        additional_special_tokens: List | Tuple | None = None,
        **kwargs,
    ):
        if codon and (nmers > 1 and nmers != 3):
            raise ValueError("Codon and nmers cannot be used together.")
        if codon:
            nmers = 3  # set to 3 to get correct vocab
        if not isinstance(alphabet, Alphabet):
            alphabet = get_alphabet(alphabet, nmers=nmers)
        super().__init__(
            alphabet=alphabet,
            nmers=nmers,
            codon=codon,
            replace_U_with_T=replace_U_with_T,
            do_upper_case=do_upper_case,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.replace_U_with_T = replace_U_with_T
        self.nmers = nmers
        self.codon = codon

    def _tokenize(self, text: str, **kwargs):
        if self.do_upper_case:
            text = text.upper()
        if self.replace_U_with_T:
            text = text.replace("U", "T")
        if self.codon:
            if len(text) % 3 != 0:
                raise ValueError(
                    f"length of input sequence must be a multiple of 3 for codon tokenization, but got {len(text)}"
                )
            return [text[i : i + 3] for i in range(0, len(text), 3)]
        if self.nmers > 1:
            return [text[i : i + self.nmers] for i in range(len(text) - self.nmers + 1)]  # noqa: E203
        return list(text)
