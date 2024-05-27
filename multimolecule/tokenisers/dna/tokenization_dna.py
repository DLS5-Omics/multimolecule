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

from ..tokenization_utils import Tokenizer
from .utils import get_vocab_list

logger = logging.get_logger(__name__)


class DnaTokenizer(Tokenizer):
    """
    Constructs a DNA tokenizer.

    Args:
        alphabet (List[str] | None, optional): List of tokens to use.
            Defaults to [IUPAC nucleotide code](https://www.bioinformatics.org/sms2/iupac.html).
        nmers (int, optional): Size of nmers to tokenize.
            Defaults to 1.
        codon (bool, optional): Whether to tokenize into codons.
            Defaults to False.
        replace_U_with_T (bool, optional): Whether to replace U with T.
            Defaults to True.
        do_upper_case (bool, optional): Whether to convert input to uppercase.
            Defaults to True.

    Examples:
        >>> from multimolecule import DnaTokenizer
        >>> tokenizer = DnaTokenizer()
        >>> tokenizer('<pad><cls><eos><unk><mask><null>ACGTNXVHDBMRWSYK.*-')["input_ids"]
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
        ValueError: length of input sequence  must be a multiple of 3 for codon tokenization, but got 10
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        alphabet: List[str] | None = None,
        nmers: int = 1,
        codon: bool = False,
        bos_token: str = "<cls>",
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        sep_token: str = "<eos>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        additional_special_tokens: List | Tuple | None = None,
        replace_U_with_T: bool = True,
        do_upper_case: bool = True,
        **kwargs,
    ):
        if codon and nmers > 1:
            raise ValueError("Codon and nmers cannot be used together.")
        if codon:
            nmers = 3  # set to 3 to get correct vocab
        super().__init__(
            alphabet=get_vocab_list(alphabet, nmers),
            bos_token=bos_token,
            cls_token=cls_token,
            pad_token=pad_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            do_upper_case=do_upper_case,
            **kwargs,
        )
        self.replace_U_with_T = replace_U_with_T
        self.nmers = nmers
        self.condon = codon

    def _tokenize(self, text: str, **kwargs):
        if self.do_upper_case:
            text = text.upper()
        if self.replace_U_with_T:
            text = text.replace("U", "T")
        if self.condon:
            if len(text) % 3 != 0:
                raise ValueError(
                    f"length of input sequence  must be a multiple of 3 for codon tokenization, but got {len(text)}"
                )
            return [text[i : i + 3] for i in range(0, len(text), 3)]
        if self.nmers > 1:
            return [text[i : i + self.nmers] for i in range(len(text) - self.nmers + 1)]  # noqa: E203
        return list(text)
