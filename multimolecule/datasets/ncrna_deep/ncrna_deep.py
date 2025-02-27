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


import random
from textwrap import wrap

from Bio import SeqIO
from Bio.Seq import Seq

random.seed(1016)


def get_seqs_with_bnoise(file: str, nperc: int = 0, dinucleotide: str = "preserve") -> list[Seq]:
    """
    Adds noise to sequences from a FASTA file by appending and prepending random bases or dinucleotides.

    Args:
        file: Path to the input FASTA file.
        nperc: Percentage of the sequence length to determine the amount of noise to add.
        dinucleotide: If 'preserve', noise is added as dinucleotides; otherwise, as single bases.

    Returns:
        (): A list of sequences with added noise.
    """
    sequences = [i.seq.transcribe() for i in SeqIO.parse(file, "fasta")]

    if nperc <= 0:
        return sequences

    bases = ["A", "U", "C", "G"]
    weights = [0.25, 0.25, 0.25, 0.25]
    noisy_sequences = []

    for record in sequences:
        original_seq = str(record)
        seq_length = len(original_seq)
        stop_noise = ""
        sbottom_noise = ""

        if dinucleotide == "preserve":
            dinucleotides = wrap(original_seq, 2)
            num_dinucs = int(0.25 * seq_length * nperc / 100)
            if num_dinucs > 0:
                stop_noise = "".join(random.choices(dinucleotides, k=num_dinucs))
                sbottom_noise = "".join(random.choices(dinucleotides, k=num_dinucs))
        else:
            num_bases = int(0.5 * seq_length * nperc / 100)
            if num_bases > 0:
                stop_noise = "".join(random.choices(bases, weights=weights, k=num_bases))
                sbottom_noise = "".join(random.choices(bases, weights=weights, k=num_bases))

        combined_seq = f"{stop_noise}{original_seq}{sbottom_noise}"
        noisy_sequences.append(combined_seq)

    return noisy_sequences
