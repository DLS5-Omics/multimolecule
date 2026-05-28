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

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from warnings import warn

import frontmatter as fm
import torch
from chanfig import Config
from safetensors.torch import load_file
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, pipeline
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from transformers.pipelines.base import Pipeline

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

text_generation_tokens = 20

reference_sequences = {
    "DNA": {
        "tumor protein p53": "ACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGG",  # https://www.ncbi.nlm.nih.gov/nuccore/NG_017013.2?from=17316&to=17499  # noqa: E501
        "BRCA1 DNA repair associated": "TCATTGGAACAGAAAGAAATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGG",  # https://www.ncbi.nlm.nih.gov/nuccore/NG_005905.2?from=93870&to=93968  # noqa: E501
        "hemoglobin subunit beta": "CATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGG",  # https://www.ncbi.nlm.nih.gov/nuccore/NG_000007.3?from=70546&to=70687  # noqa: E501
        "CF transmembrane conductance regulator": "ACTTCACTTCTAATGGTGATTATGGGAGAACTGGAGCCTTCAGAGGGTAAAATTAAGCACAGTGGAAGAATTTCATTCTGTTCTCAGTTTTCCTGGATTATGCCTGGCACCATTAAAGAAAATATCATCTTTGGTGTTTCCTATGATGAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAG",  # https://www.ncbi.nlm.nih.gov/nuccore/NG_016465.4?from=98681&to=98872  # noqa: E501
        "telomerase reverse transcriptase": "CGCGGGGGTGGCCGGGGCCAGGGCTTCCCACGTGCGCAGCAGGACGCAGCGCTGCCTGAAACTCGCGCCGCGAGGAGAGGGCGGGGCCGCGGAAAGGAAGGGGAGGGGCTGGGAGGGCCCGGAGGGGGCTGGGCCGGGGACCCGGGAGGGGTCGGGACGGGGCGGGGTCCGCGCGGAGGAGGCGGAGCTGGAAGGTGAAGGGGCAGGACGGGTGCCCGGGTCCCCAGTCCCTCCGCCACGTGGGAAGCGCGGTCCTGGGCGTCTGTGCCCGCGAATCCACTGGGAGCCCGGCCTGGCCCCGACAGCGCAGCTGCTCCGGGCGGACCCGGGG",  # https://www.ncbi.nlm.nih.gov/nuccore/NC_000005.10?from=1294990&to=1295320  # noqa: E501
        "KRAS proto-oncogene": "GCCTGCTGAAAATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAG",  # https://www.ncbi.nlm.nih.gov/nuccore/NG_007524.2?from=10609&to=10730  # noqa: E501
    },
    "cDNA": {
        "prion protein (Kanno blood group)": "ATGGCGAACCTTGGCTGCTGGATGCTGGTTCTCTTTGTGGCCACATGGAGTGACCTGGGCCTCTGC",  # https://www.ncbi.nlm.nih.gov/nuccore/M13899.1?from=50&to=115  # noqa: E501
        "interleukin 10": "ATGCACAGCTCAGCACTGCTCTGTTGCCTGGTCCTCCTGACTGGGGTGAGGGCC",  # https://www.ncbi.nlm.nih.gov/nuccore/M57627.1?from=31&to=84  # noqa: E501
        "Zaire ebolavirus": "AATGTTCAAACACTTTGTGAAGCTCTGTTAGCTGATGGTCTTGCTAAAGCATTTCCTAGCAATATGATGGTAGTCACAGAGCGTGAGCAAAAAGAAAGCTTATTGCATCAAGCATCATGGCACCACACAAGTGATGATTTTGGTGAGCATGCCACAGTTAGAGGGAGTAGCTTTGTAACTGATTTAGAGAAATACAATCTTGCATTTAGATATGAGTTTACAGCACCTTTTATAGAATATTGTAACCGTTGCTATGGTGTTAAGAATGTTTTTAATTGGATGCATTATACAATCCCACAGTGTTAT",  # https://www.ncbi.nlm.nih.gov/nuccore/DQ211657.1?from=2&to=307  # noqa: E501
        "SARS coronavirus": "ATGTTTATTTTCTTATTATTTCTTACTCTCACTAGTGGTAGTGACCTTGACCGGTGCACCACTTTTGATGATGTTCAAGCTCCTAATTACACTCAACATACTTCATCTATGAGGGGGGTTTACTATCCTGATGAAATTTTTAGATCAGACACTCTTTATTTAACTCAGGATTTATTTCTTCCATTTTATTCTAATGTTACAGGGTTTCATACTATTAATCATACGTTTGACAACCCTGTCATACCTTTTAAGGATGGTATTTATTTTGCTGCCACAGAGAAATCAAATGTTGTCCGTGGTTGGGTTTTTGGTTCTACCATGAACAACAAGTCACAGTCGGTGATTATTATTAACAATTCTACTAATGTTGTTATACGAGCATGTAACTTTGAATTGTGTGACAACCCTTTCTTTGCTGTTTCTAAACCCATGGGTACACAGACACATACTATGATATTCGATAATGCATTTAAATGCACTTTCGAGTACATATCT",  # https://www.ncbi.nlm.nih.gov/nuccore/AY536757.3?from=73&to=567  # noqa: E501
        "insulin": "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCGGGGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAACTAG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_000207.3?from=60&to=392  # noqa: E501
        "cyclin dependent kinase inhibitor 2A": "ATGGAGCCGGCGGCGGGGAGCAGCATGGAGCCTTCGGCTGACTGGCTGGCCACGGCCGCGGCCCGGGGTCGGGTAGAGGAGGTGCGGGCGCTGCTGGAGGCGGGGGCGCTGCCCAACGCACCGAATAGTTACGGTCGGAGGCCGATCCAGGTCATGATGATGGGCAGCGCCCGAGTGGCGGAGCTGCTGCTGCTCCACGGCGCGGAGCCCAACTGCGCCGACCCCGCCACTCTCACCCGACCCGTGCACGACGCTGCCCGGGAGGGCTTCCTGGACACGCTGGTGGTGCTGCACCGGGCCGGGGCGCGGCTGGACGTGCGCGATGCCTGGGGCCGTCTGCCCGTGGACCTGGCTGAGGAGCTGGGCCATCGCGATGTCGCACGGTACCTGCGCGCGGCTGCGGGGGGCACCAGAGGCAGTAACCATGCCCGCATAGATGCCGCGGAAGGTCCCTCAGACATCCCCGATTGA",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_000077.5?from=31&to=501  # noqa: E501
        "human papillomavirus type 16 E6": "ATGCACCAAAAGAGAACTGCAATGTTTCAGGACCCACAGGAGCGACCCAGAAAGTTACCACAGTTATGCACAGAGCTGCAAACAACTATACATGATATAATATTAGAATGTGTGTACTGCAAGCAACAGTTACTGCGACGTGAGGTATATGACTTTGCTTTTCGGGATTTATGCATAGTATATAGAGATGGGAATCCATATGCTGTATGTGATAAATGTTTAAAGTTTTATTCTAAAATTAGTGAGTATAGACATTATTGTTATAGTTTGTATGGAACAACATTAGAACAGCAATACAACAAACCGTTGTGTGATTTGTTAATTAGGTGTATTAACTGTCAAAAGCCACTGTGTCCTGAAGAAAAGCAAAGACATCTGGACAAAAAGCAAAGATTCCATAATATAAGGGGTCGGTGGACCGGTCGATGTATGTCTTGTTGCAGATCATCAAGAACACGTAGAGAAACCCAGCTGTAA",  # https://www.ncbi.nlm.nih.gov/nuccore/K02718.1?from=83&to=559  # noqa: E501
    },
    "ncRNA": {
        "microRNA 21": "UAGCUUAUCAGACUGAUGUUGA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_029493.1?from=8&to=29
        "microRNA 146a": "UGAGAACUGAAUUCCAUGGGUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_029701.1?from=21&to=42
        "microRNA 155": "UUAAUGCUAAUCGUGAUAGGGGUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_030784.1?from=4&to=27
        "RNA component of mitochondrial RNA processing endoribonuclease": "GGUUCGUGCUGAAGGCCUGUAUCCUAGGCUACACACUGAGGACUCUGUUCCUCCCCUUUCCGCCUAGGGGAAAGUCCCCGGACCUCGGGCAGAGAGUGCCACGUGCAUACGCACGUAGACAUUCCCCGCUUCCCACUCCAAAGUCCGCCAAGAAGCGUAUCCCGCUGAGCGGCGUGGCGCGGGGGCGUCAUCCGUCAGCUCCCUCUAGUUACGCAGGCAGUGCGUGUCCGCGCACCAACCACACGGGGCUCAUUCUCAGCGCGGCUGUAAAAAAAAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_003051.3  # noqa: E501
        "7SK small nuclear RNA": "GGAUGUGAGGGCGAUCUGGCUGCGACAUCUGUCACCCCAUUGAUCGCCAGGGUUGAUUCGGCUGAUCUGGCUGGCUAGGCGGGUGUCCCCUUCCUCCCUCACCGCUCCAUGUGCGUCCCUCCCGAAGCUGCGCGCUCGGUCGAAGAGGACGACCAUCCCCGAUAGAGGAGGACCGGUCUUCGGUCAAGGGUAUACGAGUAGCUGCGCUCCCCUGCUAGAACCUCCAAACAAGCUCUCAAGGUCCAUUUGUAGGAGAACGUAGGGUAGUCAAGCUUCCAAGACUCCAGACACAUCCAAAUGAGGCGCUGCAUGUGGCAGUCUGCCUUUCUUUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_001445.2  # noqa: E501
        "telomerase RNA component": "GGGUUGCGGAGGGUGGGCCUGGGAGGGGUGGUGGCCAUUUUUUGUCUAACCCUAACUGAGAAGGGCGUAGGCGCCGUGCUUUUGCUCCCCGCGCGCUGUUUUUCUCGCUGACUUUCAGCGGGCGGAAAAGCCUCGGCCUGCCGCCUUCCACCGUUCAUUCUAGAGCAAACAAAAAAUGUCAGCUGCUGGCCCGUUCGCCCCUCCCGGGGACCUGCGGCGGGUCGCCUGCCCAGCCCCCGAACCCCGCCUGGAGGCCGCGGUCGGCCCGGGGCUUCUCCGGAGGCACCCACUGCCACCGCGAAGAGUUGGGCUCUGUCAGCCGCGGGUCUCUCGGGGGCGAGGGCGAGGUUCAGGCCUUUCAGGCCGCAGGAAGAGGAACGGAGCGAGUCCCCGCGCGCGGCGCGAUUCCCUGAGCUGUGGGACGUGCACCCAGGACUCGGCUCACACAUGC",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_001566.3  # noqa: E501
        "vault RNA 2-1": "CGGGUCGGAGUUAGCUCAAGCGGUUACCUCCUCAUGCCGGACUUUCUAUCUGUCCAUCUCUGUGCUGGGGUUCGAGACCCGCGGGUGCUUACUGACCCUUUUAUGCAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_030583.3  # noqa: E501
        "brain cytoplasmic RNA 1": "GGCCGGGCGCGGUGGCUCACGCCUGUAAUCCCAGCUCUCAGGGAGGCUAAGAGGCGGGAGGAUAGCUUGAGCCCAGGAGUUCGAGACCUGCCUGGGCAAUAUAGCGAGACCCCGUUCUCCAGAAAAAGGAAAAAAAAAAACAAAAGACAAAAAAAAAAUAAGCGUAACUUCCCUCAAAGCAACAACCCCCCCCCCCCUUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_001568.1  # noqa: E501
        "HIV-1 TAR-WT": "GGUCUCUCUGGUUAGACCAGAUCUGAGCCUGGGAGCUCUCUGGCUAACUAGGGAACC",  # https://pmc.ncbi.nlm.nih.gov/articles/PMC1955452/  # noqa: E501
    },
    "5' UTR": {
        "NRAS proto-oncogene": "GGGGCCGGAAGUGCCGCUCCUUGGUGGGGGCUGUUCAUGGCGGUUCCGGGGUCUCCAACAUUUUUCCCGGCUGUGGUCCUAAAUCUGUCCAAAGCAGAGGCAGUGGAGCUUGAGGUUCUUGCUGGUGUGAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_002524.5?from=1&to=131  # noqa: E501
        "amyloid beta precursor protein": "GUCAGUUUCCUCGGCAGCGGUAGGCGAGAGCACGCGGAGGAGCGUGCGCGGGGGCCCCGGGAGACGGCGGCGGUGGCGGCGCGGGCAGAGCAAGGACGCGGCGGAUCCCACUCGCACAGCAGCGCACUCGGUGCCCCGCGCAGGGUCGCG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_000484.4?from=1&to=150  # noqa: E501
        "RUNX family transcription factor 1": "ACUUCUUUGGGCCUCAUAAACAACCACAGAACCACAAGUUGGGUAGCCUGGCAGUGUCAGAAGUCUGAACCCAGCAUAGUGGUCAGCAGGCAGGACGAAUCACACUGAAUGCAAACCACAGGGUUUCGCAGCGUGGUAAAAGAAAUCAUUGAGUCCCCCGCCUUCAGAAGAGGGUGCAUUUUCAGGAGGAAGCG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_001754.5?from=1&to=194  # noqa: E501
        "fragile X messenger ribonucleoprotein 1": "CUCAGUCAGGCGCUCAGCUCCGUUUCGGUUUCACUUCCGGUGGAGGGCCGCCUCUGAGCGGGCGGCGGGCCGACGGCGAGCGCGGGCGGCGGCGGUGACGGAGGCGCCGCUGCCAGGGGGCGUGCGGCAGCGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGAGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCUGGGCCUCGAGCGCCCGCAGCCCACCUCUCGGGGGCGGGCUCCCGGCGCUAGCAGGGCUGAAGAGAAG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_002024.6?from=1&to=261  # noqa: E501
        "MYC proto-oncogene": "AACUCGCUGUAGUAAUUCCAGCGAGAGGCAGAGGGAGCGAGCGGGCGGCCGGCUAGGGUGGAAGAGCCGGGCGAGCAGAGCUGCGCUGCGGGCGUCCUGGGAAGGGAGAUCCGGAGCGAAUAGGGGGCUUCGCCUCUGGCCCAGCCCUCCCGCUGAUCCCCCAGCCAGCGGUCCGCAACCCUUGCCGCAUCCACGAAACUUUGCCCAUAGCAGCGGGCGGGCACUUUGCACUGGAACUUACAACACCCGAGCAAGGACGCGACUCUCCCGACGCGGGGAGGCUAUUCUGCCCAUUUGGGGACACUUCCCCGCCGCUGCCAGGACCCGCUUCUCUGAAAGGCUCUCCUUGCAGCUGCUUAGACG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_002467.6?from=1&to=363  # noqa: E501
        "activating transcription factor 4": "CAUUUCUACUUUGCCCGCCCACAGAUGUAGUUUUCUCUGCGCGUGUGCGUUUUCCCUCCUCCCCGCCCUCAGGGUCCACGGCCACCAUGGCGUAUUAGGGGCAGCAGUGCCUGCGGCAGCAUUGGCCUUUGCAGCGGCGGCAGCAGCACCAGGCUCUGCAGCGGCAACCCCCAGCGGCUUAAGCCAUGGCGCUUCUCACGGCAUUCAGCAGCAGCGUUGCUGUAACCGACAAAGACACCUUCGAAUUAAGCACAUUCCUCGAUUCCAGCAAAGCACCGCAAC",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_182810.3?from=1&to=282  # noqa: E501
    },
    "3' UTR": {
        "Human GPI protein p137": "UUUUUAAAAGGAAAAGAUACCAAAUGCCUGCUGCUACCACCCUUUUCAAUUGCUAUGUUUUGAAAGGCACCAGUAUGUGUUUUAGAUUGAUUUAAAUGUUUCAUUUAAAUCACGGACAGUAGUUUCAGUUCUGAUGGUAUAAGCAAAACAAAUAAAACGUUUAUAAAAGUUGUAUCUUGAAACACUGGUGUUCAACAGCUAGCAGCUUAUGUGAUUCACCCCAUGCCACGUUAGUGUCACAAAUUUUAUGGUUUAUCUCCAGCAACAUUUCUCUAGUACUUGCACUUAUUAUCUGAAUUC",  # https://www.ncbi.nlm.nih.gov/nuccore/U51714.1  # noqa: E501
        "nucleophosmin 1": "GAAAAUAGUUUAAACAAUUUGUUAAAAAAUUUUCCGUCUUAUUUCAUUUCUGUAACAGUUGAUAUCUGGCUGUCCUUUUUAUAAUGCAGAGUGAGAACUUUCCCUACCGUGUUUGAUAAAUGUUGUCCAGGUUCUAUUGCCAAGAAUGUGUUGUCCAAAAUGCCUGUUUAGUUUUUAAAGAUGGAACUCCACCCUUUGCUUGGUUUUAAGUAUGUAUGGAAUGUUAUGAUAGGACAUAGUAGUAGCGGUGGUCAGACAUGGAAAUGGUGGGGAGACAAAAAUAUACAUGUGAAAUAAAACUCAGUAUUUUAAUAAAGUAGCACGGUUUCUAUUGA",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_002520.7?from=986&to=1320  # noqa: E501
        "superoxide dismutase 1": "ACAUUCCCUUGGAUGUAGUCUGAGGCCCCUUAACUCAUCUGUUAUCCUGCUAGCUGUAGAAAUGUAUCCUGAUAAACAUUAAACACUGUAAUCUUAAAAGUGUAAUUGUGUGACUUUUUCAGAGUUGCUUUAAAGUACCUGUAGUGAGAAACUGAUUUAUGAUCACUUGGAAGAUUUGUAUAGUUUUAUAAAACUCAGUUAAAAUGUCUGUUUCAAUGACCUGUAUUUUGCCAGACUUAAAUCACAGAUGGGUAUUAAACUUGUCAGAAUUUCUUUGUCAUUCAAGCCUGUGAAUAAAAACCCUGUAUGGCACUUAUUAUGAGGCUAUUAAAAGAAUCCAAAUUCAAACUAAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_000454.5?from=543&to=895  # noqa: E501
        "hemoglobin subunit alpha 2": "CUGGAGCCUCGGUAGCCGUUCCUCCUGCCCGCUGGGCCUCCCAACGGGCCCUCCUCCCCUCCUUGCACCGGCCCUUCCUGGUCUUUGAAUAAAGUCUGAGUGGGCAGCA",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_000517.6?from=468&to=576  # noqa: E501
        "BRAF proto-oncogene": "AACAAAUGAGUGAGAGAGUUCAGGAGAGUAGCAACAAAAGGAAAAUAAAUGAACAUAUGUUUGCUUAUAUGUUAAAUUGAAUAAAAUACUCUCUUUUUUUUUAAGGUGAACCAAAGAACACUUGUGUGGUUAAAGACUAGAUAUAAUUUUUCCCCAAACUAAAAUUUAUACUUAACAUUGGAUUUUUAACAUCCAAGGGUUAAAAUACAUAGACAUUGCUAAAAAUUGGCAGAGCCUCUUCUAGAGGCUUUACUUUCUGUUCCGGGUUUGUAUCAUUCACUUGGUUAUUUUAAGUAGUAAACUUCAGUUUCUCAUGCAACUUUUGUUGCCAGCUAUCACAUGUCCACUAGGGACUCCAGAAGAAGACCCUACCUAUGCCUGUGUUUGCAGGUGAGAAGUUGGCAGUCGGUUAGCCUGGG",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_004333.6?from=2528&to=2946  # noqa: E501
        "H3 clustered histone 1": "UUACUGUGGUCUCUCUGACGGUCCAAGCAAAGGCUCUUUUCAGAGCCACCACCUUUUC",  # https://www.ncbi.nlm.nih.gov/nuccore/NM_003529.3?from=451&to=508  # noqa: E501
    },
    "Protein": {
        "prion protein (Kanno blood group)": "MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSDYEDRYYRENMHRYPNQVYYRPMDEYSNQNNFVHDCVNITIKQHTVTTTTKGENFTETDVKMMERVVEQMCITQYERESQAYYQRGSSMVLFSSPPVILLISFLIFLIVG",  # https://www.ncbi.nlm.nih.gov/nuccore/M13899.1  # noqa: E501
        "interleukin 10": "MHSSALLCCLVLLTGVRASPGQGTQSENSCTHFPGNLPNMLRDLRDAFSRVKTFFQMKDQLDNLLLKESLLEDFKGYLGCQALSEMIQFYLEEVMPQAENQDPDIKAHVNSLGENLKTLRLRLRRCHRFLPCENKSKAVEQVKNAFNKLQEKGIYKAMSEFDIFINYIEAYMTMKIRN",  # https://www.ncbi.nlm.nih.gov/nuccore/M57627.1  # noqa: E501
        "Zaire ebolavirus": "NVQTLCEALLADGLAKAFPSNMMVVTEREQKESLLHQASWHHTSDDFGEHATVRGSSFVTDLEKYNLAFRYEFTAPFIEYCNRCYGVKNVFNWMHYTIPQCY",  # https://www.ncbi.nlm.nih.gov/nuccore/DQ211657.1  # noqa: E501
        "SARS coronavirus": "MFIFLLFLTLTSGSDLDRCTTFDDVQAPNYTQHTSSMRGVYYPDEIFRSDTLYLTQDLFLPFYSNVTGFHTINHTFDNPVIPFKDGIYFAATEKSNVVRGWVFGSTMNNKSQSVIIINNSTNVVIRACNFELCDNPFFAVSKPMGTQTHTMIFDNAFKCTFEYIS",  # https://www.ncbi.nlm.nih.gov/nuccore/AY536757.3  # noqa: E501
        "insulin": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",  # https://www.ncbi.nlm.nih.gov/protein/NP_000198.1  # noqa: E501
        "cyclin dependent kinase inhibitor 2A": "MEPAAGSSMEPSADWLATAAARGRVEEVRALLEAGALPNAPNSYGRRPIQVMMMGSARVAELLLLHGAEPNCADPATLTRPVHDAAREGFLDTLVVLHRAGARLDVRDAWGRLPVDLAEELGHRDVARYLRAAAGGTRGSNHARIDAAEGPSDIPD",  # https://www.ncbi.nlm.nih.gov/protein/NP_000068.1  # noqa: E501
        "human papillomavirus type 16 E6": "MHQKRTAMFQDPQERPRKLPQLCTELQTTIHDIILECVYCKQQLLRREVYDFAFRDLCIVYRDGNPYAVCDKCLKFYSKISEYRHYCYSVYGTTLEQQYNKPLCDLLIRCINCQKPLCPEEKQRHLDKKQRFHNIRGRWTGRCMSCCRSSRTRRETQL",  # https://www.ncbi.nlm.nih.gov/protein/AAD33252.1  # noqa: E501
    },
}
reference_sequences["mRNA"] = {k: v.replace("T", "U") for k, v in reference_sequences["cDNA"].items()}


def normalize_reference_tag(tag: str) -> str:
    return "".join(ch for ch in tag.lower() if ch.isalnum())


def iter_reference_sequences(tag: str):
    normalized_tag = normalize_reference_tag(tag)
    if not normalized_tag:
        return
    if normalized_tag == "rna":
        yield "ncRNA", reference_sequences["ncRNA"]
        yield "mRNA", reference_sequences["mRNA"]
        yield "5' UTR", reference_sequences["5' UTR"]
        yield "3' UTR", reference_sequences["3' UTR"]
        return
    if normalized_tag == "dna":
        yield "DNA", reference_sequences["DNA"]
        yield "cDNA", reference_sequences["cDNA"]
        return
    if tag in reference_sequences:
        yield tag, reference_sequences[tag]
    return


def load_checkpoint(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, assign=True)
    for name, state in model.state_dict().items():
        if not torch.equal(state, state_dict[name]):
            raise ValueError("State dicts do not match after conversion.")


def convert_one_hot_embeddings(
    embeddings: torch.Tensor,
    *,
    old_vocab: List[str],
    new_vocab: List[str],
    convert_word_embeddings: Callable[..., Sequence[torch.Tensor]],
    channel_dim: int = 1,
) -> torch.Tensor:
    """Convert one-hot input-channel weights using the tokenizer word-embedding conversion rules."""
    channel_first = embeddings.movedim(channel_dim, 0).contiguous()
    (converted,) = convert_word_embeddings(
        channel_first,
        old_vocab=old_vocab,
        new_vocab=new_vocab,
        mean=0.0,
        std=0.0,
        seed=None,
    )
    converted = converted.to(device=embeddings.device, dtype=embeddings.dtype)
    return converted.movedim(0, channel_dim).contiguous()


def _normalize_variant_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _bold_variant_in_table(content: str, output_path: str) -> str:
    normalized_target = _normalize_variant_name(output_path)
    pattern = re.compile(r"<td>([^<]+)</td>")

    # Try exact normalized match first
    def exact_replacer(match: re.Match) -> str:
        cell_text = match.group(1)
        if _normalize_variant_name(cell_text) == normalized_target:
            return f"<td><b>{cell_text}</b></td>"
        return match.group(0)

    result = pattern.sub(exact_replacer, content)
    if result != content:
        return result

    # Fall back to suffix match (handles e.g. "3UTRBERT-3mer" vs "utrbert-3mer")
    def suffix_replacer(match: re.Match) -> str:
        cell_text = match.group(1)
        normalized_cell = _normalize_variant_name(cell_text)
        if normalized_cell.endswith(normalized_target) and len(normalized_target) > 3:
            return f"<td><b>{cell_text}</b></td>"
        return match.group(0)

    return pattern.sub(suffix_replacer, content)


def _replace_default_variant_references(content: str, output_path: str, default_variant: str) -> str:
    target = f"multimolecule/{default_variant}"
    replacement = f"multimolecule/{output_path}"
    lines = []
    in_variant_section = False
    heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for line in content.splitlines(keepends=True):
        match = heading.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip().lower()
            if level <= 3:
                in_variant_section = level == 3 and title in {"variants", "variations"}
        lines.append(line if in_variant_section else line.replace(target, replacement))
    return "".join(lines)


def customize_readme_for_variant(readme_path: str, output_path: str, default_variant: str | None = None) -> None:
    with open(readme_path) as f:
        content = f.read()

    content = _bold_variant_in_table(content, output_path)

    if default_variant and default_variant != output_path:
        content = _replace_default_variant_references(content, output_path, default_variant)

    with open(readme_path, "w") as f:
        f.write(content)


def append_output_suffix(convert_config: ConvertConfig, suffix: str) -> None:
    suffix = suffix if suffix.startswith("-") else f"-{suffix}"
    output_name = os.path.basename(convert_config.output_path.rstrip(os.sep))
    if not output_name.endswith(suffix):
        convert_config.output_path += suffix
    if convert_config.repo_id is not None:
        repo_name = convert_config.repo_id.rsplit("/", 1)[-1]
        if not repo_name.endswith(suffix):
            convert_config.repo_id += suffix


def should_derive_output_path(convert_config: ConvertConfig, default_output_path: str) -> bool:
    output_name = os.path.basename(convert_config.output_path.rstrip(os.sep))
    default_name = os.path.basename(default_output_path.rstrip(os.sep))
    return output_name == default_name


def write_model(
    model_path: str,
    output_path: str,
    model: PreTrainedModel,
    tokenizer_config: Dict,
    default_variant: str | None = None,
):
    model.save_pretrained(output_path, safe_serialization=True)
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    if hasattr(model.config, "max_position_embeddings") and "model_max_length" not in tokenizer_config:
        position_embedding_type = getattr(model.config, "position_embedding_type", None)
        if position_embedding_type == "absolute":
            tokenizer_config["model_max_length"] = model.config.max_position_embeddings
        else:
            tokenizer_config["model_max_length"] = None
    tokenizer = tokenizer_class_from_name(tokenizer_config["tokenizer_class"])(**tokenizer_config)
    tokenizer.save_pretrained(output_path)

    readme = f"README.{output_path}.md" if f"README.{output_path}.md" in os.listdir(model_path) else "README.md"
    shutil.copy2(os.path.join(model_path, readme), os.path.join(output_path, "README.md"))
    customize_readme_for_variant(os.path.join(output_path, "README.md"), output_path, default_variant)
    update_readme(os.path.join(output_path, "README.md"), output_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for license_name in ("license.md", "license-faq.md"):
        license_path = os.path.join(repo_root, license_name)
        if os.path.exists(license_path):
            shutil.copy2(license_path, os.path.join(output_path, license_name))
        else:
            warn(f"License file not found: {license_path}")


def update_readme(readme_path: str, model: str) -> None:
    post = fm.load(readme_path)
    pipeline_tag = post.get("pipeline_tag")
    if pipeline_tag is None:
        return
    if pipeline_tag == "other":
        pipeline_tag = post["pipeline"]
    ppl = pipeline(pipeline_tag, model=model)
    ref_sequences: Dict[str, str] = {}
    ref_meta: Dict[str, Dict[str, object]] = {}
    reference_tags = list(post.get("tags") or [])
    if post.get("language"):
        reference_tags.append(post["language"])
    for tag in reference_tags:
        for sequence_type, sequences in iter_reference_sequences(str(tag)):
            if not sequences:
                continue
            for name, sequence in sequences.items():
                ref_sequences.setdefault(name, sequence)
                ref_meta.setdefault(name, {"sequence_type": sequence_type})
    if not ref_sequences:
        raise ValueError(f"No reference sequences found from model card tags/language: {reference_tags!r}")
    post["widget"] = []
    for name, sequence in tqdm(ref_sequences.items(), total=len(ref_sequences), desc="Generating widget data"):
        prepared_sequence, mask_meta = prepare_sequence(ppl, sequence)
        output = run_pipeline(ppl, prepared_sequence)
        widget_entry = {
            "example_title": name,
            "text": prepared_sequence,
            "pipeline_tag": pipeline_tag,
            "task": ppl.task,
        }
        if output is not None:
            widget_entry["output"] = output
        widget_entry.update(ref_meta.get(name, {}))
        widget_entry.update(mask_meta)
        post["widget"].append(widget_entry)
    fm.dump(post, readme_path)


def prepare_sequence(ppl: Pipeline, sequence: str) -> tuple[str, Dict[str, object]]:
    if ppl.task == "text-generation":
        return sequence[:-text_generation_tokens], {"expected_suffix": sequence[-text_generation_tokens:]}
    if ppl.task == "fill-mask":
        mask_token = getattr(ppl.tokenizer, "mask_token", None) or "<mask>"
        nmers = getattr(ppl.tokenizer, "nmers", 1) or 1
        is_codon = bool(getattr(ppl.tokenizer, "codon", False))

        if is_codon:
            span, step = 3, 3
            sequence = sequence[: len(sequence) - len(sequence) % span]
            if len(sequence) < span:
                return sequence, build_mask_meta(sequence, None)
        elif nmers > 1:
            span, step = nmers, 1
        else:
            span, step = 1, 1

        search_start = 10 if len(sequence) > 10 else 0
        if step > 1 and search_start % step != 0:
            search_start = ((search_start + step - 1) // step) * step
        search_start = min(search_start, max(len(sequence) - span, 0))

        start = None
        for i in range(search_start, len(sequence) - span + 1, step):
            if sequence[i] == "A":
                start = i
                break
        if start is None:
            start = search_start
        prepared = sequence[:start] + mask_token + sequence[start + span :]
        return prepared, build_mask_meta(sequence, start)
    return sequence, {}


def build_mask_meta(sequence: str, pos: int | None) -> Dict[str, object]:
    if pos is None:
        return {
            "mask_index": None,
            "mask_index_1based": None,
            "masked_char": None,
        }
    masked_char = sequence[pos] if sequence and 0 <= pos < len(sequence) else None
    return {
        "mask_index": pos,
        "mask_index_1based": pos + 1,
        "masked_char": masked_char,
    }


def run_pipeline(ppl: Pipeline, sequence: str) -> List[Dict] | Dict | None:
    tokenizer_kwargs = {"truncation": True}
    if ppl.task == "fill-mask":
        return [
            {"label": i["token_str"], "score": round(i["score"], 6)}
            for i in ppl(sequence, tokenizer_kwargs=tokenizer_kwargs)
        ]
    if ppl.task == "text-generation":
        result = ppl(sequence, max_new_tokens=text_generation_tokens, truncation=True)
        return {"text": result[0]["generated_text"]}
    if ppl.task == "rna-secondary-structure":
        return {"text": ppl(sequence, tokenizer_kwargs=tokenizer_kwargs)["secondary_structure"]}
    if ppl.task == "feature-extraction":
        # Embedding tensors are too large for a README and the Hub runs feature-extraction
        # widgets live; emit no pre-rendered output so the widget always reflects the model.
        return None
    # Custom MultiMolecule pipelines can return structured, model-specific payloads.
    # The Hub can run them live, so avoid blocking checkpoint conversion on
    # pre-rendering widget outputs here.
    return None


def push_to_hub(convert_config: ConvertConfig, output_path: str, repo_type: str = "model"):
    if convert_config.push_to_hub:
        if HfApi is None:
            raise ImportError("Please install huggingface_hub to push to the hub.")
        api = HfApi()
        repo_id = convert_config.repo_id
        token = convert_config.token
        if convert_config.delete_existing:
            api.delete_repo(repo_id=repo_id, repo_type=repo_type, token=token, missing_ok=True)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type=repo_type, token=token, folder_path=output_path)


def _load_golden_case(golden_root: Path, output_path: str) -> tuple[Path, dict[str, Any]]:
    model_root = golden_root / "models"
    matches = sorted(model_root.glob(f"*/{output_path}/meta.json"))
    if not matches:
        raise FileNotFoundError(f"No golden fixture found for converted checkpoint {output_path!r} under {model_root}")
    if len(matches) != 1:
        joined = ", ".join(str(path.parent) for path in matches)
        raise RuntimeError(f"Ambiguous golden fixtures for {output_path!r}: {joined}")
    meta_path = matches[0]
    return meta_path.parent, json.loads(meta_path.read_text())


def _golden_model_loaders() -> dict[str, Any]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForPreTraining

    import multimolecule.models  # noqa: F401
    from multimolecule.models.modeling_auto import (
        AutoModelForProfilePrediction,
        AutoModelForRnaSecondaryStructurePrediction,
        AutoModelForSequencePrediction,
        AutoModelForTokenPrediction,
    )
    from multimolecule.models.ribonanzanet import (
        RibonanzaNetForDegradationPrediction,
        RibonanzaNetForSequenceDropoutPrediction,
    )

    return {
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForPreTraining": AutoModelForPreTraining,
        "AutoModelForProfilePrediction": AutoModelForProfilePrediction,
        "AutoModelForRnaSecondaryStructurePrediction": AutoModelForRnaSecondaryStructurePrediction,
        "AutoModelForSequencePrediction": AutoModelForSequencePrediction,
        "AutoModelForTokenPrediction": AutoModelForTokenPrediction,
        "RibonanzaNetForDegradationPrediction": RibonanzaNetForDegradationPrediction,
        "RibonanzaNetForSequenceDropoutPrediction": RibonanzaNetForSequenceDropoutPrediction,
    }


def _golden_output_tensor(outputs: Any, key: str, model: nn.Module) -> torch.Tensor:
    if key == "vocab_embeddings":
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            raise AssertionError("Model has no input embeddings for golden output 'vocab_embeddings'")
        return embeddings.weight

    if key == "coverage":
        # `coverage` is the post-activation (e.g. softplus) prediction. The model exposes raw pre-activation
        # values under `logits`; the activated coverage is produced by `postprocess`, which returns a
        # `(coverage, channels)` tuple. Validate against that activated output, not the raw `logits`.
        if not hasattr(model, "postprocess"):
            raise AssertionError("Model has no postprocess() for golden output 'coverage'")
        processed = model.postprocess(outputs)
        value = processed[0] if isinstance(processed, tuple) else processed
    else:
        value = outputs[key] if isinstance(outputs, dict) else getattr(outputs, key)
    if value is None:
        raise AssertionError(f"Model output {key!r} is None")
    if isinstance(value, (tuple, list)):
        if not value:
            raise AssertionError(f"Model output {key!r} is empty")
        value = torch.stack(tuple(value), dim=0)
    if not isinstance(value, torch.Tensor):
        raise AssertionError(f"Model output {key!r} is {type(value).__name__}, not a tensor")
    return value


def _select_golden_output_subset(
    key: str,
    actual: torch.Tensor,
    target: torch.Tensor,
    meta: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.shape == target.shape:
        return actual, target
    is_hidden_states = key == "hidden_states"
    same_rank = actual.ndim == target.ndim
    has_extra_embedding_state = actual.shape[0] == target.shape[0] + 1
    trailing_shape_matches = actual.shape[1:] == target.shape[1:]
    if is_hidden_states and same_rank and has_extra_embedding_state and trailing_shape_matches:
        return actual[1:], target
    upstream = meta.get("upstream", {})
    if not isinstance(upstream, dict):
        return actual, target
    target_slice = upstream.get("target_slice")
    if isinstance(target_slice, list) and all(isinstance(index, int) for index in target_slice):
        if key.startswith("logits") or key == "coverage":
            sliced = actual[..., target_slice]
            return (sliced, target) if sliced.shape == target.shape else (actual, target)
        if key == "vocab_embeddings":
            sliced = actual[target_slice, ...]
            return (sliced, target) if sliced.shape == target.shape else (actual, target)
    return actual, target


def validate_checkpoint_with_golden(convert_config: ConvertConfig, output_path: str) -> None:
    golden_root = Path(convert_config.golden_root).expanduser().resolve()
    case_dir, meta = _load_golden_case(golden_root, os.path.basename(output_path.rstrip(os.sep)))
    tolerance = meta["tolerance"]
    expected = load_file(str(case_dir / "expected.safetensors"))
    inputs = load_file(str(case_dir / "inputs.safetensors"))

    auto_model = str(meta.get("auto_model", "AutoModel"))
    loader = _golden_model_loaders().get(auto_model)
    if loader is None:
        raise ValueError(f"{case_dir / 'meta.json'}: unsupported auto_model {auto_model!r}")

    model = loader.from_pretrained(output_path).float()
    model.eval()

    kwargs: dict[str, Any] = dict(inputs)
    if "hidden_states" in expected:
        kwargs["output_hidden_states"] = True
    if "attentions" in expected:
        kwargs["output_attentions"] = True

    with torch.no_grad():
        actual_outputs = model(**kwargs, return_dict=True)

    print(f"Golden fixture: {case_dir}")
    for key in meta["outputs"]:
        actual = _golden_output_tensor(actual_outputs, key, model).detach().cpu()
        target = expected[key].detach().cpu()
        actual, target = _select_golden_output_subset(key, actual, target, meta)
        torch.testing.assert_close(
            actual,
            target,
            atol=float(tolerance["atol"]),
            rtol=float(tolerance["rtol"]),
            check_dtype=False,
        )
        diff = (actual - target).abs()
        print(f"  {key}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")


def save_checkpoint(convert_config: ConvertConfig, model: PreTrainedModel, tokenizer_config: Dict):
    model_path = convert_config.root
    output_path = convert_config.output_path
    if convert_config.delete_after_validate and not convert_config.validate_golden:
        raise ValueError("--delete_after_validate requires --validate_golden")
    if os.path.exists(output_path):
        warn(f"Output directory: {output_path} already exists. Deleting it.")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    try:
        write_model(
            model_path,
            output_path,
            model,
            tokenizer_config,
            default_variant=convert_config.default_variant,
        )
        if convert_config.validate_golden:
            validate_checkpoint_with_golden(convert_config, output_path)
        push_to_hub(convert_config, output_path)
    finally:
        if convert_config.delete_after_validate:
            shutil.rmtree(output_path, ignore_errors=True)
            print(f"Deleted converted checkpoint at {output_path}")


class ConvertConfig(Config):
    checkpoint_path: str
    root: str
    output_path: str
    default_variant: str | None = None
    push_to_hub: bool = False
    delete_existing: bool = False
    validate_golden: bool = False
    delete_after_validate: bool = False
    golden_root: str = str(Path(__file__).resolve().parents[2] / "golden")
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"
