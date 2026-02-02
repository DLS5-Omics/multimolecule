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
import shutil
from typing import Dict, List
from warnings import warn

import frontmatter as fm
import torch
from chanfig import Config
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, pipeline
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from transformers.pipelines.base import Pipeline

from multimolecule.tokenisers.rna.utils import get_tokenizer_config

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

reference_sequences = {
    "cDNA": {
        "prion protein (Kanno blood group)": "ATGGCGAACCTTGGCTGCTGGATGCTGGTTCTCTTTGTGGCCACATGGAGTGACCTGGGCCTCTGC",  # https://www.ncbi.nlm.nih.gov/nuccore/M13899.1?from=50&to=115  # noqa: E501
        "interleukin 10": "ATGCACAGCTCAGCACTGCTCTGTTGCCTGGTCCTCCTGACTGGGGTGAGGGCC",  # https://www.ncbi.nlm.nih.gov/nuccore/M57627.1?from=31&to=84  # noqa: E501
        "Zaire ebolavirus": "AATGTTCAAACACTTTGTGAAGCTCTGTTAGCTGATGGTCTTGCTAAAGCATTTCCTAGCAATATGATGGTAGTCACAGAGCGTGAGCAAAAAGAAAGCTTATTGCATCAAGCATCATGGCACCACACAAGTGATGATTTTGGTGAGCATGCCACAGTTAGAGGGAGTAGCTTTGTAACTGATTTAGAGAAATACAATCTTGCATTTAGATATGAGTTTACAGCACCTTTTATAGAATATTGTAACCGTTGCTATGGTGTTAAGAATGTTTTTAATTGGATGCATTATACAATCCCACAGTGTTAT",  # https://www.ncbi.nlm.nih.gov/nuccore/DQ211657.1?from=2&to=307  # noqa: E501
        "SARS coronavirus": "ATGTTTATTTTCTTATTATTTCTTACTCTCACTAGTGGTAGTGACCTTGACCGGTGCACCACTTTTGATGATGTTCAAGCTCCTAATTACACTCAACATACTTCATCTATGAGGGGGGTTTACTATCCTGATGAAATTTTTAGATCAGACACTCTTTATTTAACTCAGGATTTATTTCTTCCATTTTATTCTAATGTTACAGGGTTTCATACTATTAATCATACGTTTGACAACCCTGTCATACCTTTTAAGGATGGTATTTATTTTGCTGCCACAGAGAAATCAAATGTTGTCCGTGGTTGGGTTTTTGGTTCTACCATGAACAACAAGTCACAGTCGGTGATTATTATTAACAATTCTACTAATGTTGTTATACGAGCATGTAACTTTGAATTGTGTGACAACCCTTTCTTTGCTGTTTCTAAACCCATGGGTACACAGACACATACTATGATATTCGATAATGCATTTAAATGCACTTTCGAGTACATATCT",  # https://www.ncbi.nlm.nih.gov/nuccore/AY536757.3?from=73&to=567  # noqa: E501
    },
    "ncRNA": {
        "microRNA 21": "UAGCUUAUCAGACUGAUGUUGA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_029493.1?from=8&to=29
        "microRNA 146a": "UGAGAACUGAAUUCCAUGGGUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_029701.1?from=21&to=42
        "microRNA 155": "UUAAUGCUAAUCGUGAUAGGGGUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_030784.1?from=4&to=27
        "metastasis associated lung adenocarcinoma transcript 1": "AGGCAUUGAGGCAGCCAGCGCAGGGGCUUCUGCUGAGGGGGCAGGCGGAGCUUGAGGAAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_002819.5?from=1&to=60  # noqa: E501
        "Pvt1 oncogene": "CCCGCGCUCCUCCGGGCAGAGCGCGUGUGGCGGCCGAGCACAUGGGCCCGCGGGCCGGGC",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_003367.4?from=1&to=60  # noqa: E501
        "telomerase RNA component": "GGGUUGCGGAGGGUGGGCCUGGGAGGGGUGGUGGCCAUUUUUUGUCUAACCCUAACUGAG",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_001566.3?from=1&to=60  # noqa: E501
        "vault RNA 2-1": "CGGGUCGGAGUUAGCUCAAGCGGUUACCUCCUCAUGCCGGACUUUCUAUCUGUCCAUCUCUGUGCUGGGGUUCGAGACCCGCGGGUGCUUACUGACCCUUUUAUGCAA",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_030583.3  # noqa: E501
        "brain cytoplasmic RNA 1": "GGCCGGGCGCGGUGGCUCACGCCUGUAAUCCCAGCUCUCAGGGAGGCUAAGAGGCGGGAGGAUAGCUUGAGCCCAGGAGUUCGAGACCUGCCUGGGCAAUAUAGCGAGACCCCGUUCUCCAGAAAAAGGAAAAAAAAAAACAAAAGACAAAAAAAAAAUAAGCGUAACUUCCCUCAAAGCAACAACCCCCCCCCCCCUUU",  # https://www.ncbi.nlm.nih.gov/nuccore/NR_001568.1  # noqa: E501
        "HIV-1 TAR-WT": "GGUCUCUCUGGUUAGACCAGAUCUGAGCCUGGGAGCUCUCUGGCUAACUAGGGAACC",  # https://pmc.ncbi.nlm.nih.gov/articles/PMC1955452/  # noqa: E501
    },
    "5' UTR": {
        "interleukin 10": "CUUUUUAAUGAAUGAAGAGGCCUCCCUGAGCUUACAAUAUAAAAGGGGGACAGAGAGGUG",  # https://www.ncbi.nlm.nih.gov/nuccore/EU751618.2?from=1&to=126  # noqa: E501
    },
    "3' UTR": {
        "Human GPI protein p137": "UUUUUAAAAGGAAAAGAUACCAAAUGCCUGCUGCUACCACCCUUUUCAAUUGCUAUGUUU",  # https://www.ncbi.nlm.nih.gov/nuccore/U51714.1?from=1&to=60  # noqa: E501
    },
    "Protein": {
        "prion protein (Kanno blood group)": "MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSDYEDRYYRENMHRYPNQVYYRPMDEYSNQNNFVHDCVNITIKQHTVTTTTKGENFTETDVKMMERVVEQMCITQYERESQAYYQRGSSMVLFSSPPVILLISFLIFLIVG",  # https://www.ncbi.nlm.nih.gov/nuccore/M13899.1  # noqa: E501
        "interleukin 10": "MHSSALLCCLVLLTGVRASPGQGTQSENSCTHFPGNLPNMLRDLRDAFSRVKTFFQMKDQLDNLLLKESLLEDFKGYLGCQALSEMIQFYLEEVMPQAENQDPDIKAHVNSLGENLKTLRLRLRRCHRFLPCENKSKAVEQVKNAFNKLQEKGIYKAMSEFDIFINYIEAYMTMKIRN",  # https://www.ncbi.nlm.nih.gov/nuccore/M57627.1  # noqa: E501
        "Zaire ebolavirus": "NVQTLCEALLADGLAKAFPSNMMVVTEREQKESLLHQASWHHTSDDFGEHATVRGSSFVTDLEKYNLAFRYEFTAPFIEYCNRCYGVKNVFNWMHYTIPQCY",  # https://www.ncbi.nlm.nih.gov/nuccore/DQ211657.1  # noqa: E501
        "SARS coronavirus": "MFIFLLFLTLTSGSDLDRCTTFDDVQAPNYTQHTSSMRGVYYPDEIFRSDTLYLTQDLFLPFYSNVTGFHTINHTFDNPVIPFKDGIYFAATEKSNVVRGWVFGSTMNNKSQSVIIINNSTNVVIRACNFELCDNPFFAVSKPMGTQTHTMIFDNAFKCTFEYIS",  # https://www.ncbi.nlm.nih.gov/nuccore/AY536757.3  # noqa: E501
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


def write_model(
    model_path: str,
    output_path: str,
    model: PreTrainedModel,
    tokenizer_config: Dict | None = None,
):
    model.save_pretrained(output_path, safe_serialization=True)
    model.save_pretrained(output_path, safe_serialization=False)
    if tokenizer_config is None:
        tokenizer_config = get_tokenizer_config()
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
    for sequence_type, sequences in iter_reference_sequences(post["tags"][-1]):
        if not sequences:
            continue
        for name, sequence in sequences.items():
            ref_sequences[name] = sequence
            ref_meta[name] = {"sequence_type": sequence_type}
    if not ref_sequences:
        raise ValueError(f"Sequence type '{post['pipeline_tag']}' not found in reference sequences.")
    post["widget"] = []
    for name, sequence in tqdm(ref_sequences.items(), total=len(ref_sequences), desc="Generating widget data"):
        prepared_sequence, mask_meta = prepare_sequence(ppl, sequence)
        output = run_pipeline(ppl, prepared_sequence)
        widget_entry = {
            "example_title": name,
            "text": prepared_sequence,
            "output": output,
            "pipeline_tag": pipeline_tag,
            "task": ppl.task,
        }
        widget_entry.update(ref_meta.get(name, {}))
        widget_entry.update(mask_meta)
        post["widget"].append(widget_entry)
    fm.dump(post, readme_path)


def prepare_sequence(ppl: Pipeline, sequence: str) -> tuple[str, Dict[str, object]]:
    if ppl.task != "fill-mask":
        return sequence, {}
    mask_token = getattr(ppl.tokenizer, "mask_token", None) or "<mask>"
    nmers = getattr(ppl.tokenizer, "nmers", 1) or 1
    is_codon = bool(getattr(ppl.tokenizer, "codon", False))

    if is_codon:
        span, step = 3, 3
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


def run_pipeline(ppl: Pipeline, sequence: str) -> List[Dict] | Dict:
    tokenizer_kwargs = {"truncation": True}
    if ppl.task == "fill-mask":
        return [
            {"label": i["token_str"], "score": round(i["score"], 6)}
            for i in ppl(sequence, tokenizer_kwargs=tokenizer_kwargs)
        ]
    if ppl.task == "rna-secondary-structure":
        return {"text": ppl(sequence, tokenizer_kwargs=tokenizer_kwargs)["secondary_structure"]}
    raise RecursionError(f"Pipeline {ppl.task} is not supported")


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


def save_checkpoint(
    convert_config: ConvertConfig,
    model: PreTrainedModel,
    tokenizer_config: Dict | None = None,
):
    model_path, output_path = convert_config.root, convert_config.output_path
    if os.path.exists(output_path):
        warn(f"Output directory: {output_path} already exists. Deleting it.")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    write_model(model_path, output_path, model, tokenizer_config)
    push_to_hub(convert_config, output_path)


class ConvertConfig(Config):
    checkpoint_path: str
    root: str
    output_path: str
    push_to_hub: bool = False
    delete_existing: bool = False
    repo_id: str | None = None
    token: str | None = None

    def post(self):
        if self.repo_id is None:
            self.repo_id = f"multimolecule/{self.output_path}"
