---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
---

# GENERanno

Pre-trained model on genome sequences using a masked language modeling (MLM) objective at single-nucleotide resolution.

## Disclaimer

This is an UNOFFICIAL implementation of the [GENERanno: A Genomic Foundation Model for Metagenomic Annotation](https://doi.org/10.1101/2025.06.04.656517) by Wei Wu et al.

The OFFICIAL repository of GENERanno is at [GenerTeam/GENERanno](https://github.com/GenerTeam/GENERanno).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints produce the same intermediate representations as the original implementation, with a maximum absolute logit difference below `1e-4` on the matched vocabulary columns.

**The team releasing GENERanno did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

GENERanno is a LLaMA-style transformer encoder with rotary position embeddings, grouped-query attention, RMSNorm, and SwiGLU feed-forward blocks, pre-trained as a masked language model on large eukaryotic and prokaryotic genome corpora. Each nucleotide (A, C, G, T, N) is a single token, preserving single-nucleotide resolution for downstream annotation tasks. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/generanno-eukaryote](https://huggingface.co/multimolecule/generanno-eukaryote)**: GENERanno pre-trained on 386B base pairs of eukaryotic DNA.
- **[multimolecule/generanno-prokaryote](https://huggingface.co/multimolecule/generanno-prokaryote)**: GENERanno pre-trained on 715B base pairs of prokaryotic DNA.

### Model Specification

<table>
<thead>
  <tr>
    <th>Num Layers</th>
    <th>Hidden Size</th>
    <th>Num Heads</th>
    <th>Num KV Heads</th>
    <th>Intermediate Size</th>
    <th>Num Parameters (M)</th>
    <th>Max Num Tokens</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>28</td>
    <td>1280</td>
    <td>16</td>
    <td>4</td>
    <td>3520</td>
    <td>493</td>
    <td>8192</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.generanno](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/generanno)
- **Paper**: [GENERanno: A Genomic Foundation Model for Metagenomic Annotation](https://doi.org/10.1101/2025.06.04.656517)
- **Developed by**: Wei Wu, Qiuyi Li, Mingyang Li, Kun Fu, Fuli Feng, Jieping Ye, Hui Xiong, Zheng Wang
- **Model type**: LLaMA-style encoder with RoPE, GQA, RMSNorm, SwiGLU
- **Original Repository**: [GenerTeam/GENERanno-eukaryote-0.5b-base](https://huggingface.co/GenerTeam/GENERanno-eukaryote-0.5b-base), [GenerTeam/GENERanno-prokaryote-0.5b-base](https://huggingface.co/GenerTeam/GENERanno-prokaryote-0.5b-base)
- **Upstream license**: MIT

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Masked Language Modeling

You can use this model directly with a pipeline for masked language modeling:

```python
import multimolecule  # you must import multimolecule to register models
from transformers import pipeline

predictor = pipeline("fill-mask", model="multimolecule/generanno-eukaryote")
output = predictor("ATCG<mask>TGCA")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import GenerannoModel
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/generanno-eukaryote")
model = GenerannoModel.from_pretrained("multimolecule/generanno-eukaryote")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import GenerannoForSequencePrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/generanno-eukaryote")
model = GenerannoForSequencePrediction.from_pretrained("multimolecule/generanno-eukaryote")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.tensor([1])

output = model(**input, labels=label)
```

#### Token Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for token classification or regression.

Here is how to use this model as backbone to fine-tune for a nucleotide-level task in PyTorch:

```python
import torch
from multimolecule import GenerannoForTokenPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/generanno-eukaryote")
model = GenerannoForTokenPrediction.from_pretrained("multimolecule/generanno-eukaryote")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

#### Contact Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

Here is how to use this model as backbone to fine-tune for a contact-level task in PyTorch:

```python
import torch
from multimolecule import GenerannoForContactPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/generanno-eukaryote")
model = GenerannoForContactPrediction.from_pretrained("multimolecule/generanno-eukaryote")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

GENERanno used Masked Language Modeling (MLM) as the pre-training objective. Tokens are masked and the model is trained to predict the original nucleotides given the surrounding context. This is comparable to the Cloze task in natural language modeling.

### Training Data

- **eukaryote-0.5b**: 386 billion base pairs of eukaryotic DNA.
- **prokaryote-0.5b**: 715 billion base pairs of prokaryotic DNA.

The upstream tokenizer keeps each nucleotide as an individual token (single-nucleotide resolution) and includes a small set of domain-specific control tokens used during fine-tuning for gene annotation (e.g. `<cds>`, `<tRNA>`, `<rRNA>`). The MultiMolecule conversion remaps the underlying nucleotide embeddings into the shared MultiMolecule DNA vocabulary while preserving the trained backbone weights.

### Training Procedure

The model uses the LLaMA-style stack: RMSNorm pre-norm, SwiGLU feed-forward, rotary position embeddings with `theta = 5e5`, and grouped-query attention (16 query heads, 4 key/value heads). Context length is 8192 tokens.

## Citation

```bibtex
@article{wu2025generanno,
  title   = {{GENERanno}: A Genomic Foundation Model for Metagenomic Annotation},
  author  = {Wu, Wei and Li, Qiuyi and Li, Mingyang and Fu, Kun and Feng, Fuli and Ye, Jieping and Xiong, Hui and Wang, Zheng},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.04.656517},
  url     = {https://www.biorxiv.org/content/10.1101/2025.06.04.656517v2}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If you use MultiMolecule in your research, you must cite the MultiMolecule project as follows:

```bibtex
@software{chen_2024_12638419,
  author    = {Chen, Zhiyuan and Zhu, Sophia Y.},
  title     = {MultiMolecule},
  doi       = {10.5281/zenodo.12638419},
  publisher = {Zenodo},
  url       = {https://doi.org/10.5281/zenodo.12638419},
  year      = 2024,
  month     = may,
  day       = 4
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [GENERanno paper](https://doi.org/10.1101/2025.06.04.656517) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
