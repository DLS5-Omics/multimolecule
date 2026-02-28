---
language: rna
tags:
  - Biology
  - RNA
  - ncRNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
  - multimolecule/rfam
  - multimolecule/ensembl-genome-browser
  - multimolecule/nucleotide
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# RiNALMo

Pre-trained model on non-coding RNA (ncRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks](https://doi.org/10.48550/arXiv.2403.00043) by Rafael Josip Penić, et al.

The OFFICIAL repository of RiNALMo is at [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RiNALMo did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RiNALMo is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/rinalmo-giga](https://huggingface.co/multimolecule/rinalmo-giga)**: The RiNALMo model with 650 million parameters.
- **[multimolecule/rinalmo-mega](https://huggingface.co/multimolecule/rinalmo-mega)**: The RiNALMo model with 150 million parameters.
- **[multimolecule/rinalmo-micro](https://huggingface.co/multimolecule/rinalmo-micro)**: The RiNALMo model with 30 million parameters.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variants</th>
    <th>Num Layers</th>
    <th>Hidden Size</th>
    <th>Num Heads</th>
    <th>Intermediate Size</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
    <th>Max Num Tokens</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>RiNALMo-Giga</td>
    <td>33</td>
    <td>1280</td>
    <td rowspan="3">20</td>
    <td>5120</td>
    <td>650.88</td>
    <td>709.59</td>
    <td>354.31</td>
    <td rowspan="3">1022</td>
  </tr>
  <tr>
    <td>RiNALMo-Mega</td>
    <td>30</td>
    <td>640</td>
    <td>2560</td>
    <td>148.04</td>
    <td>171.76</td>
    <td>85.54</td>
  </tr>
  <tr>
    <td>RiNALMo-Micro</td>
    <td>12</td>
    <td>480</td>
    <td>1920</td>
    <td>33.48</td>
    <td>40.26</td>
    <td>20.01</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.rinalmo](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rinalmo)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: [RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks](https://doi.org/10.48550/arXiv.2403.00043)
- **Developed by**: Rafael Josip Penić, Tin Vlašić, Roland G. Huber, Yue Wan, Mile Šikić
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo)

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

predictor = pipeline("fill-mask", model="multimolecule/rinalmo-giga")
output = predictor("gguc<mask>cucugguuagaccagaucugagccu")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RiNALMoModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-giga")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo-giga")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-giga")
model = RiNALMoForSequencePrediction.from_pretrained("multimolecule/rinalmo-giga")

text = "UAGCUUAUCAGACUGAUGUUG"
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
from multimolecule import RnaTokenizer, RiNALMoForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-giga")
model = RiNALMoForTokenPrediction.from_pretrained("multimolecule/rinalmo-giga")

text = "UAGCUUAUCAGACUGAUGUUG"
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
from multimolecule import RnaTokenizer, RiNALMoForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-giga")
model = RiNALMoForContactPrediction.from_pretrained("multimolecule/rinalmo-giga")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RiNALMo used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RiNALMo model was pre-trained on a cocktail of databases including [RNAcentral](https://rnacentral.org), [Rfam](https://rfam.org), [Ensembl Genome Browser](https://ensembl.org), and [Nucleotide](https://ncbi.nlm.nih.gov/nucleotide).
The training data contains 36 million unique ncRNA sequences.

To ensure sequence diversity in each training batch, RiNALMo clustered the sequences with [MMSeqs2](https://github.com/soedinglab/MMseqs2) into 17 million clusters and then sampled each sequence in the batch from a different cluster.

RiNALMo preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RiNALMo used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 7 NVIDIA A100 GPUs with 80GiB memories.

- Batch Size: 1344
- Epochs: 6
- Learning rate: 5e-5
- Learning rate scheduler: Cosine
- Learning rate warm-up: 2,000 steps
- Learning rate minimum: 1e-5
- Dropout: 0.1

## Citation

```bibtex
@ARTICLE{Penic2025-qf,
  title     = "{RiNALMo}: general-purpose {RNA} language models can generalize
               well on structure prediction tasks",
  author    = "Peni{\'c}, Rafael Josip and Vla{\v s}i{\'c}, Tin and Huber,
               Roland G and Wan, Yue and {\v S}iki{\'c}, Mile",
  abstract  = "While RNA has recently been recognized as an interesting
               small-molecule drug target, many challenges remain to be
               addressed before we take full advantage of it. This emphasizes
               the necessity to improve our understanding of its structures and
               functions. Over the years, sequencing technologies have produced
               an enormous amount of unlabeled RNA data, which hides a huge
               potential. Motivated by the successes of protein language
               models, we introduce RiboNucleic Acid Language Model (RiNALMo)
               to unveil the hidden code of RNA. RiNALMo is the largest RNA
               language model to date, with 650M parameters pre-trained on 36M
               non-coding RNA sequences from several databases. It can extract
               hidden knowledge and capture the underlying structure
               information implicitly embedded within the RNA sequences.
               RiNALMo achieves state-of-the-art results on several downstream
               tasks. Notably, we show that its generalization capabilities
               overcome the inability of other deep learning methods for
               secondary structure prediction to generalize on unseen RNA
               families.",
  journal   = "Nature Communications",
  publisher = "Springer Science and Business Media LLC",
  volume    =  16,
  number    =  1,
  pages     = "5671",
  month     =  jul,
  year      =  2025,
  copyright = "https://creativecommons.org/licenses/by-nc-nd/4.0",
  language  = "en"
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

Please contact the authors of the [RiNALMo paper](https://doi.org/10.48550/arXiv.2403.00043) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
