---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
widget:
  - example_title: "Homo sapiens PRNP mRNA for prion"
    text: "AGC<mask>CAUUAUGGCGAACCUUGGCUGCUG"
    output:
      - label: "AAA"
        score: 0.05433480441570282
      - label: "AUC"
        score: 0.04437034949660301
      - label: "AAU"
        score: 0.03882088139653206
      - label: "ACA"
        score: 0.037016965448856354
      - label: "ACC"
        score: 0.03563101962208748
---

# mRNA-FM

Pre-trained model on mRNA CoDing Sequence (CDS) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions](https://doi.org/10.1101/2022.08.06.503062) by Jiayang Chen, Zhihang Hue, Siqi Sun, et al.

The OFFICIAL repository of RNA-FM is at [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RNA-FM did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RNA-FM is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm)**: The RNA-FM model pre-trained on non-coding RNA sequences.
- **[multimolecule/mrnafm](https://huggingface.co/multimolecule/mrnafm)**: The RNA-FM model pre-trained on messenger RNA sequences.

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
    <td>RNA-FM</td>
    <td rowspan="2">12</td>
    <td>640</td>
    <td rowspan="2">20</td>
    <td rowspan="2">5120</td>
    <td>99.52</td>
    <td>25.68</td>
    <td>12.83</td>
    <td rowspan="2">1024</td>
  </tr>
  <tr>
    <td>mRNA-FM</td>
    <td>1280</td>
    <td>239.25</td>
    <td>61.43</td>
    <td>30.7</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.rnafm](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rnafm)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: [Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions](https://doi.org/10.1101/2022.08.06.503062)
- **Developed by**: Jiayang Chen, Zhihang Hu, Siqi Sun, Qingxiong Tan, Yixuan Wang, Qinze Yu, Licheng Zong, Liang Hong, Jin Xiao, Tao Shen, Irwin King, Yu Li
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ESM](https://huggingface.co/facebook/esm2_t48_15B_UR50D)
- **Original Repository**: [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Masked Language Modeling

You can use this model directly with a pipeline for masked language modeling:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline

>>> unmasker = pipeline("fill-mask", model="multimolecule/mrnafm")
>>> unmasker("agc<mask>cauuauggcgaaccuuggcugcug")
[{'score': 0.05433480441570282,
  'token': 6,
  'token_str': 'AAA',
  'sequence': 'AGC AAA CAU UAU GGC GAA CCU UGG CUG CUG'},
 {'score': 0.04437034949660301,
  'token': 22,
  'token_str': 'AUC',
  'sequence': 'AGC AUC CAU UAU GGC GAA CCU UGG CUG CUG'},
 {'score': 0.03882088139653206,
  'token': 9,
  'token_str': 'AAU',
  'sequence': 'AGC AAU CAU UAU GGC GAA CCU UGG CUG CUG'},
 {'score': 0.037016965448856354,
  'token': 11,
  'token_str': 'ACA',
  'sequence': 'AGC ACA CAU UAU GGC GAA CCU UGG CUG CUG'},
 {'score': 0.03563101962208748,
  'token': 12,
  'token_str': 'ACC',
  'sequence': 'AGC ACC CAU UAU GGC GAA CCU UGG CUG CUG'}]
```

#### RNA Secondary Structure Prediction

You can use this model to predict the secondary structure of an RNA sequence:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline

>>> predictor = pipeline("rna-secondary-structure", model="multimolecule/mrnafm")
>>> predictor("agcagucauuauggcgaa")
{'sequence': 'AGC AGU CAU UAU GGC GAA',
 'secondary_structure': '((([(]',
 'contact_map': [[0.5119704604148865, 0.5045265555381775, 0.494497150182724, 0.4931190013885498, 0.4915284812450409, 0.5020371675491333],
  [0.5045265555381775, 0.5034880042076111, 0.5013145804405212, 0.49390116333961487, 0.5006486773490906, 0.49380120635032654],
  [0.494497150182724, 0.5013145804405212, 0.5010323524475098, 0.5058367252349854, 0.5021511912345886, 0.49284809827804565],
  [0.4931190013885498, 0.49390116333961487, 0.5058367252349854, 0.4988723397254944, 0.5004245042800903, 0.5055262446403503],
  [0.4915284812450409, 0.5006486773490906, 0.5021511912345886, 0.5004245042800903, 0.4953134059906006, 0.5076138377189636],
  [0.5020371675491333, 0.49380120635032654, 0.49284809827804565, 0.5055262446403503, 0.5076138377189636, 0.4958533048629761]]}
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaFmModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
model = RnaFmModel.from_pretrained("multimolecule/mrnafm")

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
from multimolecule import RnaTokenizer, RnaFmForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
model = RnaFmForSequencePrediction.from_pretrained("multimolecule/mrnafm")

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
from multimolecule import RnaTokenizer, RnaFmForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
model = RnaFmForTokenPrediction.from_pretrained("multimolecule/mrnafm")

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
from multimolecule import RnaTokenizer, RnaFmForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/mrnafm")
model = RnaFmForContactPrediction.from_pretrained("multimolecule/mrnafm")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RNA-FM used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RNA-FM model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

RNA-FM applied [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) with a cut-off at 100% sequence identity to remove redundancy from the RNAcentral. The final dataset contains 23.7 million non-redundant RNA sequences.

RNA-FM preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RNA-FM used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 8 NVIDIA A100 GPUs with 80GiB memories.

- Learning rate: 1e-4
- Learning rate scheduler: Inverse square root
- Learning rate warm-up: 10,000 steps
- Weight decay: 0.01

## Citation

**BibTeX**:

```bibtex
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [RNA-FM paper](https://doi.org/10.1101/2022.08.06.503062) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
