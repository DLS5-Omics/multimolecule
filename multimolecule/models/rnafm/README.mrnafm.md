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
  - example_title: "PRNP"
    text: "CTG<mask>AAGCGGCCCACGCGGACTGACGGGCGGGGG"
    output:
      - label: "GAG"
        score: 0.09500275552272797
      - label: "GGC"
        score: 0.09362148493528366
      - label: "AAG"
        score: 0.07337076216936111
      - label: "GAC"
        score: 0.07307938486337662
      - label: "GUG"
        score: 0.06616155058145523
---

# mRNA-FM

Pre-trained model on mRNA CoDing Sequence (CDS) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions](https://doi.org/10.1101/2022.08.06.503062) by Jiayang Chen, Zhihang Hue, Siqi Sun, et al.

The OFFICIAL repository of RNA-FM is at [ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM).

!!! Success "Reproducibility"

    The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RNA-FM did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RNA-FM is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variations

- **[`multimolecule/rnafm`](https://huggingface.co/multimolecule/rnafm)**: The RNA-FM model pre-trained on non-coding RNA sequences.
- **[`multimolecule/mrnafm`](https://huggingface.co/multimolecule/mrnafm)**: The RNA-FM model pre-trained on mRNA coding sequences.

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
- **Data**: [RNAcentral](https://rnacentral.org)
- **Paper**: [Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions](https://doi.org/10.1101/2022.08.06.503062)
- **Developed by**: Jiayang Chen, Zhihang Hu, Siqi Sun, Qingxiong Tan, Yixuan Wang, Qinze Yu, Licheng Zong, Liang Hong, Jin Xiao, Tao Shen, Irwin King, Yu Li
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ESM](https://huggingface.co/facebook/esm2_t48_15B_UR50D)
- **Original Repository**: [https://github.com/ml4bio/RNA-FM](https://github.com/ml4bio/RNA-FM)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='multimolecule/mrnafm')
>>> unmasker("ctg<mask>aagcggcccacgcggactgacgggcggggg")

[{'score': 0.09500275552272797,
  'token': 58,
  'token_str': 'GAG',
  'sequence': 'CUG GAG AAG CGG CCC ACG CGG ACU GAC GGG CGG GGG'},
 {'score': 0.09362148493528366,
  'token': 67,
  'token_str': 'GGC',
  'sequence': 'CUG GGC AAG CGG CCC ACG CGG ACU GAC GGG CGG GGG'},
 {'score': 0.07337076216936111,
  'token': 8,
  'token_str': 'AAG',
  'sequence': 'CUG AAG AAG CGG CCC ACG CGG ACU GAC GGG CGG GGG'},
 {'score': 0.07307938486337662,
  'token': 57,
  'token_str': 'GAC',
  'sequence': 'CUG GAC AAG CGG CCC ACG CGG ACU GAC GGG CGG GGG'},
 {'score': 0.06616155058145523,
  'token': 73,
  'token_str': 'GUG',
  'sequence': 'CUG GUG AAG CGG CCC ACG CGG ACU GAC GGG CGG GGG'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaFmModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/mrnafm')
model = RnaFmModel.from_pretrained('multimolecule/mrnafm')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RnaFmForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/mrnafm')
model = RnaFmForSequencePrediction.from_pretrained('multimolecule/mrnafm')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.tensor([1])

output = model(**input, labels=label)
```

#### Nucleotide Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for nucleotide classification or regression.

Here is how to use this model as backbone to fine-tune for a nucleotide-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RnaFmForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/mrnafm')
model = RnaFmForNucleotidePrediction.from_pretrained('multimolecule/mrnafm')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

#### Contact Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

Here is how to use this model as backbone to fine-tune for a contact-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RnaFmForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/mrnafm')
model = RnaFmForContactPrediction.from_pretrained('multimolecule/mrnafm')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RNA-FM used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RNA-FM model was pre-trained on [RNAcentral](https://rnacentral.org). RNAcentral is a comprehensive database of non-coding RNA sequences from a wide range of species. It combines 47 different databases, adding up to around 27 million RNA sequences in total.

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

#### PreTraining

The model was trained on 8 NVIDIA A100 GPUs with 80GiB memories.

- Learning rate: 1e-4
- Weight decay: 0.01
- Learning rate scheduler: inverse square root
- Learning rate warm-up: 10,000 steps

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
