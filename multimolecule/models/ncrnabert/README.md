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
  - example_title: "HIV-1"
    text: "GGUC<mask>CUCUGGUUAGACCAGAUCUGAGCCU"
    output:
      - label: "<eos>"
        score: 0.19942431151866913
      - label: "I"
        score: 0.1465310901403427
      - label: "*"
        score: 0.1448192000389099
      - label: "<unk>"
        score: 0.14174020290374756
      - label: "<cls>"
        score: 0.13194777071475983
  - example_title: "microRNA-21"
    text: "UAGC<mask>UAUCAGACUGAUGUUG"
    output:
      - label: "<eos>"
        score: 0.19946657121181488
      - label: "I"
        score: 0.14641942083835602
      - label: "*"
        score: 0.14452320337295532
      - label: "<unk>"
        score: 0.14180712401866913
      - label: "<cls>"
        score: 0.13223469257354736
---

# ncRNABert

Pre-trained model on non-coding RNA (ncRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the ncRNABert: Deciphering the landscape of non-coding RNA using language model by Lei Wang, Xudong Li, et al.

The OFFICIAL repository of ncRNABert is at [wangleiofficial/ncRNABert](https://github.com/wangleiofficial/ncRNABert).

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of ncRNABert.
>
> The ncRNABert apply `softmax` in the `-2` dimension when computing the attention probs. This makes the output of `attention_probs @ value_layer` unreliable when the input sequences are not of the same length (i.e., have padding tokens).
> MultiMolecule applied a workaround to ensure that the attention masks are applied correctly, but this may lead to different results compared to the original implementation.

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of ncRNABert does not prepend `<bos>` (`<cls>`) and append `<eos>` tokens to the input sequence.
> This should not affect the performance of the model in most cases, but it can lead to unexpected behavior in some cases.
>
> Please set `bos_token=None, eos_token=None` in the tokenizer and set `bos_token_id=None, eos_token_id=None` in the model configuration if you want the exact behavior of the original implementation.

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ncRNABert did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ncRNABert is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of 5’ untranslated regions (5’UTRs) in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/ncrnabert](https://huggingface.co/multimolecule/ncrnabert)**: The ncRNABert model pre-trained on single nucleotide data.
- **[multimolecule/ncrnabert-3mer](https://huggingface.co/multimolecule/ncrnabert-3mer)**: The ncRNABert model pre-trained on 3-mer data.

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
    <td>ncRNABert</td>
    <td rowspan="2">24</td>
    <td rowspan="2">1024</td>
    <td rowspan="2">16</td>
    <td rowspan="2">4096</td>
    <td rowspan="2">303.31</td>
    <td rowspan="2">78.96</td>
    <td rowspan="2">39.46</td>
    <td rowspan="2">512</td>
  </tr>
  <tr>
    <td>ncRNABert-3mer</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.ncrnabert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ncrnabert)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: ncRNABert: Deciphering the landscape of non-coding RNA using language model
- **Developed by**: Lei Wang, Xudong Li, Zhidong Xue, Yan Wang
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [wangleiofficial/ncRNABert](https://github.com/wangleiofficial/ncRNABert)

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

>>> unmasker = pipeline("fill-mask", model="multimolecule/ncrnabert")
>>> unmasker("gguc<mask>cucugguuagaccagaucugagccu")
[{'score': 0.19942431151866913,
  'token': 2,
  'token_str': '<eos>',
  'sequence': 'G G U C C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.1465310901403427,
  'token': 25,
  'token_str': 'I',
  'sequence': 'G G U C I C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.1448192000389099,
  'token': 23,
  'token_str': '*',
  'sequence': 'G G U C * C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.14174020290374756,
  'token': 3,
  'token_str': '<unk>',
  'sequence': 'G G U C C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.13194777071475983,
  'token': 1,
  'token_str': '<cls>',
  'sequence': 'G G U C C U C U G G U U A G A C C A G A U C U G A G C C U'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, NcRnaBertModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ncrnabert")
model = NcRnaBertModel.from_pretrained("multimolecule/ncrnabert")

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
from multimolecule import RnaTokenizer, NcRnaBertForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ncrnabert")
model = NcRnaBertForSequencePrediction.from_pretrained("multimolecule/ncrnabert")

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
from multimolecule import RnaTokenizer, NcRnaBertForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ncrnabert")
model = NcRnaBertForTokenPrediction.from_pretrained("multimolecule/ncrnabert")

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
from multimolecule import RnaTokenizer, NcRnaBertForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ncrnabert")
model = NcRnaBertForContactPrediction.from_pretrained("multimolecule/ncrnabert")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

ncRNABert used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The ncRNABert model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

### Training Procedure

#### Preprocessing

ncRNABert used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 8 NVIDIA A40 GPUs with 48GiB memories.

- Steps: 1,000,000
- Optimizer: AdamW
- Learning rate: 1e-5
- Learning rate warm-up: 1,000 steps
- Learning rate scheduler: Cosine
- Weight decay: 0.01

## Citation

**BibTeX**:

```bibtex
@article{wang2025ncrnabert,
  author={Wang, Lei and Li, Xudong and Xue, Zhidong and Wang, Yan},
  title={ncRNABert: Deciphering the landscape of non-coding RNA using language model},
  journal={Under Review},
  year={2025}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the ncRNABert paper for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
