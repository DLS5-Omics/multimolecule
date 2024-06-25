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
  - example_title: "microRNA-21"
    text: "UAGC<mask>UAUCAGACUGAUGUUGA"
    output:
      - label: "G"
        score: 0.09372635930776596
      - label: "R"
        score: 0.08816102892160416
      - label: "A"
        score: 0.08292599022388458
      - label: "<eos>"
        score: 0.07841548323631287
      - label: "V"
        score: 0.073448047041893
---

# RNAErnie

Pre-trained model on non-coding RNA (ncRNA) using a multi-stage masked language modeling (MLM) objective.

## Statement

_Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning_ is published in [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00836-4), which is a Closed Access / Author-Fee journal.

> Machine learning has been at the forefront of the movement for free and open access to research.
>
> We see no role for closed access or author-fee publication in the future of machine learning research and believe the adoption of these journals as an outlet of record for the machine learning community would be a retrograde step.

The MultiMolecule team is committed to the principles of open access and open science.

We do NOT endorse the publication of manuscripts in Closed Access / Author-Fee journals and encourage the community to support Open Access journals and conferences.

Please consider signing the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu).

## Disclaimer

This is an UNOFFICIAL implementation of the RNAErnie: An RNA Language Model with Structure-enhanced Representations by Ning Wang, Jiang Bian,
Haoyi Xiong, et al.

The OFFICIAL repository of RNAErnie is at [CatIIIIIIII/RNAErnie](https://github.com/CatIIIIIIII/RNAErnie).

> [!WARNING]
> The MultiMolecule team is unable to confirm that the provided model and checkpoints are producing the same intermediate representations as the original implementation.
> This is because
>
> The proposed method is published in a Closed Access / Author-Fee journal.

**The team releasing RNAErnie did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RNAErnie is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

Note that during the conversion process, additional tokens such as `[IND]` and ncRNA class symbols are removed.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 12         | 768         | 12        | 3072              | 86.06              | 22.36     | 11.17    | 512            |

### Links

- **Code**: [multimolecule.rnaernie](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rnaernie)
- **Weights**: [`multimolecule/rnaernie`](https://huggingface.co/multimolecule/rnaernie)
- **Data**: [RNAcentral](https://rnacentral.org)
- **Paper**: Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning
- **Developed by**: Ning Wang, Jiang Bian, Yuchen Li, Xuhong Li, Shahid Mumtaz, Linghe Kong, Haoyi Xiong.
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ERNIE](https://huggingface.co/nghuyong/ernie-3.0-base-zh)
- **Original Repository**: [https://github.com/CatIIIIIIII/RNAErnie](https://github.com/CatIIIIIIII/RNAErnie)

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
>>> unmasker = pipeline('fill-mask', model='multimolecule/rnaernie')
>>> unmasker("uagc<mask>uaucagacugauguuga")

[{'score': 0.09372635930776596,
  'token': 8,
  'token_str': 'G',
  'sequence': 'U A G C G U A U C A G A C U G A U G U U G A'},
 {'score': 0.08816102892160416,
  'token': 11,
  'token_str': 'R',
  'sequence': 'U A G C R U A U C A G A C U G A U G U U G A'},
 {'score': 0.08292599022388458,
  'token': 6,
  'token_str': 'A',
  'sequence': 'U A G C A U A U C A G A C U G A U G U U G A'},
 {'score': 0.07841548323631287,
  'token': 2,
  'token_str': '<eos>',
  'sequence': 'U A G C U A U C A G A C U G A U G U U G A'},
 {'score': 0.073448047041893,
  'token': 20,
  'token_str': 'V',
  'sequence': 'U A G C V U A U C A G A C U G A U G U U G A'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaErnieModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
model = RnaErnieModel.from_pretrained('multimolecule/rnaernie')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RnaErnieForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
model = RnaErnieForSequencePrediction.from_pretrained('multimolecule/rnaernie')

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
from multimolecule import RnaTokenizer, RnaErnieForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
model = RnaErnieForNucleotidePrediction.from_pretrained('multimolecule/rnaernie')

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
from multimolecule import RnaTokenizer, RnaErnieForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
model = RnaErnieForContactPrediction.from_pretrained('multimolecule/rnaernie')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RNAErnie used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RNAErnie model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral/).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

RNAErnie used a subset of RNAcentral for pre-training. The subset contains 23 million sequences.
RNAErnie preprocessed all tokens by replacing "T"s with "S"s.

Note that [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RNAErnie used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

RNAErnie uses a special 3-stage training pipeline to pre-train the model, each with a different masking strategy:

Base-level Masking: The masking applies to each nucleotide in the sequence.
Subsequence-level Masking: The masking applies to subsequences of 4-8bp in the sequence.
Motif-level Masking: The model is trained on motif datasets.

The model was trained on 4 NVIDIA V100 GPUs with 32GiB memories.

- Batch size: 50
- Learning rate: 1e-4
- Weight decay: 0.01
- Optimizer: AdamW
- Steps: 2,580,000
- Learning rate warm-up: 129,000 steps
- Learning rate cool-down: 129,000 steps
- Minimum learning rate: 5e-5

## Citation

Citation information is not available for papers published in Closed Access / Author-Fee journals.

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the RNAErnie paper for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
