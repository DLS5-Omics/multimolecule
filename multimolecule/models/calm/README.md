---
language: dna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/ena
library_name: multimolecule
---

# CaLM

Pre-trained model on protein-coding DNA (cDNA) using a masked language modeling (MLM) objective.

## Statement

Codon language embeddings provide strong signals for use in protein engineering is published in [Nature Machine Intelligence](https://doi.org/10.1038/s42256-024-00791-0), which is a Closed Access / Author-Fee Journal.

> Machine learning has been at the forefront of the movement for free and open access to research.
>
> We see no role for closed access or author-fee publication in the future of machine learning research and believe the adoption of these journals as an outlet of record for the machine learning community would be a retrograde step.

The MoltiMolecule team is committed to the principles of open access and open science.

We do NOT endorse the publication of manuscripts on Closed Access / Author-Fee Journals and encourage the community to support Open Access Journals.

Please consider signing the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu).

## Disclaimer

This is an UNOFFICIAL implementation of the [Codon language embeddings provide strong signals for use in protein engineering](https://doi.org/10.1101/2022.12.15.519894) by Carlos Outeiral and Charlotte M. Deane.

The OFFICIAL repository of CaLM is at [oxpig/CaLM](https://github.com/oxpig/CaLM).

**The team releasing CaLM did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

CaLM is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of protein-coding DNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of DNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 12         | 768         | 12        | 3072              | 85.75              | 22.36     | 11.17    | 1024           |

### Links

- **Code**: [multimolecule.calm](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/calm)
- **Weights**: [multimolecule/calm](https://huggingface.co/multimolecule/calm)
- **Data**: [European Nucleotide Archive](https://ebi.ac.uk/ena)
- **Paper**: [Codon language embeddings provide strong signals for use in protein engineering](https://doi.org/10.1101/2022.12.15.519894)
- **Developed by**: Carlos Outeiral, Charlotte M. Deane
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ESM](https://huggingface.co/facebook/esm2_t48_15B_UR50D)
- **Original Repository**: [https://github.com/oxpig/CaLM](https://github.com/oxpig/CaLM)

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
>>> unmasker = pipeline('fill-mask', model='multimolecule/calm')
>>> unmasker("gcc<mask>cgctgacagccgcgg")

[{'score': 0.049356844276189804,
  'token': 42,
  'token_str': 'CGC',
  'sequence': 'GCC CGC CGC UGA CAG CCG CGG'},
 {'score': 0.038425520062446594,
  'token': 67,
  'token_str': 'GGC',
  'sequence': 'GCC GGC CGC UGA CAG CCG CGG'},
 {'score': 0.031837038695812225,
  'token': 62,
  'token_str': 'GCC',
  'sequence': 'GCC GCC CGC UGA CAG CCG CGG'},
 {'score': 0.02501658722758293,
  'token': 32,
  'token_str': 'CAC',
  'sequence': 'GCC CAC CGC UGA CAG CCG CGG'},
 {'score': 0.024257611483335495,
  'token': 37,
  'token_str': 'CCC',
  'sequence': 'GCC CCC CGC UGA CAG CCG CGG'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, CaLmModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/calm')
model = CaLmModel.from_pretrained('multimolecule/calm')

text = "GCCAGTCGCTGACAGCCGCGG"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, CaLmForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/calm')
model = CaLmForSequencePrediction.from_pretrained('multimolecule/calm')

text = "GCCAGTCGCTGACAGCCGCGG"
input = tokenizer(text, return_tensors='pt')
label = torch.tensor([1])

output = model(**input, labels=label)
```

#### Nucleotide Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for nucleotide classification or regression.

Here is how to use this model as backbone to fine-tune for a nucleotide-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, CaLmForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/calm')
model = CaLmForNucleotidePrediction.from_pretrained('multimolecule/calm')

text = "GCCAGTCGCTGACAGCCGCGG"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

#### Contact Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

Here is how to use this model as backbone to fine-tune for a contact-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, CaLmForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/calm')
model = CaLmForContactPrediction.from_pretrained('multimolecule/calm')

text = "GCCAGTCGCTGACAGCCGCGG"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

CaLM used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 25% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The CaLM model was pre-trained coding sequences of all organisms available on the [European Nucleotide Archive (ENA)](https://ebi.ac.uk/ena). European Nucleotide Archive provides a comprehensive record of the world’s nucleotide sequencing information, covering raw sequencing data, sequence assembly information and functional annotation.

CaLM collected coding sequences of all organisms from ENA on April 2022, including 114,214,475 sequences. Only high level assembly information (dataclass CON) were used. Sequences matching the following criteria were filtered out:

- with unknown nucleotides (`N`, `Y`, `R`)
- start codon is not `ATG`
- contains interstitial stop codons
- number of nucleotides is not a multiple of three

To reduce redundancy, CaLM grouped the entries by organism, and apply CD-HIT (CD-HIT-EST) with a cut-off at 40% sequence identity to the translated protein sequences.

The final dataset contains 9,858,385 cDNA sequences.

Note that the alphabet in the original implementation is RNA instead of DNA, therefore, we use [`RnaTokenizer`][multimolecule.RnaTokenizer] to tokenize the sequences. `RnaTokenizer` of `multimolecule` will convert "U"s to "T"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

CaLM used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 25% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

The model was trained on 4 NVIDIA Quadro RTX4000 GPUs with 8GiB memories.

- Learning rate: 1e-4
- Optimizer: AdamW
- Learning rate scheduler: cosine
- Learning rate warm-up: 1,000 steps
- Epochs: 14
- Batch Size: 1,000

## Citation

**BibTeX**:

```bibtex
@article {outeiral2022coodn,
	author = {Outeiral, Carlos and Deane, Charlotte M.},
	title = {Codon language embeddings provide strong signals for protein engineering},
	elocation-id = {2022.12.15.519894},
	year = {2022},
	doi = {10.1101/2022.12.15.519894},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Protein representations from deep language models have yielded state-of-the-art performance across many tasks in computational protein engineering. In recent years, progress has primarily focused on parameter count, with recent models{\textquoteright} capacities surpassing the size of the very datasets they were trained on. Here, we propose an alternative direction. We show that large language models trained on codons, instead of amino acid sequences, provide high-quality representations that outperform comparable state-of-the-art models across a variety of tasks. In some tasks, like species recognition, prediction of protein and transcript abundance, or melting point estimation, we show that a language model trained on codons outperforms every other published protein language model, including some that contain over 50 times more parameters. These results suggest that, in addition to commonly studied scale and model complexity, the information content of biological data provides an orthogonal direction to improve the power of machine learning in biology.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/12/19/2022.12.15.519894},
	eprint = {https://www.biorxiv.org/content/early/2022/12/19/2022.12.15.519894.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [CaLM paper](https://doi.org/10.1101/2022.12.15.519894) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
