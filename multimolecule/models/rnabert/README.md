---
language: rna
tags:
  - Biology
  - RNA
  - ncRNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# RNABERT

Pre-trained model on non-coding RNA (ncRNA) using masked language modeling (MLM) and structural alignment learning (SAL) objectives.

## Disclaimer

This is an UNOFFICIAL implementation of the [Informative RNA-base embedding for functional RNA clustering and structural alignment](https://doi.org/10.1093/nargab/lqac012) by Manato Akiyama and Yasubumi Sakakibara.

The OFFICIAL repository of RNABERT is at [mana438/RNABERT](https://github.com/mana438/RNABERT).

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RNABERT.
>
> The original implementation of RNABERT does not prepend `<bos>` (`<cls>`) and append `<eos>` tokens to the input sequence.
> This should not affect the performance of the model in most cases, but it can lead to unexpected behavior in some cases.
>
> Please set `bos_token=None, eos_token=None` in the tokenizer and set `bos_token_id=None, eos_token_id=None` in the model configuration if you want the exact behavior of the original implementation.

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RNABERT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RNABERT is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 6          | 120         | 12        | 40                | 0.48               | 0.15      | 0.08     | 440            |

### Links

- **Code**: [multimolecule.rnabert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rnabert)
- **Weights**: [multimolecule/rnabert](https://huggingface.co/multimolecule/rnabert)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: [Informative RNA-base embedding for functional RNA clustering and structural alignment](https://doi.org/10.1093/nargab/lqac012)
- **Developed by**: JManato Akiyama and Yasubumi Sakakibara
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [mana438/RNABERT](https://github.com/mana438/RNABERT)

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

predictor = pipeline("fill-mask", model="multimolecule/rnabert")
output = predictor("gguc<mask>cucugguuagaccagaucugagccu")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaBertModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
model = RnaBertModel.from_pretrained("multimolecule/rnabert")

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
from multimolecule import RnaTokenizer, RnaBertForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
model = RnaBertForSequencePrediction.from_pretrained("multimolecule/rnabert")

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
from multimolecule import RnaTokenizer, RnaBertForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
model = RnaBertForTokenPrediction.from_pretrained("multimolecule/rnabert")

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
from multimolecule import RnaTokenizer, RnaBertForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnabert")
model = RnaBertForContactPrediction.from_pretrained("multimolecule/rnabert")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RNABERT has two pre-training objectives: masked language modeling (MLM) and structural alignment learning (SAL).

- **Masked Language Modeling (MLM)**: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.
- **Structural Alignment Learning (SAL)**: the model learns to predict the structural alignment of two RNA sequences. The model is trained to predict the alignment score of two RNA sequences using the Needleman-Wunsch algorithm.

### Training Data

The RNABERT model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

RNABERT used a subset of 76, 237 human ncRNA sequences from RNAcentral for pre-training.
RNABERT preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RNABERT preprocess the dataset by applying 10 different mask patterns to the 72, 237 human ncRNA sequences. The final dataset contains 722, 370 sequences. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 1 NVIDIA V100 GPU.

## Citation

```bibtex
@article{akiyama2022informative,
    author = {Akiyama, Manato and Sakakibara, Yasubumi},
    title = "{Informative RNA base embedding for RNA structural alignment and clustering by deep representation learning}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {4},
    number = {1},
    pages = {lqac012},
    year = {2022},
    month = {02},
    abstract = "{Effective embedding is actively conducted by applying deep learning to biomolecular information. Obtaining better embeddings enhances the quality of downstream analyses, such as DNA sequence motif detection and protein function prediction. In this study, we adopt a pre-training algorithm for the effective embedding of RNA bases to acquire semantically rich representations and apply this algorithm to two fundamental RNA sequence problems: structural alignment and clustering. By using the pre-training algorithm to embed the four bases of RNA in a position-dependent manner using a large number of RNA sequences from various RNA families, a context-sensitive embedding representation is obtained. As a result, not only base information but also secondary structure and context information of RNA sequences are embedded for each base. We call this ‘informative base embedding’ and use it to achieve accuracies superior to those of existing state-of-the-art methods on RNA structural alignment and RNA family clustering tasks. Furthermore, upon performing RNA sequence alignment by combining this informative base embedding with a simple Needleman–Wunsch alignment algorithm, we succeed in calculating structural alignments with a time complexity of O(n2) instead of the O(n6) time complexity of the naive implementation of Sankoff-style algorithm for input RNA sequence of length n.}",
    issn = {2631-9268},
    doi = {10.1093/nargab/lqac012},
    url = {https://doi.org/10.1093/nargab/lqac012},
    eprint = {https://academic.oup.com/nargab/article-pdf/4/1/lqac012/42577168/lqac012.pdf},
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

Please contact the authors of the [RNABERT paper](https://doi.org/10.1093/nargab/lqac012) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
