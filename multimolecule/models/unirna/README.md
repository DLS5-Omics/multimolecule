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

# Uni-RNA

Pre-trained model on RNA using a masked language modeling (MLM) objective.

> [!WARNING]
> This model is currently in a **Development Preview** state. It is not yet ready for production use and may contain incomplete or experimental features. The MultiMolecule team is actively working on improving this model, and we welcome feedback from the community.

## Disclaimer

This is the OFFICIAL implementation of the [Uni-RNA: Universal Pre-Trained Models Revolutionize RNA Research](https://doi.org/10.1101/2023.07.11.548588) by Xi Wang, Ruichu Gu, Zhiyuan Chen, et al.

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

## Model Details

Uni-RNA is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 16         | 1024        | 16        | 3072              | 868                | 44.05     | 22.01    | 1024           |

### Links

- **Code**: [multimolecule.unirna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/unirna)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: [Uni-RNA: Universal Pre-Trained Models Revolutionize RNA Research](https://doi.org/10.1101/2023.07.11.548588)
- **Developed by**: Xi Wang, Ruichu Gu, Zhiyuan Chen, Yongge Li, Xiaohong Ji, Guolin Ke, Han Wen
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ESM](https://huggingface.co/facebook/esm2_t48_15B_UR50D)

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

predictor = pipeline("fill-mask", model="multimolecule/unirna")
output = predictor("gguc<mask>cucugguuagaccagaucugagccu")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaFmModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/unirna")
model = RnaFmModel.from_pretrained("multimolecule/unirna")

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


tokenizer = RnaTokenizer.from_pretrained("multimolecule/unirna")
model = RnaFmForSequencePrediction.from_pretrained("multimolecule/unirna")

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


tokenizer = RnaTokenizer.from_pretrained("multimolecule/unirna")
model = RnaFmForTokenPrediction.from_pretrained("multimolecule/unirna")

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


tokenizer = RnaTokenizer.from_pretrained("multimolecule/unirna")
model = RnaFmForContactPrediction.from_pretrained("multimolecule/unirna")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

Uni-RNA used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The Uni-RNA model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

Uni-RNA applied [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) with a cut-off at 100% sequence identity to remove redundancy from the RNAcentral. The final dataset contains 23.7 million non-redundant RNA sequences.

Uni-RNA preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

Uni-RNA used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

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
@article {Wang2023.07.11.548588,
	author = {Wang, Xi and Gu, Ruichu and Chen, Zhiyuan and Li, Yongge and Ji, Xiaohong and Ke, Guolin and Wen, Han},
	title = {UNI-RNA: UNIVERSAL PRE-TRAINED MODELS REVOLUTIONIZE RNA RESEARCH},
	elocation-id = {2023.07.11.548588},
	year = {2023},
	doi = {10.1101/2023.07.11.548588},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {RNA molecules play a crucial role as intermediaries in diverse biological processes. Attaining a profound understanding of their function can substantially enhance our comprehension of life{\textquoteright}s activities and facilitate drug development for numerous diseases. The advent of high-throughput sequencing technologies makes vast amounts of RNA sequence data accessible, which contains invaluable information and knowledge. However, deriving insights for further application from such an immense volume of data poses a significant challenge. Fortunately, recent advancements in pre-trained models have surfaced as a revolutionary solution for addressing such challenges owing to their exceptional ability to automatically mine and extract hidden knowledge from massive datasets. Inspired by the past successes, we developed a novel context-aware deep learning model named Uni-RNA that performs pre-training on the largest dataset of RNA sequences at the unprecedented scale to date. During this process, our model autonomously unraveled the obscured evolutionary and structural information embedded within the RNA sequences. As a result, through fine-tuning, our model achieved the state-of-the-art (SOTA) performances in a spectrum of downstream tasks, including both structural and functional predictions. Overall, Uni-RNA established a new research paradigm empowered by the large pre-trained model in the field of RNA, enabling the community to unlock the power of AI at a whole new level to significantly expedite the pace of research and foster groundbreaking discoveries.Competing Interest StatementPatents have been filed based on the methods described in this manuscript. All authors are employees of DP Technology, Beijing.},
	URL = {https://www.biorxiv.org/content/early/2023/07/12/2023.07.11.548588},
	eprint = {https://www.biorxiv.org/content/early/2023/07/12/2023.07.11.548588.full.pdf},
	journal = {bioRxiv}
}

```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [Uni-RNA paper](https://doi.org/10.1101/2023.07.11.548588) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
