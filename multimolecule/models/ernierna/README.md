---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
library_name: multimolecule
---

# ERNIE-RNA

Pre-trained model on non-coding RNA (ncRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations](https://doi.org/10.1101/2024.03.17.585376) by Weijie Yin, Zhaoyu Zhang, Liang He, et al.

The OFFICIAL repository of ERNIE-RNA is at [Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA).

**The team releasing ERNIE-RNA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ERNIE-RNA is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variations

- **[`multimolecule/ernierna`](https://huggingface.co/multimolecule/ernierna)**: The ERNIE-RNA model pre-trained on non-coding RNA sequences.
- **[`multimolecule/ernierna.ss`](https://huggingface.co/multimolecule/ernierna.ss)**: The ERNIE-RNA model fine-tuned on RNA secondary structure prediction.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 12         | 768         | 12        | 3072              | 85.67              | 22.36     | 11.17    | 1024           |

### Links

- **Code**: [multimolecule.ernierna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ernierna)
- **Data**: [RNAcentral](https://rnacentral.org)
- **Paper**: [ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations](https://doi.org/10.1101/2024.03.17.585376)
- **Developed by**: Weijie Yin, Zhaoyu Zhang, Liang He, Rui Jiang, Shuo Zhang, Gan Liu, Xuegong Zhang, Tao Qin, Zhen Xie
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [ERNIE](https://huggingface.co/nghuyong/ernie-3.0-base-zh)
- **Original Repository**: [https://github.com/Bruce-ywj/ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)

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
>>> unmasker = pipeline('fill-mask', model='multimolecule/ernierna')
>>> unmasker("uagc<mask>uaucagacugauguuga")

[{'score': 0.47122013568878174,
  'token': 8,
  'token_str': 'G',
  'sequence': 'U A G C G U A U C A G A C U G A U G U U G A'},
 {'score': 0.45008403062820435,
  'token': 6,
  'token_str': 'A',
  'sequence': 'U A G C A U A U C A G A C U G A U G U U G A'},
 {'score': 0.03330205753445625,
  'token': 24,
  'token_str': '*',
  'sequence': 'U A G C * U A U C A G A C U G A U G U U G A'},
 {'score': 0.025803660973906517,
  'token': 25,
  'token_str': '-',
  'sequence': 'U A G C - U A U C A G A C U G A U G U U G A'},
 {'score': 0.018313394859433174,
  'token': 23,
  'token_str': '.',
  'sequence': 'U A G C. U A U C A G A C U G A U G U U G A'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, ErnieRnaModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/ernierna')
model = ErnieRnaModel.from_pretrained('multimolecule/ernierna')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, ErnieRnaForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/ernierna')
model = ErnieRnaForSequencePrediction.from_pretrained('multimolecule/ernierna')

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
from multimolecule import RnaTokenizer, ErnieRnaForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/ernierna')
model = ErnieRnaForNucleotidePrediction.from_pretrained('multimolecule/ernierna')

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
from multimolecule import RnaTokenizer, ErnieRnaForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/ernierna')
model = ErnieRnaForContactPrediction.from_pretrained('multimolecule/ernierna')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

ERNIE-RNA used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The ERNIE-RNA model was pre-trained on [RNAcentral](https://rnacentral.org). RNAcentral is a comprehensive database of non-coding RNA sequences from a wide range of species. It combines 47 different databases, adding up to around 34 million RNA sequences in total.

ERNIE-RNA applied [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) with a cut-off at 100% sequence identity to remove redundancy from the RNAcentral, resulting 25 million unique sequences. Sequences longer than 1024 nucleotides were subsequently excluded. The final dataset contains 20.4 million non-redundant RNA sequences.
ERNIE-RNA preprocessed all tokens by replacing "T"s with "S"s.

Note that [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

ERNIE-RNA used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

The model was trained on 24 NVIDIA V100 GPUs with 32GiB memories.

- Learning rate: 1e-4
- Weight decay: 0.01
- Learning rate warm-up: 20,000 steps

## Citation

**BibTeX**:

```bibtex
@article {Yin2024.03.17.585376,
	author = {Yin, Weijie and Zhang, Zhaoyu and He, Liang and Jiang, Rui and Zhang, Shuo and Liu, Gan and Zhang, Xuegong and Qin, Tao and Xie, Zhen},
	title = {ERNIE-RNA: An RNA Language Model with Structure-enhanced Representations},
	elocation-id = {2024.03.17.585376},
	year = {2024},
	doi = {10.1101/2024.03.17.585376},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {With large amounts of unlabeled RNA sequences data produced by high-throughput sequencing technologies, pre-trained RNA language models have been developed to estimate semantic space of RNA molecules, which facilities the understanding of grammar of RNA language. However, existing RNA language models overlook the impact of structure when modeling the RNA semantic space, resulting in incomplete feature extraction and suboptimal performance across various downstream tasks. In this study, we developed a RNA pre-trained language model named ERNIE-RNA (Enhanced Representations with base-pairing restriction for RNA modeling) based on a modified BERT (Bidirectional Encoder Representations from Transformers) by incorporating base-pairing restriction with no MSA (Multiple Sequence Alignment) information. We found that the attention maps from ERNIE-RNA with no fine-tuning are able to capture RNA structure in the zero-shot experiment more precisely than conventional methods such as fine-tuned RNAfold and RNAstructure, suggesting that the ERNIE-RNA can provide comprehensive RNA structural representations. Furthermore, ERNIE-RNA achieved SOTA (state-of-the-art) performance after fine-tuning for various downstream tasks, including RNA structural and functional predictions. In summary, our ERNIE-RNA model provides general features which can be widely and effectively applied in various subsequent research tasks. Our results indicate that introducing key knowledge-based prior information in the BERT framework may be a useful strategy to enhance the performance of other language models.Competing Interest StatementOne patent based on the study was submitted by Z.X. and W.Y., which is entitled as "A Pre-training Approach for RNA Sequences and Its Applications"(application number, no 202410262527.5). The remaining authors declare no competing interests.},
	URL = {https://www.biorxiv.org/content/early/2024/03/17/2024.03.17.585376},
	eprint = {https://www.biorxiv.org/content/early/2024/03/17/2024.03.17.585376.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [ERNIE-RNA paper](https://doi.org/10.1101/2024.03.17.585376) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
