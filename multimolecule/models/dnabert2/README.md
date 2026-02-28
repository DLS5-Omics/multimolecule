---
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/genbank
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
---

# DNABERT-2

Pre-trained model on multi-species genome using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes](https://doi.org/10.48550/arXiv.2306.15006) by Zhihan Zhou, et al.

The OFFICIAL repository of DNABERT-2 is at [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DNABERT-2 did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DNABERT-2 is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of multi-species genome sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of DNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

<table>
<thead>
  <tr>
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
    <td>12</td>
    <td>768</td>
    <td>12</td>
    <td>3072</td>
    <td>117.07</td>
    <td>125.83</td>
    <td>62.92</td>
    <td>512</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.dnabert2](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/dnabert2)
- **Data**: [GenBank](https://www.ncbi.nlm.nih.gov/genbank)
- **Paper**: [DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes](https://doi.org/10.48550/arXiv.2306.15006)
- **Developed by**: Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana V Davuluri, Han Liu
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [MosaicBERT](https://huggingface.co/mosaicml/mosaic-bert-base)
- **Original Repository**: [zhihan1996/DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)

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

predictor = pipeline("fill-mask", model="multimolecule/dnabert2")
output = predictor("ATCG<mask>TGCA")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import DnaBert2Model
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert2")
model = DnaBert2Model.from_pretrained("multimolecule/dnabert2")

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
from multimolecule import DnaBert2ForSequencePrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert2")
model = DnaBert2ForSequencePrediction.from_pretrained("multimolecule/dnabert2")

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
from multimolecule import DnaBert2ForTokenPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert2")
model = DnaBert2ForTokenPrediction.from_pretrained("multimolecule/dnabert2")

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
from multimolecule import DnaBert2ForContactPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert2")
model = DnaBert2ForContactPrediction.from_pretrained("multimolecule/dnabert2")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

DNABERT-2 used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The DNABERT-2 model was pre-trained on multi-species genome sequences from [GenBank](https://www.ncbi.nlm.nih.gov/genbank).
The dataset encompasses genomes from 135 species, spread across 6 categories. In total, the dataset includes 32.49 billion nucleotide bases, nearly 12 times the volume of the human genome dataset.
All sequences with `N` are excluded, retaining only sequences that consist of `A`, `T`, `C`, and `G`.

DNABERT-2 uses Byte Pair Encoding (BPE) tokenization with a vocabulary size of 4096. This replaces the k-mer tokenization used in the original DNABERT, providing improved computational and sample efficiency.

### Training Procedure

#### Preprocessing

DNABERT-2 used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 8 NVIDIA RTX 2080Ti GPUs.

- Batch size: 4,096
- Steps: 500,000
- Optimizer: AdamW(β1=0.9, β2=0.98, ε=1e-6)
- Learning rate: 5e-4
- Learning rate warm-up: 30,000 steps
- Learning rate scheduler: Linear
- Minimum learning rate: 0
- Weight decay: 1e-5

## Citation

```bibtex
@inproceedings{zhou2024dnabert,
  title={{DNABERT}-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes},
  author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana V Davuluri and Han Liu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=oMLQB4EZE1}
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

Please contact the authors of the [DNABERT-2 paper](https://doi.org/10.48550/arXiv.2306.15006) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
