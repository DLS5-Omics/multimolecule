---
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/genbank
library_name: multimolecule
pipeline_tag: feature-extraction
mask_token: "<mask>"
---

# DNABERT-S

Pre-trained model on multi-species genome using a contrastive learning objective for species-aware DNA embeddings.

## Disclaimer

This is an UNOFFICIAL implementation of the [DNABERT-S: pioneering species differentiation with species-aware DNA embeddings](https://doi.org/10.1093/bioinformatics/btaf188) by Zhihan Zhou, et al.

The OFFICIAL repository of DNABERT-S is at [MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DNABERT-S did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DNABERT-S is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model built upon [DNABERT-2](https://huggingface.co/multimolecule/dnabert2) and fine-tuned with contrastive learning for species-aware DNA embeddings. The model was trained using the proposed Curriculum Contrastive Learning (C²LR) strategy with the Manifold Instance Mixup (MI-Mix) training objective.

DNABERT-S shares the same architecture as DNABERT-2: it uses Byte Pair Encoding (BPE) tokenization, Attention with Linear Biases (ALiBi) instead of learned position embeddings, and incorporates a Gated Linear Unit (GeGLU) MLP and FlashAttention for improved efficiency.

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

- **Code**: [multimolecule.dnaberts](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/dnaberts)
- **Data**: [GenBank](https://www.ncbi.nlm.nih.gov/genbank)
- **Paper**: [DNABERT-S: pioneering species differentiation with species-aware DNA embeddings](https://doi.org/10.1093/bioinformatics/btaf188)
- **Developed by**: Zhihan Zhou, Weimin Wu, Harrison Ho, Jiayi Wang, Lizhen Shi, Ramana V Davuluri, Zhong Wang, Han Liu
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [MosaicBERT](https://huggingface.co/mosaicml/mosaic-bert-base)
- **Original Repository**: [MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Feature Extraction

You can use this model directly with a pipeline for feature extraction:

```python
import multimolecule  # you must import multimolecule to register models
from transformers import pipeline

predictor = pipeline("feature-extraction", model="multimolecule/dnaberts")
output = predictor("ATCGATCGATCG")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import DnaBertSModel
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnaberts")
model = DnaBertSModel.from_pretrained("multimolecule/dnaberts")

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
from multimolecule import DnaBertSForSequencePrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnaberts")
model = DnaBertSForSequencePrediction.from_pretrained("multimolecule/dnaberts")

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
from multimolecule import DnaBertSForTokenPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnaberts")
model = DnaBertSForTokenPrediction.from_pretrained("multimolecule/dnaberts")

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
from multimolecule import DnaBertSForContactPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnaberts")
model = DnaBertSForContactPrediction.from_pretrained("multimolecule/dnaberts")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

DNABERT-S uses a two-phase Curriculum Contrastive Learning (C²LR) strategy. In phase I, the model is trained with Weighted SimCLR for one epoch. In phase II, the model is further trained with Manifold Instance Mixup (MI-Mix) for two epochs. The training starts from the pre-trained DNABERT-2 checkpoint.

### Training Data

The DNABERT-S model was trained on pairs of non-overlapping DNA sequences from the same species, sourced from [GenBank](https://www.ncbi.nlm.nih.gov/genbank). The dataset consists of 47,923 pairs from 17,636 viral genomes, 1 million pairs from 5,011 fungi genomes, and 1 million pairs from 6,402 bacteria genomes. From the total of 2,047,923 pairs, 2 million were randomly selected for training and the rest were used as validation data. All DNA sequences are 10,000 bp in length.

### Training Procedure

#### Pre-training

The model was trained on 8 NVIDIA A100 80GB GPUs.

- Temperature (τ): 0.05
- Hyperparameter (α): 1.0
- Epochs: 1 (phase I, Weighted SimCLR) + 2 (phase II, MI-Mix)
- Optimizer: Adam
- Learning rate: 3e-6
- Batch size: 48
- Checkpointing: Every 10,000 steps, best selected on validation loss
- Training time: ~48 hours

## Citation

```bibtex
@article{zhou2025dnaberts,
  title={{DNABERT-S}: pioneering species differentiation with species-aware {DNA} embeddings},
  author={Zhou, Zhihan and Wu, Weimin and Ho, Harrison and Wang, Jiayi and Shi, Lizhen and Davuluri, Ramana V and Wang, Zhong and Liu, Han},
  journal={Bioinformatics},
  volume={41},
  pages={i255--i264},
  year={2025},
  doi={10.1093/bioinformatics/btaf188}
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

Please contact the authors of the [DNABERT-S paper](https://doi.org/10.1093/bioinformatics/btaf188) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
