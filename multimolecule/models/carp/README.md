---
language: protein
tags:
  - Biology
  - Protein
license: agpl-3.0
datasets:
  - multimolecule/uniref
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# CARP

Pre-trained convolutional protein language model using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of [Convolutions are competitive with transformers for protein sequence pretraining](https://doi.org/10.1016/j.cels.2024.01.008) by Kevin K. Yang, et al.

The OFFICIAL repository of CARP is at [microsoft/protein-sequence-models](https://github.com/microsoft/protein-sequence-models).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing CARP did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

CARP is a family of ByteNet-style convolutional protein language models. It uses learned token embeddings, a stack of residual dilated 1D convolution blocks, and a final layer normalization before the masked-language-model decoder. The models were pre-trained on the March 2020 release of UniRef50 using the same masked language modeling task as BERT and ESM-1b.

### Variants

- **[multimolecule/carp-600k](https://huggingface.co/multimolecule/carp-600k)**: The CARP model with about 600 thousand parameters.
- **[multimolecule/carp-38m](https://huggingface.co/multimolecule/carp-38m)**: The CARP model with about 38 million parameters.
- **[multimolecule/carp-76m](https://huggingface.co/multimolecule/carp-76m)**: The CARP model with about 76 million parameters.
- **[multimolecule/carp-640m](https://huggingface.co/multimolecule/carp-640m)**: The CARP model with about 640 million parameters.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variant</th>
    <th>Num Layers</th>
    <th>Hidden Size</th>
    <th>Intermediate Size</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
    <th>Max Num Tokens</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CARP-600k</td>
    <td>16</td>
    <td>128</td>
    <td>64</td>
    <td>0.61</td>
    <td>1.25</td>
    <td>0.61</td>
    <td rowspan="4">1024</td>
  </tr>
  <tr>
    <td>CARP-38M</td>
    <td>16</td>
    <td rowspan="2">1024</td>
    <td rowspan="2">512</td>
    <td>37.90</td>
    <td>77.68</td>
    <td>38.70</td>
  </tr>
  <tr>
    <td>CARP-76M</td>
    <td>32</td>
    <td>75.74</td>
    <td>155.26</td>
    <td>77.36</td>
  </tr>
  <tr>
    <td>CARP-640M</td>
    <td>56</td>
    <td>1280</td>
    <td>1280</td>
    <td>642.96</td>
    <td>1317.22</td>
    <td>657.73</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.carp](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/carp)
- **Data**: [UniRef50](https://www.uniprot.org/help/uniref)
- **Paper**: [Convolutions are competitive with transformers for protein sequence pretraining](https://doi.org/10.1016/j.cels.2024.01.008)
- **Developed by**: Kevin K. Yang, Nicolo Fusi, Alex X. Lu
- **Model type**: ByteNet-style convolutional protein masked language model
- **Original Repository**: [microsoft/protein-sequence-models](https://github.com/microsoft/protein-sequence-models)

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

predictor = pipeline("fill-mask", model="multimolecule/carp-600k")
output = predictor("MVLSPADKTNVKAAW<mask>KVGAHAGEYGAEALER")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import ProteinTokenizer, CarpModel


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/carp-600k")
model = CarpModel.from_pretrained("multimolecule/carp-600k")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import ProteinTokenizer, CarpForSequencePrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/carp-600k")
model = CarpForSequencePrediction.from_pretrained("multimolecule/carp-600k")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
input = tokenizer(text, return_tensors="pt")
label = torch.tensor([1])

output = model(**input, labels=label)
```

#### Token Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for token classification or regression.

Here is how to use this model as backbone to fine-tune for a residue-level task in PyTorch:

```python
import torch
from multimolecule import ProteinTokenizer, CarpForTokenPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/carp-600k")
model = CarpForTokenPrediction.from_pretrained("multimolecule/carp-600k")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
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
from multimolecule import ProteinTokenizer, CarpForContactPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/carp-600k")
model = CarpForContactPrediction.from_pretrained("multimolecule/carp-600k")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

CARP was trained with Masked Language Modeling (MLM) as the pre-training objective. Masked residues are predicted from the surrounding protein sequence using bidirectional dilated convolution blocks rather than self-attention layers.

### Training Data

CARP was pre-trained on the March 2020 release of [UniRef50](https://www.uniprot.org/help/uniref).

### Training Procedure

#### Preprocessing

The released CARP checkpoints use the protein alphabet from the official `sequence_models` package. During conversion, equivalent amino-acid and special-token rows are mapped into the MultiMolecule protein tokenizer vocabulary.

#### Pre-training

The model was trained with masked language modeling over a ByteNet-style residual dilated convolution stack.
Please refer to the original paper for details on the training setup.

## Citation

```bibtex
@article{yang2024convolutions,
  author  = {Yang, Kevin K. and Fusi, Nicolo and Lu, Alex X.},
  title   = {Convolutions are competitive with transformers for protein sequence pretraining},
  journal = {Cell Systems},
  volume  = {15},
  number  = {3},
  pages   = {286--294.e2},
  year    = {2024},
  doi     = {10.1016/j.cels.2024.01.008},
  url     = {https://doi.org/10.1016/j.cels.2024.01.008},
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If MultiMolecule supports your research, please cite the MultiMolecule project as follows:

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

Please contact the authors of the [CARP paper](https://doi.org/10.1016/j.cels.2024.01.008) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
