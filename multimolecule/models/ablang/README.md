---
language: protein
tags:
  - Biology
  - Protein
  - Antibody
license: agpl-3.0
datasets:
  - multimolecule/oas
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# AbLang

Pre-trained antibody language model using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of [AbLang: an antibody language model for completing antibody sequences](https://doi.org/10.1093/bioadv/vbac046) by Tobias H. Olsen, et al.

The OFFICIAL repository of AbLang is at [oxpig/AbLang](https://github.com/oxpig/AbLang).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing AbLang did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

AbLang v1 is an encoder-only Transformer trained on antibody sequences from the Observed Antibody Space (OAS). The official release provides separate heavy-chain and light-chain checkpoints. Both variants use the same architecture and vocabulary, but they were trained on chain-specific data and are represented as separate MultiMolecule variants.

### Variants

- **[multimolecule/ablang-heavy](https://huggingface.co/multimolecule/ablang-heavy)**: AbLang v1 trained on heavy-chain antibody sequences.
- **[multimolecule/ablang-light](https://huggingface.co/multimolecule/ablang-light)**: AbLang v1 trained on light-chain antibody sequences.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variant</th>
    <th>Chain Type</th>
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
    <td>AbLang-Heavy</td>
    <td>Heavy</td>
    <td rowspan="2">12</td>
    <td rowspan="2">768</td>
    <td rowspan="2">12</td>
    <td rowspan="2">3072</td>
    <td rowspan="2">85.83</td>
    <td rowspan="2">28.18</td>
    <td rowspan="2">14.06</td>
    <td rowspan="2">159</td>
  </tr>
  <tr>
    <td>AbLang-Light</td>
    <td>Light</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.ablang](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ablang)
- **Data**: [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/)
- **Paper**: [AbLang: an antibody language model for completing antibody sequences](https://doi.org/10.1093/bioadv/vbac046)
- **Developed by**: Tobias H. Olsen, Iain H. Moal, Charlotte M. Deane
- **Model type**: Encoder-only Transformer for antibody masked language modeling
- **Original Repository**: [oxpig/AbLang](https://github.com/oxpig/AbLang)

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

predictor = pipeline("fill-mask", model="multimolecule/ablang-heavy")
output = predictor("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSY<mask>MSWVRQAPGKGLEWVSA")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given antibody sequence in PyTorch:

```python
from multimolecule import AbLangModel, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang-heavy")
model = AbLangModel.from_pretrained("multimolecule/ablang-heavy")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import AbLangForSequencePrediction, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang-heavy")
model = AbLangForSequencePrediction.from_pretrained("multimolecule/ablang-heavy")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
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
from multimolecule import AbLangForTokenPrediction, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang-heavy")
model = AbLangForTokenPrediction.from_pretrained("multimolecule/ablang-heavy")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

## Training Details

AbLang was trained with masked language modeling (MLM) as the pre-training objective.

### Training Data

AbLang was trained on antibody sequences from the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/).
The heavy-chain model was trained on 14,126,724 sequences, and the light-chain model was trained on 187,068 sequences.

### Training Procedure

#### Pre-training

The heavy-chain and light-chain checkpoints were trained separately on chain-specific OAS sequences.
Please refer to the original paper for details on the training setup.

## Citation

```bibtex
@article{olsen2022ablang,
  title   = {AbLang: an antibody language model for completing antibody sequences},
  author  = {Olsen, Tobias H. and Moal, Iain H. and Deane, Charlotte M.},
  journal = {Bioinformatics Advances},
  volume  = {2},
  number  = {1},
  pages   = {vbac046},
  year    = {2022},
  doi     = {10.1093/bioadv/vbac046},
  url     = {https://doi.org/10.1093/bioadv/vbac046},
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

Please contact the authors of the [AbLang paper](https://doi.org/10.1093/bioadv/vbac046) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
