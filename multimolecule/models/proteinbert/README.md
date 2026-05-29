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

# ProteinBERT

Pre-trained model on protein sequences and Gene Ontology annotations using a combined language modeling and annotation prediction objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [ProteinBERT: a universal deep-learning model of protein sequence and function](https://doi.org/10.1093/bioinformatics/btac020) by Nadav Brandes, et al.

The OFFICIAL repository of ProteinBERT is at [nadavbra/protein_bert](https://github.com/nadavbra/protein_bert).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ProteinBERT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ProteinBERT is a protein language model with coupled local residue representations and a global protein representation.
It is pre-trained on UniRef90 with a sequence language modeling objective and a Gene Ontology annotation recovery objective.
ProteinBERT uses convolutional local branches and global-attention layers instead of quadratic self-attention, so the architecture has no learned positional table and can be evaluated on variable sequence lengths.

### Model Specification

| Num Layers | Hidden Size | Global Hidden Size | Num Heads | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | ------------------ | --------- | -------- | -------------- |
| 6          | 128         | 512                | 4         | 15.98              | 7.16      | 3.54     | 1024           |

### Links

- **Code**: [multimolecule.proteinbert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/proteinbert)
- **Data**: [UniRef90](https://www.uniprot.org/help/uniref)
- **Paper**: [ProteinBERT: a universal deep-learning model of protein sequence and function](https://doi.org/10.1093/bioinformatics/btac020)
- **Developed by**: Nadav Brandes, Dan Ofer, Yam Peleg, Nadav Rappoport, Michal Linial
- **Model type**: Protein language model with local convolutional branches and global-attention layers
- **Original Repository**: [nadavbra/protein_bert](https://github.com/nadavbra/protein_bert)

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

predictor = pipeline("fill-mask", model="multimolecule/proteinbert")
output = predictor("MVLSPADKTNVKAAW<mask>KVGAHAGEYGAEALER")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import ProteinTokenizer, ProteinBertModel


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/proteinbert")
model = ProteinBertModel.from_pretrained("multimolecule/proteinbert")

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
from multimolecule import ProteinTokenizer, ProteinBertForSequencePrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/proteinbert")
model = ProteinBertForSequencePrediction.from_pretrained("multimolecule/proteinbert")

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
from multimolecule import ProteinTokenizer, ProteinBertForTokenPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/proteinbert")
model = ProteinBertForTokenPrediction.from_pretrained("multimolecule/proteinbert")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (1, len(text)))

output = model(**input, labels=label)
```

## Training Details

### Training Data

ProteinBERT is pre-trained on approximately 106 million protein sequences from UniRef90 and Gene Ontology annotations.

### Training Procedure

ProteinBERT is trained with a combined objective over masked protein sequence recovery and Gene Ontology annotation prediction.
Please refer to the original paper for details on the training setup.

## Citation

```bibtex
@article{brandes2022proteinbert,
  title   = {ProteinBERT: a universal deep-learning model of protein sequence and function},
  author  = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
  year    = {2022},
  journal = {Bioinformatics},
  volume  = {38},
  number  = {8},
  pages   = {2102--2110},
  doi     = {10.1093/bioinformatics/btac020},
  url     = {https://doi.org/10.1093/bioinformatics/btac020},
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

Please contact the authors of the [ProteinBERT paper](https://doi.org/10.1093/bioinformatics/btac020) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
