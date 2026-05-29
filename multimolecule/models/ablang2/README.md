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

# AbLang2

Pre-trained model on paired and unpaired antibody sequences using a modified masked language modeling objective.

## Disclaimer

This is an UNOFFICIAL implementation of [Addressing the antibody germline bias and its effect on language models for improved antibody design](https://doi.org/10.1093/bioinformatics/btae618) by Tobias H. Olsen, et al.

The OFFICIAL repository of AbLang2 is at [oxpig/AbLang2](https://github.com/oxpig/AbLang2).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing AbLang2 did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

AbLang2 is an antibody-specific encoder-only protein language model trained to reduce antibody germline bias in masked residue prediction. It uses multi-head self-attention with rotary position embeddings and SwiGLU feed-forward blocks. The released paired model is trained on paired and unpaired antibody sequence data and is optimized for non-germline residue prediction.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 12         | 480         | 20        | 1920              | 44.82              | 24.48     | 12.20    | 256            |

> [!NOTE]
> `Max Num Tokens` reflects the training sequence length of the released checkpoint. AbLang2 uses rotary position
> embeddings and has no `max_position_embeddings` field, so the architecture itself does not impose a hard length limit.

### Links

- **Code**: [multimolecule.ablang2](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ablang2)
- **Data**: [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/)
- **Paper**: [Addressing the antibody germline bias and its effect on language models for improved antibody design](https://doi.org/10.1093/bioinformatics/btae618)
- **Developed by**: Tobias H. Olsen, Iain H. Moal, Charlotte M. Deane
- **Model type**: Encoder-only antibody language model with rotary position embeddings and SwiGLU feed-forward blocks
- **Original Repository**: [oxpig/AbLang2](https://github.com/oxpig/AbLang2)

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

predictor = pipeline("fill-mask", model="multimolecule/ablang2")
output = predictor("EVQLVESGGGLVQPGGSLRLSCAAS<mask>FTFSSYAMSWVRQAPGKGLEWV")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given antibody sequence in PyTorch:

```python
from multimolecule import ProteinTokenizer, AbLang2Model


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang2")
model = AbLang2Model.from_pretrained("multimolecule/ablang2")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import ProteinTokenizer, AbLang2ForSequencePrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang2")
model = AbLang2ForSequencePrediction.from_pretrained("multimolecule/ablang2")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
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
from multimolecule import ProteinTokenizer, AbLang2ForTokenPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang2")
model = AbLang2ForTokenPrediction.from_pretrained("multimolecule/ablang2")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (1, len(text)))

output = model(**input, labels=label)
```

#### Contact Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

Here is how to use this model as backbone to fine-tune for a contact-level task in PyTorch:

```python
import torch
from multimolecule import ProteinTokenizer, AbLang2ForContactPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/ablang2")
model = AbLang2ForContactPrediction.from_pretrained("multimolecule/ablang2")

text = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (1, len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

AbLang2 was trained with masked language modeling as the pre-training objective. The model is bidirectional, so each masked position attends to surrounding residues on both sides.

### Training Data

AbLang2 is trained on sequences derived from the Observed Antibody Space (OAS), including 35.6 million unpaired heavy/light-chain sequences and 1.26 million paired antibody sequences for the final released model.

### Training Procedure

The AbLang2 paper focuses on reducing antibody germline bias in residue prediction and model-guided antibody design.
Please refer to the original paper for details on the training setup.

## Citation

```bibtex
@article{olsen2024ablang2,
  title   = {Addressing the antibody germline bias and its effect on language models for improved antibody design},
  author  = {Olsen, Tobias H. and Moal, Iain H. and Deane, Charlotte M.},
  year    = {2024},
  journal = {Bioinformatics},
  volume  = {40},
  number  = {11},
  pages   = {btae618},
  doi     = {10.1093/bioinformatics/btae618},
  url     = {https://doi.org/10.1093/bioinformatics/btae618},
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

Please contact the authors of the [AbLang2 paper](https://doi.org/10.1093/bioinformatics/btae618) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
