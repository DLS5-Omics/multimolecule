---
language: protein
tags:
  - Biology
  - Protein
license: agpl-3.0
datasets:
  - multimolecule/uniref
  - multimolecule/oas
  - multimolecule/scop
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# AMPLIFY

Pre-trained model on protein sequences using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Protein Language Models: Is Scaling Necessary?](https://doi.org/10.1101/2024.09.23.614603) by Quentin Fournier, Robert M. Vernon, Almer van der Sloot, Benjamin Schulz, Sarath Chandar, and Christopher James Langmead.

The OFFICIAL repository of AMPLIFY is at [chandar-lab/AMPLIFY](https://github.com/chandar-lab/AMPLIFY).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints reproduce the upstream logits to within `1e-4` absolute tolerance on a 144-residue test sequence.

**The team releasing AMPLIFY did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

AMPLIFY is a modern encoder-only protein language model with RMSNorm, SwiGLU, and rotary position embeddings. It is pre-trained on a clustered corpus of UniRef100, the Observed Antibody Space, and SCOP (collected as [UR100P](https://huggingface.co/datasets/chandar-lab/UR100P)) using a masked language modeling objective. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/amplify-120m](https://huggingface.co/multimolecule/amplify-120m)**: The AMPLIFY model with 120 million parameters.
- **[multimolecule/amplify-350m](https://huggingface.co/multimolecule/amplify-350m)**: The AMPLIFY model with 350 million parameters.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variants</th>
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
    <td>AMPLIFY-120M</td>
    <td>24</td>
    <td>640</td>
    <td>10</td>
    <td>2560</td>
    <td>118.67</td>
    <td>137.34</td>
    <td>68.58</td>
    <td rowspan="2">2048</td>
  </tr>
  <tr>
    <td>AMPLIFY-350M</td>
    <td>32</td>
    <td>960</td>
    <td>15</td>
    <td>3840</td>
    <td>354.91</td>
    <td>394.98</td>
    <td>197.30</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.amplify](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/amplify)
- **Weights**: [multimolecule/amplify-120m](https://huggingface.co/multimolecule/amplify-120m), [multimolecule/amplify-350m](https://huggingface.co/multimolecule/amplify-350m)
- **Data**: [chandar-lab/UR100P](https://huggingface.co/datasets/chandar-lab/UR100P)
- **Paper**: [Protein Language Models: Is Scaling Necessary?](https://doi.org/10.1101/2024.09.23.614603)
- **Developed by**: Quentin Fournier, Robert M. Vernon, Almer van der Sloot, Benjamin Schulz, Sarath Chandar, Christopher James Langmead
- **Model type**: Encoder-only Transformer with RMSNorm, SwiGLU, and rotary position embeddings
- **Original Repository**: [chandar-lab/AMPLIFY](https://github.com/chandar-lab/AMPLIFY)

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

predictor = pipeline("fill-mask", model="multimolecule/amplify-120m")
output = predictor("MVLSPADKTNVKAAW<mask>KVGAHAGEYGAEALER")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import ProteinTokenizer, AMPLIFYModel


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/amplify-120m")
model = AMPLIFYModel.from_pretrained("multimolecule/amplify-120m")

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
from multimolecule import ProteinTokenizer, AMPLIFYForSequencePrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/amplify-120m")
model = AMPLIFYForSequencePrediction.from_pretrained("multimolecule/amplify-120m")

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
from multimolecule import ProteinTokenizer, AMPLIFYForTokenPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/amplify-120m")
model = AMPLIFYForTokenPrediction.from_pretrained("multimolecule/amplify-120m")

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
from multimolecule import ProteinTokenizer, AMPLIFYForContactPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/amplify-120m")
model = AMPLIFYForContactPrediction.from_pretrained("multimolecule/amplify-120m")

text = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

AMPLIFY was trained with Masked Language Modeling (MLM) as the pre-training objective: 15% of the residues in the input are randomly selected as prediction targets, and the model is asked to recover the original amino acids from the surrounding context. The model is bidirectional (encoder-only) so the prediction at each masked position attends to the entire sequence.

### Training Data

AMPLIFY was pre-trained on the [UR100P](https://huggingface.co/datasets/chandar-lab/UR100P) dataset, which is a curated union of:

- **UniRef100**: All UniProt sequences clustered at 100% sequence identity.
- **Observed Antibody Space (OAS)**: A large compendium of antibody repertoires.
- **SCOP**: Structurally classified protein domains.

### Training Procedure

#### Preprocessing

AMPLIFY uses masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the residues are masked.
- In 80% of the cases, the masked residues are replaced by `<mask>`.
- In 10% of the cases, the masked residues are replaced by a random residue (different) from the one they replace.
- In the 10% remaining cases, the masked residues are left as is.

#### Pre-training

Training is performed in two stages, both on the UR100P dataset:

- **Stage 1**: trained for 1,000,000 steps at a maximum length of 512 residues with a cosine-decayed learning rate of `1e-3`.
- **Stage 2**: trained for an additional 25,000 (120M) or 50,000 (350M) steps at a maximum length of 2,048 residues with a constant learning rate of `1e-4`.

Both stages use AdamW with betas `(0.9, 0.95)`, weight decay `0.01`, gradient clipping `1.0`, mixed-precision `bf16` with `tf32`, a total batch size of 4,096 sequences, and DeepSpeed ZeRO stage 3.

## Citation

**BibTeX**:

```bibtex
@article{Fournier2024.09.23.614603,
  title     = {Protein Language Models: Is Scaling Necessary?},
  author    = {Fournier, Quentin and Vernon, Robert M. and van der Sloot, Almer and Schulz, Benjamin and Chandar, Sarath and Langmead, Christopher James},
  year      = {2024},
  journal   = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2024.09.23.614603},
  url       = {https://www.biorxiv.org/content/early/2024/09/23/2024.09.23.614603},
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

Please contact the authors of the [AMPLIFY paper](https://doi.org/10.1101/2024.09.23.614603) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
