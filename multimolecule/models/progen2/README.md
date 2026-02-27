---
tags:
  - Biology
  - Protein
license: agpl-3.0
datasets:
  - multimolecule/uniref
  - multimolecule/bfd
  - multimolecule/oas
library_name: multimolecule
pipeline_tag: text-generation
---

# ProGen2

Pre-trained model on protein sequences using a causal language modeling (CLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [ProGen2: Exploring the Boundaries of Protein Language Models](https://doi.org/10.1016/j.cels.2023.10.002) by Erik Nijkamp, Jeffrey A. Ruffolo, et al.

The OFFICIAL repository of ProGen2 is at [enijkamp/progen](https://github.com/enijkamp/progen2).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ProGen2 did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ProGen2 is a [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b)-style model pre-trained on a large corpus of protein sequences in a self-supervised fashion. This means that the model was trained on the raw amino acids of protein sequences only, with an automatic process to generate inputs and labels from those sequences. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/progen2-xlarge](https://huggingface.co/multimolecule/progen2-xlarge)**: The ProGen2 model pre-trained on Uniref90 and BFD30 with 6.4 billion parameters.
- **[multimolecule/progen2-large](https://huggingface.co/multimolecule/progen2-large)**: The ProGen2 model pre-trained on Uniref90 and BFD30 with 2.7 billion parameters.
- **[multimolecule/progen2-bfd90](https://huggingface.co/multimolecule/progen2-bfd90)**: The ProGen2 model pre-trained on Uniref90 and BFD90 with 2.7 billion parameters.
- **[multimolecule/progen2-base](https://huggingface.co/multimolecule/progen2-base)**: The ProGen2 model pre-trained on Uniref90 and BFD30 with 764 million parameters.
- **[multimolecule/progen2-oas](https://huggingface.co/multimolecule/progen2-oas)**: The ProGen2 model pre-trained on OAS with 764 million parameters.
- **[multimolecule/progen2-medium](https://huggingface.co/multimolecule/progen2-medium)**: The ProGen2 model pre-trained on Uniref90 and BFD30 with 764 million parameters.
- **[multimolecule/progen2-small](https://huggingface.co/multimolecule/progen2-small)**: The ProGen2 model pre-trained on Uniref90 and BFD30 with 151 million parameters.

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
    <th>Context Length</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ProGen2-xlarge</td>
    <td rowspan="3">32</td>
    <td>4096</td>
    <td>16</td>
    <td>16384</td>
    <td>6443.66</td>
    <td>6735.76</td>
    <td>3367.27</td>
    <td rowspan="3">1024</td>
  </tr>
  <tr>
    <td>ProGen2-large</td>
    <td rowspan="2">2560</td>
    <td rowspan="2">32</td>
    <td rowspan="2">10240</td>
    <td rowspan="2">2517.34</td>
    <td rowspan="2">2664.21</td>
    <td rowspan="2">1331.45</td>
  </tr>
  <tr>
    <td>ProGen2-bfd90</td>
  </tr>
  <tr>
    <td>ProGen2-base</td>
    <td rowspan="3">27</td>
    <td rowspan="3">1536</td>
    <td rowspan="4">16</td>
    <td rowspan="3">6144</td>
    <td rowspan="3">764.81</td>
    <td rowspan="3">826.85</td>
    <td rowspan="3">413.12</td>
    <td>2048</td>
  </tr>
  <tr>
    <td>ProGen2-oas</td>
    <td rowspan="3">1024</td>
  </tr>
  <tr>
    <td>ProGen2-medium</td>
  </tr>
  <tr>
    <td>ProGen2-small</td>
    <td>12</td>
    <td>1024</td>
    <td>4096</td>
    <td>151.15</td>
    <td>167.74</td>
    <td>83.75</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.progen2](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/progen2)
- **Weights**: [multimolecule/progen2](https://huggingface.co/multimolecule/progen2-base)
- **Data**: [UniRef](https://www.uniprot.org/uniref), [BFD](https://bfd.mmseqs.com)
- **Paper**: [ProGen2: Exploring the Boundaries of Protein Language Models](https://doi.org/10.1016/j.cels.2023.10.002)
- **Developed by**: Erik Nijkamp, Jeffrey A. Ruffolo, Eli N. Weinstein, Nikhil Naik, Ali Madani
- **Model type**: [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b)
- **Original Repository**: [enijkamp/progen2](https://github.com/enijkamp/progen2)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Text Generation

You can use this model directly with a pipeline for text generation:

```python
import multimolecule  # you must import multimolecule to register models
from transformers import pipeline

generator = pipeline("text-generation", model="multimolecule/progen2-base")
output = generator("MGHGVSRPPVVTLR", max_new_tokens=50)
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import ProteinTokenizer, ProGen2Model


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/progen2-base")
model = ProGen2Model.from_pretrained("multimolecule/progen2-base")

text = "MGHGVSRPPVVTLRPAVLDDCPVLWR"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import ProteinTokenizer, ProGen2ForSequencePrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/progen2-base")
model = ProGen2ForSequencePrediction.from_pretrained("multimolecule/progen2-base")

text = "MGHGVSRPPVVTLRPAVLDDCPVLWR"
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
from multimolecule import ProteinTokenizer, ProGen2ForTokenPrediction


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/progen2-base")
model = ProGen2ForTokenPrediction.from_pretrained("multimolecule/progen2-base")

text = "MGHGVSRPPVVTLRPAVLDDCPVLWR"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

## Training Details

ProGen2 used Causal Language Modeling (CLM) as the pre-training objective: given a protein sequence, the model is trained to predict the next amino acid token autoregressively.

### Training Data

The ProGen2 models were pre-trained on protein sequence databases:

- **Uniref90**: A clustered version of the UniProt database at 90% sequence identity, containing approximately 135 million sequences.
- **BFD30**: The Big Fantastic Database clustered at 30% sequence identity, approximately one-third the size of Uniref90.
- **BFD90**: The Big Fantastic Database clustered at 90% sequence identity, approximately twice the size of Uniref90.
- **OAS**: The Observed Antibody Space database, clustered at 85% sequence identity.

Different model variants were trained on different combinations:

- **progen2-small, progen2-medium, progen2-base, progen2-large, progen2-xlarge**: Trained on Uniref90 and BFD30.
- **progen2-bfd90**: Trained on Uniref90 and BFD90.
- **progen2-oas**: Trained on the OAS database.

### Training Procedure

ProGen2 used causal language modeling (CLM) as the pre-training objective.

#### Pre-training

The model was trained on Google TPU-v3 pods using JAX.

- Batch size: 500,000 -- 1,000,000
- Steps: 350,000 -- 400,000
- Optimizer: Adam(β1=0.9, β2=0.999, ε=1e-8)
- Learning rate: 1e-5 -- 6e-4
- Learning rate scheduler: Cosine
- Learning rate warm-up: 3,000 -- 10,000 steps
- Weight decay: 0.1
- Maximum Gradient Norm: 0.8 -- 1.0

## Citation

**BibTeX**:

```bibtex
@ARTICLE{Nijkamp2023-jz,
  title     = "{ProGen2}: Exploring the boundaries of protein language models",
  author    = "Nijkamp, Erik and Ruffolo, Jeffrey A and Weinstein, Eli N and
               Naik, Nikhil and Madani, Ali",
  abstract  = "Attention-based models trained on protein sequences have
               demonstrated incredible success at classification and generation
               tasks relevant for artificial-intelligence-driven protein
               design. However, we lack a sufficient understanding of how very
               large-scale models and data play a role in effective protein
               model development. We introduce a suite of protein language
               models, named ProGen2, that are scaled up to 6.4B parameters and
               trained on different sequence datasets drawn from over a billion
               proteins from genomic, metagenomic, and immune repertoire
               databases. ProGen2 models show state-of-the-art performance in
               capturing the distribution of observed evolutionary sequences,
               generating novel viable sequences, and predicting protein
               fitness without additional fine-tuning. As large model sizes and
               raw numbers of protein sequences continue to become more widely
               accessible, our results suggest that a growing emphasis needs to
               be placed on the data distribution provided to a protein
               sequence model. Our models and code are open sourced for
               widespread adoption in protein engineering. A record of this
               paper's Transparent Peer Review process is included in the
               supplemental information.",
  journal   = "Cell Syst.",
  publisher = "Elsevier BV",
  volume    =  14,
  number    =  11,
  pages     = "968--978.e3",
  month     =  nov,
  year      =  2023,
  keywords  = "fitness prediction; language modeling; protein design",
  copyright = "http://www.elsevier.com/open-access/userlicense/1.0/",
  language  = "en"
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

Please contact the authors of the [ProGen2 paper](https://doi.org/10.1016/j.cels.2023.10.002) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
