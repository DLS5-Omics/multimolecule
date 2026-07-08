---
language: protein
tags:
  - Biology
  - Protein
  - Masked Language Modeling
license: agpl-3.0
datasets:
  - multimolecule/uniref
  - multimolecule/mgnify
  - multimolecule/jgi
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: <mask>
---

# ESM Cambrian (ESMC)

Pre-trained model on protein sequences using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Language Modeling Materializes a World Model of Protein Biology](https://doi.org/10.64898/2026.06.03.729735) by Salvatore Candido, Thomas Hayes, Alexander Derry, Roshan Rao, Zeming Lin, Alexander Rives, et al.

The OFFICIAL repository of ESMC is at [Biohub/esm](https://github.com/Biohub/esm).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ESMC did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ESM Cambrian is an encoder-only protein language model trained on billions of protein sequences in a self-supervised fashion. The model learns protein representations by predicting masked amino-acid tokens from sequence context. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/esmc-6b](https://huggingface.co/multimolecule/esmc-6b)**: The ESMC model with 6 billion parameters.
- **[multimolecule/esmc-600m](https://huggingface.co/multimolecule/esmc-600m)**: The ESMC model with 600 million parameters.
- **[multimolecule/esmc-300m](https://huggingface.co/multimolecule/esmc-300m)**: The ESMC model with 300 million parameters.

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
    <td>ESMC-6B</td>
    <td>80</td>
    <td>2560</td>
    <td>40</td>
    <td>6912</td>
    <td>6351.87</td>
    <td>6716.35</td>
    <td>3355.44</td>
    <td rowspan="3">2048</td>
  </tr>
  <tr>
    <td>ESMC-600M</td>
    <td>36</td>
    <td>1152</td>
    <td>18</td>
    <td>3072</td>
    <td>574.97</td>
    <td>631.66</td>
    <td>315.28</td>
  </tr>
  <tr>
    <td>ESMC-300M</td>
    <td>30</td>
    <td>960</td>
    <td>15</td>
    <td>2560</td>
    <td>332.95</td>
    <td>370.71</td>
    <td>184.97</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.esmc](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/esmc)
- **Data**: [UniRef](https://www.uniprot.org/help/uniref), [MGnify](https://www.ebi.ac.uk/metagenomics/), and [Joint Genome Institute](https://jgi.doe.gov/)
- **Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://doi.org/10.64898/2026.06.03.729735)
- **Developed by**: Salvatore Candido, Thomas Hayes, Alexander Derry, Roshan Rao, Zeming Lin, Robert Verkuil, Bryan Wu, Jin Sub Lee, Elise S. Bruguera, Jehan A. Keval, Mykhailo Kopylov, John E. Pak, Wesley Wu, Neil Thomas, Samson Mataraso, Alvin Hsu, Ashton C. Trotman-Grant, Kilian Fatras, Allan dos Santos Costa, Rohil Badkundri, Halil Akin, Deniz Oktay, Jonathan Deaton, Elizabeth Montabana, Hrishita Sitwala, Yue Yu, Marius Wiggert, Dylan Alexander Carlin, Anthony W. Goering, Tomasz Blazejewski, McCullen Sandora, Michael Hla, Tina Z. Jia, Leon H. Kloker, Nicholas J. Sofroniew, Masatoshi Uehara, Jassi Pannu, Sharrol Bachas, Daniel S. Liu, Tom Sercu, Alexander Rives
- **Model type**: Encoder-only Transformer with rotary position embeddings, QK layer normalization, and SwiGLU feed-forward blocks
- **Original Repository**: [Biohub/esm](https://github.com/Biohub/esm)

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

predictor = pipeline("fill-mask", model="multimolecule/esmc-300m")
output = predictor("MSK<mask>EELFTGVVPILVELDGDVNGHK")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import EsmCModel, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/esmc-300m")
model = EsmCModel.from_pretrained("multimolecule/esmc-300m")

text = "MSKGEELFTGVVPILVELDGDVNGHK"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import EsmCForSequencePrediction, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/esmc-300m")
model = EsmCForSequencePrediction.from_pretrained("multimolecule/esmc-300m")

text = "MSKGEELFTGVVPILVELDGDVNGHK"
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
from multimolecule import EsmCForTokenPrediction, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/esmc-300m")
model = EsmCForTokenPrediction.from_pretrained("multimolecule/esmc-300m")

text = "MSKGEELFTGVVPILVELDGDVNGHK"
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
from multimolecule import EsmCForContactPrediction, ProteinTokenizer


tokenizer = ProteinTokenizer.from_pretrained("multimolecule/esmc-300m")
model = EsmCForContactPrediction.from_pretrained("multimolecule/esmc-300m")

text = "MSKGEELFTGVVPILVELDGDVNGHK"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

ESMC used Masked Language Modeling (MLM) as the pre-training objective: taking a protein sequence, the model predicts amino-acid tokens that have been masked from the input sequence.

### Training Data

ESMC was trained on protein sequences from [UniRef](https://www.uniprot.org/help/uniref), [MGnify](https://www.ebi.ac.uk/metagenomics/), and the [Joint Genome Institute](https://jgi.doe.gov/). The released materials describe ESMC as trained on approximately 2.8 billion protein sequences drawn from across life.

The upstream model card reports that sequence data was clustered at 70% sequence identity, resulting in 83 million UniRef clusters, 372 million MGnify clusters, and 2 billion JGI clusters.

Note that [`ProteinTokenizer`][multimolecule.ProteinTokenizer] will tokenize protein sequences using the MultiMolecule protein vocabulary for you.

### Training Procedure

#### Preprocessing

ESMC tokenizes protein chains as amino-acid sequences and uses masked amino-acid prediction from surrounding sequence context. The released models use rotary position embeddings and were trained with context length up to 2048 tokens.

#### Pre-training

The upstream model card describes a two-stage training schedule.

- Stage 1: 1,000,000 steps with context length 512 and 64% metagenomic data.
- Stage 2: 500,000 steps with context length 2048 and 37.5% metagenomic data.
- Training FLOPs: 1.26e22 for ESMC-300M, 2.17e22 for ESMC-600M, and 2.37e23 for ESMC-6B.

## Citation

```bibtex
@UNPUBLISHED{Candido2026-wk,
  title       = "Language modeling materializes a world model of protein
                 biology",
  author      = "Candido, Salvatore and Hayes, Thomas and Derry, Alexander and
                 Rao, Roshan and Lin, Zeming and Verkuil, Robert and Wu, Bryan
                 Z and Lee, Jin Sub and Bruguera, Elise S and Keval, Jehan A
                 and Kopylov, Mykhailo and Pak, John E and Wu, Wesley and
                 Thomas, Neil and Mataraso, Samson and Hsu, Alvin and
                 Trotman-Grant, Ashton C and Fatras, Kilian and dos Santos
                 Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil and
                 Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth and
                 Sitwala, Hrishita and Yu, Yue and Wiggert, Marius and Carlin,
                 Dylan Alexander and Goering, Anthony W and Blazejewski, Tomasz
                 and Sandora, Mccullen and Hla, Michael and Jia, Tina Z and
                 Kloker, Leon H and Sofroniew, Nicholas J and Uehara, Masatoshi
                 and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S and
                 Sercu, Tom and Rives, Alexander",
  abstract    = "Abstract Proteins are fundamental to life. The full extent of
                 their biology is beyond our ability to characterize with
                 experimental approaches in the physical laboratory. Accurate
                 digital representations could accelerate the discovery of
                 protein biology through virtual experiments. We propose
                 language modeling to learn unified and general representations
                 that can be scaled to all of protein biology. Building on
                 these representations, we develop a structure prediction model
                 that exceeds the performance of established methods for
                 biomolecular complex prediction across benchmarks, including
                 for the interactions of antibodies with their targets. A
                 simple search procedure yields high experimental success rates
                 for the discovery of proteins with nanomolar binding
                 affinities for both miniproteins and single-chain antibodies,
                 a modality critical for therapeutic design. Study of the
                 concepts in the language model's representation space reveals
                 a systematic organization aligned with the reductionist
                 understanding of proteins developed through empirical science.
                 Leveraging this organization, we generate a comprehensive map
                 of protein biology encompassing over 6.8 billion sequences and
                 1.1 billion predicted structures, identifying connections
                 across known and unknown biology. As a whole, this shows
                 language modeling as a powerful substrate for representing the
                 biology of proteins, operating across scales from the
                 prediction and design of protein interactions at the atomic
                 level, to identifying properties of proteins at different
                 levels of granularity and abstraction, to the scale of mapping
                 connections between proteins across billions of years of
                 evolution.",
  journal     = "bioRxiv",
  institution = "bioRxiv",
  month       =  jun,
  year        =  2026,
  copyright   = "http://creativecommons.org/licenses/by/4.0/"
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

Please contact the authors of the [ESMC paper](https://doi.org/10.64898/2026.06.03.729735) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
