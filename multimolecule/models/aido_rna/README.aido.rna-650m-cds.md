---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/ena
library_name: multimolecule
base_model: multimolecule/aido.rna-650m
pipeline_tag: fill-mask
mask_token: "<mask>"
widget:
  - example_title: "HIV-1"
    text: "GGUC<mask>CUCUGGUUAGACCAGAUCUGAGCCU"
    output:
      - label: "A"
        score: 0.15881139039993286
      - label: "R"
        score: 0.15044376254081726
      - label: "G"
        score: 0.14251668751239777
      - label: "V"
        score: 0.1298484206199646
      - label: "M"
        score: 0.1239432692527771
  - example_title: "microRNA-21"
    text: "UAGC<mask>UAUCAGACUGAUGUUG"
    output:
      - label: "A"
        score: 0.1757601946592331
      - label: "M"
        score: 0.1494324952363968
      - label: "R"
        score: 0.1302214413881302
      - label: "V"
        score: 0.1291552037000656
      - label: "C"
        score: 0.12704865634441376
---

# AIDO.RNA

Pre-trained model on non-coding RNA (ncRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://doi.org/10.1101/2024.11.28.625345) by Shuxian Zou, Tianhua Tao, Sazan Mahbub, et al.

The OFFICIAL repository of AIDO.RNA is at [genbio-ai/AIDO](https://github.com/genbio-ai/AIDO).

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of AIDO.RNA.
>
> The original implementation of AIDO.RNA uses a special tokenizer that identifies `U` and `T` as different tokens.
>
> This behaviour is not supported by MultiMolecule.

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing AIDO.RNA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

AIDO.RNA is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/aido.rna-650m](https://huggingface.co/multimolecule/aido.rna-650m)**: The AIDO.RNA model with 650 million parameters.
- **[multimolecule/aido.rna-1.6b](https://huggingface.co/multimolecule/aido.rna-1.6b)**: The AIDO.RNA model with 1.6 billion parameters.

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
    <td>AIDO.RNA-650M</td>
    <td>33</td>
    <td>1280</td>
    <td>20</td>
    <td>3392</td>
    <td>648.38</td>
    <td>168.25</td>
    <td>80.09</td>
    <td rowspan="2">1022</td>
  </tr>
  <tr>
    <td>AIDO.RNA-1.6B</td>
    <td>32</td>
    <td>2048</td>
    <td>32</td>
    <td>5440</td>
    <td>1650.29</td>
    <td>415.67</td>
    <td>207.77</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.aido_rna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/aido_rna)
- **Weights**: [multimolecule/aido.rna](https://huggingface.co/multimolecule/aido.rna)
- **Data**: [multimolecule/rnacentral](https://huggingface.co/datasets/multimolecule/rnacentral)
- **Paper**: [A Large-Scale Foundation Model for RNA Function and Structure Prediction](https://doi.org/10.1101/2024.11.28.625345)
- **Developed by**: Shuxian Zou, Tianhua Tao, Sazan Mahbub, Caleb N. Ellington, Robin Algayres, Dian Li, Yonghao Zhuang, Hongyi Wang, Le Song, Eric P. Xing
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [genbio-ai/AIDO](https://github.com/genbio-ai/AIDO)

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

>>> unmasker = pipeline("fill-mask", model="multimolecule/aido.rna-650m")
>>> unmasker("gguc<mask>cucugguuagaccagaucugagccu")
[{'score': 0.15881139039993286,
  'token': 6,
  'token_str': 'A',
  'sequence': 'G G U C A C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.15044376254081726,
  'token': 11,
  'token_str': 'R',
  'sequence': 'G G U C R C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.14251668751239777,
  'token': 8,
  'token_str': 'G',
  'sequence': 'G G U C G C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.1298484206199646,
  'token': 20,
  'token_str': 'V',
  'sequence': 'G G U C V C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.1239432692527771,
  'token': 16,
  'token_str': 'M',
  'sequence': 'G G U C M C U C U G G U U A G A C C A G A U C U G A G C C U'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, AidoRnaModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/aido.rna-650m")
model = AidoRnaModel.from_pretrained("multimolecule/aido.rna-650m")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")

output = model(**input)
```

#### Sequence Classification / Regression

> [!NOTE]
> This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, AidoRnaForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/aido.rna-650m")
model = AidoRnaForSequencePrediction.from_pretrained("multimolecule/aido.rna-650m")

text = "UAGCUUAUCAGACUGAUGUUG"
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
from multimolecule import RnaTokenizer, AidoRnaForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/aido.rna-650m")
model = AidoRnaForTokenPrediction.from_pretrained("multimolecule/aido.rna-650m")

text = "UAGCUUAUCAGACUGAUGUUG"
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
from multimolecule import RnaTokenizer, AidoRnaForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/aido.rna-650m")
model = AidoRnaForContactPrediction.from_pretrained("multimolecule/aido.rna-650m")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

AIDO.RNA used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The AIDO.RNA model was pre-trained on [RNAcentral](https://multimolecule.danling.org/datasets/rnacentral) and [MARS](https://ngdc.cncb.ac.cn/omix/release/OMIX003037).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

AIDO.RNA applied SeqKit to remove duplicated sequences in the RNAcentral, resulting 42 million unique sequences.

Note that AIDO.RNA identifies `U` and `T` as different tokens, which is not supported by MultiMolecule. During model conversion, the embeddings of `T` is discarded. This means that the model will not be able to distinguish between `U` and `T` in the input sequences.

### Training Procedure

#### Preprocessing

AIDO.RNA used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

- Epochs: 6
- Optimizer: AdamW
- Learning rate: 5e-5
- Learning rate warm-up: 2,000 steps
- Learning rate scheduler: Cosine
- Minimum learning rate: 1e-5
- Weight decay: 0.01

## Citation

**BibTeX**:

```bibtex
@article {Zou2024.11.28.625345,
	author = {Zou, Shuxian and Tao, Tianhua and Mahbub, Sazan and Ellington, Caleb N. and Algayres, Robin and Li, Dian and Zhuang, Yonghao and Wang, Hongyi and Song, Le and Xing, Eric P.},
	title = {A Large-Scale Foundation Model for RNA Function and Structure Prediction},
	elocation-id = {2024.11.28.625345},
	year = {2024},
	doi = {10.1101/2024.11.28.625345},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Originally marginalized as an intermediate in the information flow from DNA to protein, RNA has become the star of modern biology, holding the key to precision therapeutics, genetic engineering, evolutionary origins, and our understanding of fundamental cellular processes. Yet RNA is as mysterious as it is prolific, serving as an information store, a messenger, and a catalyst, spanning many underchar-acterized functional and structural classes. Deciphering the language of RNA is important not only for a mechanistic understanding of its biological functions but also for accelerating drug design. Toward this goal, we introduce AIDO.RNA, a pre-trained module for RNA in an AI-driven Digital Organism [1]. AIDO.RNA contains a scale of 1.6 billion parameters, trained on 42 million non-coding RNA (ncRNA) sequences at single-nucleotide resolution, and it achieves state-of-the-art performance on a comprehensive set of tasks, including structure prediction, genetic regulation, molecular function across species, and RNA sequence design. AIDO.RNA after domain adaptation learns to model essential parts of protein translation that protein language models, which have received widespread attention in recent years, do not. More broadly, AIDO.RNA hints at the generality of biological sequence modeling and the ability to leverage the central dogma to improve many biomolecular representations. Models and code are available through ModelGenerator in https://github.com/genbio-ai/AIDO and on Hugging Face.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/11/29/2024.11.28.625345},
	eprint = {https://www.biorxiv.org/content/early/2024/11/29/2024.11.28.625345.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [AIDO.RNA paper](https://doi.org/10.1101/2024.11.28.625345) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
