---
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/gencode-human
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
---

# DNABERT

Pre-trained model on human genome using a masked language modeling (MLM) objective with k-mer tokenization.

## Disclaimer

This is an UNOFFICIAL implementation of the [DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome](https://doi.org/10.1093/bioinformatics/btab083) by Yanrong Ji, Zhihan Zhou, et al.

The OFFICIAL repository of DNABERT is at [jerryji1993/DNABERT](https://github.com/jerryji1993/DNABERT).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DNABERT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DNABERT is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on the human genome with k-mer tokenization in a self-supervised fashion. This means that the model was trained on the raw nucleotides of DNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

- **[multimolecule/dnabert-3mer](https://huggingface.co/multimolecule/dnabert-3mer)**: The DNABERT model pre-trained on 3-mer data.
- **[multimolecule/dnabert-4mer](https://huggingface.co/multimolecule/dnabert-4mer)**: The DNABERT model pre-trained on 4-mer data.
- **[multimolecule/dnabert-5mer](https://huggingface.co/multimolecule/dnabert-5mer)**: The DNABERT model pre-trained on 5-mer data.
- **[multimolecule/dnabert-6mer](https://huggingface.co/multimolecule/dnabert-6mer)**: The DNABERT model pre-trained on 6-mer data.

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
    <td>dnabert-6mer</td>
    <td rowspan="4">12</td>
    <td rowspan="4">768</td>
    <td rowspan="4">12</td>
    <td rowspan="4">3072</td>
    <td>89.19</td>
    <td rowspan="4">96.86</td>
    <td rowspan="4">48.43</td>
    <td rowspan="4">512</td>
  </tr>
  <tr>
    <td>dnabert-5mer</td>
    <td>86.83</td>
  </tr>
  <tr>
    <td>dnabert-4mer</td>
    <td>86.24</td>
  </tr>
  <tr>
    <td>dnabert-3mer</td>
    <td>86.10</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.dnabert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/dnabert)
- **Data**: [multimolecule/gencode-human](https://huggingface.co/datasets/multimolecule/gencode-human)
- **Paper**: [DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome](https://doi.org/10.1093/bioinformatics/btab083)
- **Developed by**: Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repositories**: [jerryji1993/DNABERT](https://github.com/jerryji1993/DNABERT)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Masked Language Modeling

> [!WARNING]
> Default transformers pipeline does not support K-mer tokenization.

You can use this model directly with a pipeline for masked language modeling:

```python
import multimolecule  # you must import multimolecule to register models
from transformers import pipeline

predictor = pipeline("fill-mask", model="multimolecule/dnabert")
output = predictor("ATCG<mask>TGCA")
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import DnaBertModel
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert")
model = DnaBertModel.from_pretrained("multimolecule/dnabert")

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
from multimolecule import DnaBertForSequencePrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert")
model = DnaBertForSequencePrediction.from_pretrained("multimolecule/dnabert")

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
from multimolecule import DnaBertForTokenPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert")
model = DnaBertForTokenPrediction.from_pretrained("multimolecule/dnabert")

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
from multimolecule import DnaBertForContactPrediction
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("multimolecule/dnabert")
model = DnaBertForContactPrediction.from_pretrained("multimolecule/dnabert")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

DNABERT used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The DNABERT model was pre-trained on the human genome. The training data consists of DNA sequences from the human reference genome (GRCh38.p13), with all sequences containing only the four canonical nucleotides (A, T, C, G).

### Training Procedure

#### Preprocessing

DNABERT used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked. In the last 20,000 steps, the masking rate is increased to 20%.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

Since DNABERT used k-mer tokenizer, it masks the entire k-mer instead of individual nucleotides to avoid information leakage.

For example, if the k-mer is 3, the sequence `"TAGCGTAT"` will be tokenized as `["TAG", "AGC", "GCG", "CGT", "GTA", "TAT"]`. If the nucleotide `"C"` is masked, the adjacent tokens will also be masked, resulting `["TAG", "<mask>", "<mask>", "<mask>", "GTA", "TAT"]`.

#### Pre-training

The model was trained on 8 NVIDIA RTX 2080Ti GPUs.

- Batch size: 2,000
- Steps: 120,000
- Learning rate: 4e-4
- Learning rate scheduler: Linear
- Learning rate warm-up: 10,000 steps

## Citation

```bibtex
@ARTICLE{Ji2021-cj,
  title     = "{DNABERT}: pre-trained Bidirectional Encoder Representations
               from Transformers model for {DNA-language} in genome",
  author    = "Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V",
  abstract  = "MOTIVATION: Deciphering the language of non-coding DNA is one of
               the fundamental problems in genome research. Gene regulatory
               code is highly complex due to the existence of polysemy and
               distant semantic relationship, which previous informatics
               methods often fail to capture especially in data-scarce
               scenarios. RESULTS: To address this challenge, we developed a
               novel pre-trained bidirectional encoder representation, named
               DNABERT, to capture global and transferrable understanding of
               genomic DNA sequences based on up and downstream nucleotide
               contexts. We compared DNABERT to the most widely used programs
               for genome-wide regulatory elements prediction and demonstrate
               its ease of use, accuracy and efficiency. We show that the
               single pre-trained transformers model can simultaneously achieve
               state-of-the-art performance on prediction of promoters, splice
               sites and transcription factor binding sites, after easy
               fine-tuning using small task-specific labeled data. Further,
               DNABERT enables direct visualization of nucleotide-level
               importance and semantic relationship within input sequences for
               better interpretability and accurate identification of conserved
               sequence motifs and functional genetic variant candidates.
               Finally, we demonstrate that pre-trained DNABERT with human
               genome can even be readily applied to other organisms with
               exceptional performance. We anticipate that the pre-trained
               DNABERT model can be fined tuned to many other sequence analyses
               tasks. AVAILABILITY AND IMPLEMENTATION: The source code,
               pretrained and finetuned model for DNABERT are available at
               GitHub (https://github.com/jerryji1993/DNABERT). SUPPLEMENTARY
               INFORMATION: Supplementary data are available at Bioinformatics
               online.",
  journal   = "Bioinformatics",
  publisher = "Oxford University Press (OUP)",
  volume    =  37,
  number    =  15,
  pages     = "2112--2120",
  month     =  aug,
  year      =  2021,
  copyright = "https://academic.oup.com/journals/pages/open\_access/funder\_policies/chorus/standard\_publication\_model",
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

Please contact the authors of the [DNABERT paper](https://doi.org/10.1093/bioinformatics/btab083) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
