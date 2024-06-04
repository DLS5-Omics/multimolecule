---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
  - multimolecule/rfam
  - multimolecule/ensembl-genome-browser
  - multimolecule/nucleotide
library_name: multimolecule
---

# RiNALMo

Pre-trained model on non-coding RNA (ncRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks](https://doi.org/10.48550/arXiv.2403.00043) by Rafael Josip Penić, et al.

The OFFICIAL repository of RiNALMo is at [lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo).

**The team releasing RiNALMo did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RiNALMo is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of non-coding RNA sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variations

- **[`multimolecule/rinalmo`](https://huggingface.co/multimolecule/rinalmo)**: The pre-trained RiNALMo-GiGa model.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variants</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
    <th>max token length</th>
  </tr></thead>
<tbody>
  <tr>
    <td>RiNALMo</td>
    <td>650.88</td>
    <td>168.92</td>
    <td>84.43</td>
    <td>1022</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.rinalmo](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rinalmo)
- **Data**: [RNAcentral](https://rnacentral.org)
- **Paper**: [RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks](https://doi.org/10.48550/arXiv.2403.00043)
- **Developed by**: Rafael Josip Penić, Tin Vlašić, Roland G. Huber, Yue Wan, Mile Šikić
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [https://github.com/lbcb-sci/RiNALMo](https://github.com/lbcb-sci/RiNALMo)

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
>>> unmasker = pipeline('fill-mask', model='multimolecule/rinalmo')
>>> unmasker("uagc<mask>uaucagacugauguuga")

[{'score': 0.28896641731262207,
  'token': 6,
  'token_str': 'A',
  'sequence': 'U A G C A U A U C A G A C U G A U G U U G A'},
 {'score': 0.27602624893188477,
  'token': 9,
  'token_str': 'U',
  'sequence': 'U A G C U U A U C A G A C U G A U G U U G A'},
 {'score': 0.18329711258411407,
  'token': 12,
  'token_str': 'X',
  'sequence': 'U A G C X U A U C A G A C U G A U G U U G A'},
 {'score': 0.1668907254934311,
  'token': 7,
  'token_str': 'C',
  'sequence': 'U A G C C U A U C A G A C U G A U G U U G A'},
 {'score': 0.08479981869459152,
  'token': 8,
  'token_str': 'G',
  'sequence': 'U A G C G U A U C A G A C U G A U G U U G A'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RiNALMoModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model = RiNALMoModel.from_pretrained('multimolecule/rinalmo')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model = RiNALMoForSequencePrediction.from_pretrained('multimolecule/rinalmo')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.tensor([1])

output = model(**input, labels=label)
```

#### Nucleotide Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for nucleotide classification or regression.

Here is how to use this model as backbone to fine-tune for a nucleotide-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model = RiNALMoForNucleotidePrediction.from_pretrained('multimolecule/rinalmo')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

#### Contact Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

Here is how to use this model as backbone to fine-tune for a contact-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
model = RiNALMoForContactPrediction.from_pretrained('multimolecule/rinalmo')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RiNALMo used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RiNALMo model was pre-trained on a cocktail of databases including [RNAcentral](https://rnacentral.org), [Rfam](https://rfam.org), [Ensembl Genome Browser](https://ensembl.org), and [Nucleotide](https://ncbi.nlm.nih.gov/nucleotide). The training data contains 36 million unique ncRNA sequences.

To ensure sequence diversity in each training batch, RiNALMo clustered the sequences with [MMSeqs2](https://github.com/soedinglab/MMseqs2) into 17 million clusters and then sampled each sequence in the batch from a different cluster.

RiNALMo preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RiNALMo used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

The model was trained on 7 NVIDIA A100 GPUs with 80GiB memories.

- Learning rate: 5e-5
- Learning rate scheduler: cosine
- Learning rate warm-up: 2,000 steps
- Learning rate minimum: 1e-5
- Epochs: 6
- Batch Size: 1344
- Dropout: 0.1

## Citation

**BibTeX**:

```bibtex
@article{penic2024rinalmo,
  title={RiNALMo: General-Purpose RNA Language Models Can Generalize Well on Structure Prediction Tasks},
  author={Penić, Rafael Josip and Vlašić, Tin and Huber, Roland G. and Wan, Yue and Šikić, Mile},
  journal={arXiv preprint arXiv:2403.00043},
  year={2024}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [RiNALMo paper](https://doi.org/10.48550/arXiv.2403.00043) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
