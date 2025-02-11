---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/ribonanza
library_name: multimolecule
base_model: multimolecule/ribonanzanet
---

# RibonanzaNet

> [!IMPORTANT]
> This model is in a future release of MultiMolecule, and is under development.
> This model card is not final and will be updated in the future.

Pre-trained model on RNA chemical mapping for modeling RNA structure and other properties.

## Disclaimer

This is an UNOFFICIAL implementation of the [Ribonanza: deep learning of RNA structure through dual crowdsourcing](https://doi.org/10.1101/2024.02.24.581671) by Shujun He, Rui Huang, et al.

The OFFICIAL repository of RibonanzaNet is at [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet).

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet does not prepend `<cls>` and append `<eos>` tokens to the input sequence.
> This should not affect the performance of the model in most cases, but it can lead to unexpected behavior in some cases.
>
> Please set `bos_token=None, cls_token=None, eos_token = None` in the tokenizer and set `bos_token_id=None, cls_token_id=None, eos_token_id=None` in the model configuration if you want the exact behavior of the original implementation.

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet applied `dropout-residual-norm` path twice to the output of the Self-Attention layer.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_norm=True` in the model configuration to apply the `dropout-residual-norm` path once.
>
> See more at [issue #3](https://github.com/Shujun-He/RibonanzaNet/issues/3)

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet does not apply attention mask correctly.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_mask=True` in the model configuration to apply the correct attention mask.
>
> See more at [issue #4](https://github.com/Shujun-He/RibonanzaNet/issues/4), [issue #5](https://github.com/Shujun-He/RibonanzaNet/issues/5), and [issue #7](https://github.com/Shujun-He/RibonanzaNet/issues/7)

> [!WARNING]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet applies dropout in an axis different from the one described in the paper.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_pairwise_dropout=True` in the model configuration to follow the description in the paper.
>
> See more at [issue #6](https://github.com/Shujun-He/RibonanzaNet/issues/6)

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RibonanzaNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RibonanzaNet is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model.
RibonanzaNet follows the modification from the [RNAdegformer](https://academic.oup.com/bib/article/24/1/bbac581/6986359) where it introduces a 1D convolution with residual connection at the beginning of each encoder layer.
Different from RNAdegformer, RibonanzaNet does not apply deconvolution at the end of the encoder layers, and updates the pairwise representation through outer product mean and triangular update.

RibonanzaNet is pre-trained on a large corpus of RNA sequences with chemical mapping (2A3 and DMS) measurements.
Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 9          | 256         | 8         | 1024              | 11.37              | 53.65     | 26.66    | inf            |

### Links

- **Code**: [multimolecule.ribonanzanet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ribonanzanet)
- **Weights**: [multimolecule/ribonanzanet](https://huggingface.co/multimolecule/ribonanzanet)
- **Data**: [multimolecule/ribonanza](https://multimolecule.danling.org/datasets/ribonanza)
- **Paper**: [Ribonanza: deep learning of RNA structure through dual crowdsourcing](https://doi.org/10.1101/2024.02.24.581671)
- **Developed by**: Shujun He, Rui Huang, Jill Townley, Rachael C. Kretsch, Thomas G. Karagianes, David B.T. Cox, Hamish Blair, Dmitry Penzar, Valeriy Vyaltsev, Elizaveta Aristova, Arsenii Zinkevich, Artemy Bakulin, Hoyeol Sohn, Daniel Krstevski, Takaaki Fukui, Fumiya Tatematsu, Yusuke Uchida, Donghoon Jang, Jun Seong Lee, Roger Shieh, Tom Ma, Eduard Martynov, Maxim V. Shugaev, Habib S.T. Bukhari, Kazuki Fujikawa, Kazuki Onodera, Christof Henkel, Shlomo Ron, Jonathan Romano, John J. Nicol, Grace P. Nye, Yuan Wu, Christian Choe, Walter Reade, Eterna participants, Rhiju Das
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

This model is fine-tuned for RNA degradation prediction.

You can use this model directly to predict the degradation of an RNA sequence:

```python
>>> from multimolecule import RnaTokenizer, RibonanzaNetForDegradationPrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
>>> model = RibonanzaNetForDegradationPrediction.from_pretrained("multimolecule/ribonanzanet")
>>> output = model(**tokenizer("agcagucauuauggcgaa", return_tensors="pt"))

>>> output.keys()
odict_keys(['logits_reactivity', 'logits_deg_Mg_pH10', 'logits_deg_pH10', 'logits_deg_Mg_50C', 'logits_deg_50C'])

>>> output.logits_reactivity.squeeze()
tensor([0.4037, 0.2204, 0.0287, 0.1867, 0.5964, 0.3698, 0.2202, 0.7558, 1.3360,
        0.6030, 0.6923, 0.4518, 0.2663, 0.2863, 0.1949, 0.4807, 0.5269, 1.1370],
       grad_fn=<SqueezeBackward0>)
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RibonanzaNetModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
model = RibonanzaNetModel.from_pretrained("multimolecule/ribonanzanet")

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
from multimolecule import RnaTokenizer, RibonanzaNetForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
model = RibonanzaNetForSequencePrediction.from_pretrained("multimolecule/ribonanzanet")

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
from multimolecule import RnaTokenizer, RibonanzaNetForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
model = RibonanzaNetForTokenPrediction.from_pretrained("multimolecule/ribonanzanet")

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
from multimolecule import RnaTokenizer, RibonanzaNetForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
model = RibonanzaNetForContactPrediction.from_pretrained("multimolecule/ribonanzanet")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RibonanzaNet used chemical mapping data as the training objective. The model takes an RNA sequence as input and predicts the chemical reactivity of each nucleotide.

### Training Data

The RibonanzaNet model was trained on the Ribonanza dataset. Ribonanza is a dataset of chemical mapping measurements on two million diverse RNA sequences. The data was collected from the crowdsourced initiative Eterna, as well as expert databases such as [Rfam](https://huggingface.co/datasets/multimolecule/rfam), the PDB archive, Pseudobase, and the RNA Mapping Database.

### Training Procedure

RibonanzaNet was trained using a three-stage process:

#### Initial Training

The initial model was trained using sequences that had either or both 2A3/DMS profiles with a signal-to-noise ratio (SNR) above 1.0. This dataset comprised 214,831 training sequences.

#### Pre-training

1. Noisy Training Data: The model was first pre-trained on the data with a signal-to-noise ratio (SNR) below 1.0 using predictions from top 3 Kaggle models as pseudo-labels. This dataset comprised 563,796 sequences.
2. Experimental Determined Data: The model was then further trained for 10 epochs using only the true labels of sequences with high SNR (either 2A3 or DMS profiles).

#### Final Training

1. Noisy Training Data: The model was first pre-trained on all training and testing data using predictions from top 3 Kaggle models as pseudo-labels. This dataset comprised 1,907,619 sequences.
2. Experimental Determined Data: The model was then further annelaed on the true training labels.

The model was trained on 10 NVIDIA L40S GPUs with 48GiB memories.

Sequence flip augmentation was applied to the training data.

## Citation

**BibTeX**:

```bibtex
@article{He2024.02.24.581671,
  author       = {He, Shujun and Huang, Rui and Townley, Jill and Kretsch, Rachael C. and Karagianes, Thomas G. and Cox, David B.T. and Blair, Hamish and Penzar, Dmitry and Vyaltsev, Valeriy and Aristova, Elizaveta and Zinkevich, Arsenii and Bakulin, Artemy and Sohn, Hoyeol and Krstevski, Daniel and Fukui, Takaaki and Tatematsu, Fumiya and Uchida, Yusuke and Jang, Donghoon and Lee, Jun Seong and Shieh, Roger and Ma, Tom and Martynov, Eduard and Shugaev, Maxim V. and Bukhari, Habib S.T. and Fujikawa, Kazuki and Onodera, Kazuki and Henkel, Christof and Ron, Shlomo and Romano, Jonathan and Nicol, John J. and Nye, Grace P. and Wu, Yuan and Choe, Christian and Reade, Walter and Eterna participants and Das, Rhiju},
  title        = {Ribonanza: deep learning of RNA structure through dual crowdsourcing},
  elocation-id = {2024.02.24.581671},
  year         = {2024},
  doi          = {10.1101/2024.02.24.581671},
  publisher    = {Cold Spring Harbor Laboratory},
  abstract     = {Prediction of RNA structure from sequence remains an unsolved problem, and progress has been slowed by a paucity of experimental data. Here, we present Ribonanza, a dataset of chemical mapping measurements on two million diverse RNA sequences collected through Eterna and other crowdsourced initiatives. Ribonanza measurements enabled solicitation, training, and prospective evaluation of diverse deep neural networks through a Kaggle challenge, followed by distillation into a single, self-contained model called RibonanzaNet. When fine tuned on auxiliary datasets, RibonanzaNet achieves state-of-the-art performance in modeling experimental sequence dropout, RNA hydrolytic degradation, and RNA secondary structure, with implications for modeling RNA tertiary structure.Competing Interest StatementStanford University is filing patent applications based on concepts described in this paper. R.D. is a cofounder of Inceptive.},
  url          = {https://www.biorxiv.org/content/early/2024/06/11/2024.02.24.581671},
  eprint       = {https://www.biorxiv.org/content/early/2024/06/11/2024.02.24.581671.full.pdf},
  journal      = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [RibonanzaNet paper](https://doi.org/10.1101/2024.02.24.581671) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
