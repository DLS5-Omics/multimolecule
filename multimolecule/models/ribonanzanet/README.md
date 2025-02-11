---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/ribonanza
library_name: multimolecule
---

> [!IMPORTANT]
> This model is in a future release of MultiMolecule, and is under development.
> This model card is not final and will be updated in the future.

# RibonanzaNet

Pre-trained model on RNA chemical mapping for modeling RNA structure and other properties.

## Disclaimer

This is an UNOFFICIAL implementation of the [Ribonanza: deep learning of RNA structure through dual crowdsourcing](https://doi.org/10.1101/2024.02.24.581671) by Shujun He, Rui Huang, et al.

The OFFICIAL repository of RibonanzaNet is at [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet).

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet applied `dropout-residual-norm` path twice to the output of the Self-Attention layer.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_norm=True` in the model configuration to apply the `dropout-residual-norm` path once.
>
> See more at [issue #3](https://github.com/Shujun-He/RibonanzaNet/issues/3)

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet does not apply attention mask correctly.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_mask=True` in the model configuration to apply the correct attention mask.
>
> See more at [issue #4](https://github.com/Shujun-He/RibonanzaNet/issues/4), [issue #5](https://github.com/Shujun-He/RibonanzaNet/issues/5), and [issue #7](https://github.com/Shujun-He/RibonanzaNet/issues/7)

> [!CAUTION]
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

RibonanzaNet is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of RNA sequences. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 9          | 256         | 8         | 1024              | 11.37              | 107.31    | 53.32    | inf            |

### Links

- **Code**: [multimolecule.ribonanzanet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ribonanzanet)
- **Weights**: [`multimolecule/ribonanzanet`](https://huggingface.co/multimolecule/ribonanzanet)
- **Data**: [Ribonanza](https://multimolecule.danling.org/datasets/ribonanza)
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

You can use this model directly to predict chemical mapping:

```python
>>> from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
>>> model = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
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

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

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

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for nucleotide classification or regression.

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

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for contact classification or regression.

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

RibonanzaNet used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RibonanzaNet model was pre-trained on [Ribonanza](https://multimolecule.danling.org/datasets/ribonanza).
RNAcentral is a free, public resource that offers integrated access to a comprehensive and up-to-date set of non-coding RNA sequences provided by a collaborating group of [Expert Databases](https://rnacentral.org/expert-databases) representing a broad range of organisms and RNA types.

RibonanzaNet applied [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) with a cut-off at 100% sequence identity to remove redundancy from the RNAcentral. The final dataset contains 23.7 million non-redundant RNA sequences.

RibonanzaNet preprocessed all tokens by replacing "U"s with "T"s.

Note that during model conversions, "T" is replaced with "U". [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

RibonanzaNet used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

The model was trained on 10 NVIDIA L40S GPUs with 48GiB memories.

- Learning rate: 1e-4
- Weight decay: 0.01

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
