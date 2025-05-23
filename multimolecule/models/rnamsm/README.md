---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/rfam
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
widget:
  - example_title: "HIV-1"
    text: "GGUC<mask>CUCUGGUUAGACCAGAUCUGAGCCU"
    output:
      - label: "U"
        score: 0.25111356377601624
      - label: "W"
        score: 0.1200353354215622
      - label: "K"
        score: 0.10132723301649094
      - label: "D"
        score: 0.08383019268512726
      - label: "A"
        score: 0.05737845227122307
  - example_title: "microRNA-21"
    text: "UAGC<mask>UAUCAGACUGAUGUUG"
    output:
      - label: "U"
        score: 0.36363866925239563
      - label: "K"
        score: 0.2319803684949875
      - label: "G"
        score: 0.14799022674560547
      - label: "D"
        score: 0.062339700758457184
      - label: "W"
        score: 0.04046041890978813
---

# RNA-MSM

Pre-trained model on non-coding RNA (ncRNA) with multi (homologous) sequence alignment using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Multiple sequence alignment-based RNA language model and its application to structural inference](https://doi.org/10.1093/nar/gkad1031) by Yikun Zhang, Mei Lang, Jiuhong Jiang, Zhiqiang Gao, et al.

The OFFICIAL repository of RNA-MSM is at [yikunpku/RNA-MSM](https://github.com/yikunpku/RNA-MSM).

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RNA-MSM.
>
> The original implementation of RNA-MSM does not append `<eos>` token to the end of the input sequence in consistent to MSA Transformer.
> This should not affect the performance of the model in most cases, but it can lead to unexpected behavior in some cases.
>
> Please set `eos_token=None` in the tokenizer and set `eos_token_id=None` in the model configuration if you want the exact behavior of the original implementation.
>
> See more at [issue #10](https://github.com/yikunpku/RNA-MSM/issues/10)

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RNA-MSM did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RNA-MSM is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model.
RNA-MSM follows the [MSA](https://doi.org/10.1101/2021.02.12.430858) architecture, where it uses a column-wise attention and a row-wise attention to reduce the computational complexity over conventional self-attention.

RNA-MSM is pre-trained on a large corpus of non-coding RNA sequences with multiple sequence alignment (MSA) in a self-supervised fashion.
This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts.
Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | --------- | ----------------- | ------------------ | --------- | -------- | -------------- |
| 10         | 768         | 12        | 3072              | 95.92              | 21.66     | 10.57    | 1024           |

### Links

- **Code**: [multimolecule.rnamsm](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rnamsm)
- **Weights**: [multimolecule/rnamsm](https://huggingface.co/multimolecule/rnamsm)
- **Data**: [multimolecule/rfam](https://huggingface.co/datasets/multimolecule/rfam)
- **Paper**: [Multiple sequence alignment-based RNA language model and its application to structural inference](https://doi.org/10.1093/nar/gkad1031)
- **Developed by**: Yikun Zhang, Mei Lang, Jiuhong Jiang, Zhiqiang Gao, Fan Xu, Thomas Litfin, Ke Chen, Jaswinder Singh, Xiansong Huang, Guoli Song, Yonghong Tian, Jian Zhan, Jie Chen, Yaoqi Zhou
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [MSA](https://doi.org/10.1101/2021.02.12.430858)
- **Original Repository**: [yikunpku/RNA-MSM](https://github.com/yikunpku/RNA-MSM)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Masked Language Modeling

You can use this model directly with a pipeline for masked language modeling:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline

>>> unmasker = pipeline("fill-mask", model="multimolecule/rnamsm")
>>> unmasker("gguc<mask>cucugguuagaccagaucugagccu")
[{'score': 0.25111356377601624,
  'token': 9,
  'token_str': 'U',
  'sequence': 'G G U C U C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.1200353354215622,
  'token': 14,
  'token_str': 'W',
  'sequence': 'G G U C W C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.10132723301649094,
  'token': 15,
  'token_str': 'K',
  'sequence': 'G G U C K C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.08383019268512726,
  'token': 18,
  'token_str': 'D',
  'sequence': 'G G U C D C U C U G G U U A G A C C A G A U C U G A G C C U'},
 {'score': 0.05737845227122307,
  'token': 6,
  'token_str': 'A',
  'sequence': 'G G U C A C U C U G G U U A G A C C A G A U C U G A G C C U'}]
```

#### RNA Secondary Structure Prediction

You can use this model directly with a pipeline for secondary structure prediction:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline

>>> predictor = pipeline("rna-secondary-structure", model="multimolecule/rnamsm")
>>> predictor("ggucuc")
{'sequence': 'G G U C U C',
 'secondary_structure': '......',
 'contact_map': [[0.00261497194878757, 0.0022659720852971077, 0.0036333396565169096, 0.003973186947405338, 0.0034661777317523956, 0.0029443716630339622],
  [0.0022659730166196823, 0.002837304025888443, 0.003722205525264144, 0.0034382310695946217, 0.003498978214338422, 0.0030326189007610083],
  [0.0036333396565169096, 0.003722205525264144, 0.0026848132256418467, 0.002787571167573333, 0.0028246103320270777, 0.0030541368760168552],
  [0.003973186947405338, 0.0034382310695946217, 0.002787571167573333, 0.0028833637479692698, 0.0027405587024986744, 0.0029016658663749695],
  [0.0034661777317523956, 0.003498978214338422, 0.0028246103320270777, 0.0027405587024986744, 0.002930478658527136, 0.003173925681039691],
  [0.0029443716630339622, 0.0030326189007610083, 0.0030541368760168552, 0.0029016658663749695, 0.003173925681039691, 0.003476995974779129]]}
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, RnaMsmModel


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnamsm")
model = RnaMsmModel.from_pretrained("multimolecule/rnamsm")

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
from multimolecule import RnaTokenizer, RnaMsmForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnamsm")
model = RnaMsmForSequencePrediction.from_pretrained("multimolecule/rnamsm")

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
from multimolecule import RnaTokenizer, RnaMsmForTokenPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnamsm")
model = RnaMsmForNucleotidPrediction.from_pretrained("multimolecule/rnamsm")

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
from multimolecule import RnaTokenizer, RnaMsmForContactPrediction


tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnamsm")
model = RnaMsmForContactPrediction.from_pretrained("multimolecule/rnamsm")

text = "UAGCUUAUCAGACUGAUGUUG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

RNA-MSM used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The RNA-MSM model was pre-trained on [Rfam](https://rfam.org).
The Rfam database is a collection of RNA sequence families of structural RNAs including non-coding RNA genes as well as cis-regulatory elements.
RNA-MSM used Rfam 14.7 which contains 4,069 RNA families.

To avoid potential overfitting in structural inference, RNA-MSM excluded families with experimentally determined structures, such as ribosomal RNAs, transfer RNAs, and small nuclear RNAs. The final dataset contains 3,932 RNA families. The median value for the number of MSA sequences for these families by RNAcmap3 is 2,184.

To increase the number of homologous sequences, RNA-MSM used an automatic pipeline, RNAcmap3, for homolog search and sequence alignment. RNAcmap3 is a pipeline that combines the BLAST-N, INFERNAL, Easel, RNAfold and evolutionary coupling tools to generate homologous sequences.

RNA-MSM preprocessed all tokens by replacing "T"s with "U"s and substituting "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", "N" with "X".

Note that [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`. `RnaTokenizer` does not perform other substitutions.

### Training Procedure

#### Preprocessing

RNA-MSM used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### Pre-training

The model was trained on 8 NVIDIA V100 GPUs with 32GiB memories.

- Batch Size: 1
- Epochs: 300
- Optimizer: Adam
- Learning rate: 3e-4
- Learning rate warm-up: 16,000 steps
- Weight decay: 3e-4
- Dropout: 0.1

## Citation

**BibTeX**:

```bibtex
@article{zhang2023multiple,
    author = {Zhang, Yikun and Lang, Mei and Jiang, Jiuhong and Gao, Zhiqiang and Xu, Fan and Litfin, Thomas and Chen, Ke and Singh, Jaswinder and Huang, Xiansong and Song, Guoli and Tian, Yonghong and Zhan, Jian and Chen, Jie and Zhou, Yaoqi},
    title = "{Multiple sequence alignment-based RNA language model and its application to structural inference}",
    journal = {Nucleic Acids Research},
    volume = {52},
    number = {1},
    pages = {e3-e3},
    year = {2023},
    month = {11},
    abstract = "{Compared with proteins, DNA and RNA are more difficult languages to interpret because four-letter coded DNA/RNA sequences have less information content than 20-letter coded protein sequences. While BERT (Bidirectional Encoder Representations from Transformers)-like language models have been developed for RNA, they are ineffective at capturing the evolutionary information from homologous sequences because unlike proteins, RNA sequences are less conserved. Here, we have developed an unsupervised multiple sequence alignment-based RNA language model (RNA-MSM) by utilizing homologous sequences from an automatic pipeline, RNAcmap, as it can provide significantly more homologous sequences than manually annotated Rfam. We demonstrate that the resulting unsupervised, two-dimensional attention maps and one-dimensional embeddings from RNA-MSM contain structural information. In fact, they can be directly mapped with high accuracy to 2D base pairing probabilities and 1D solvent accessibilities, respectively. Further fine-tuning led to significantly improved performance on these two downstream tasks compared with existing state-of-the-art techniques including SPOT-RNA2 and RNAsnap2. By comparison, RNA-FM, a BERT-based RNA language model, performs worse than one-hot encoding with its embedding in base pair and solvent-accessible surface area prediction. We anticipate that the pre-trained RNA-MSM model can be fine-tuned on many other tasks related to RNA structure and function.}",
    doi = {10.1093/nar/gkad1031},
    url = {https://doi.org/10.1093/nar/gkad1031},
    eprint = {https://academic.oup.com/nar/article-pdf/52/1/e3/55443207/gkad1031.pdf},
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [RNA-MSM paper](https://doi.org/10.1093/nar/gkad1031) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
