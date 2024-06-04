---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/ucsc-genome-browser
library_name: multimolecule
---

# SpliceBERT

Pre-trained model on messenger RNA precursor (pre-mRNA) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction](https://doi.org/10.1101/2023.01.31.526427) by Ken Chen, et al.

The OFFICIAL repository of SpliceBERT is at [chenkenbio/SpliceBERT](https://github.com/chenkenbio/SpliceBERT).

**The team releasing SpliceBERT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

SpliceBERT is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of messenger RNA precursor sequences in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variations

- **[`multimolecule/splicebert`](https://huggingface.co/multimolecule/splicebert)**: The SpliceBERT model.
- **[`multimolecule/splicebert.510nt`](https://huggingface.co/multimolecule/splicebert.510nt)**: The intermediate SpliceBERT model.
- **[`multimolecule/splicebert-human.510nt`](https://huggingface.co/multimolecule/splicebert-human.510nt)**: The intermediate SpliceBERT model pre-trained on human data only.

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
    <td>splicebert</td>
    <td rowspan="3">6</td>
    <td rowspan="3">512</td>
    <td rowspan="3">16</td>
    <td rowspan="3">2048</td>
    <td>19.72</td>
    <td rowspan="3">5.04</td>
    <td rowspan="3">2.52</td>
    <td>1024</td>
  </tr>
  <tr>
    <td>splicebert.510nt</td>
    <td rowspan="2">19.45</td>
    <td rowspan="2">510</td>
  </tr>
  <tr>
    <td>splicebert-human.510nt</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.splicebert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/splicebert)
- **Data**: [UCSC Genome Browser](https://genome.ucsc.edu)
- **Paper**: [Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction](https://doi.org/10.1101/2023.01.31.526427)
- **Developed by**: Ken Chen, Yue Zhou, Maolin Ding, Yu Wang, Zhixiang Ren, Yuedong Yang
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [FlashAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)
- **Original Repository**: [https://github.com/chenkenbio/SpliceBERT](https://github.com/chenkenbio/SpliceBERT)

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
>>> unmasker = pipeline('fill-mask', model='multimolecule/splicebert')
>>> unmasker("uagc<mask>uaucagacugauguuga")

[{'score': 0.09628374129533768,
  'token': 6,
  'token_str': 'A',
  'sequence': 'U A G C A U A U C A G A C U G A U G U U G A'},
 {'score': 0.09019321203231812,
  'token': 19,
  'token_str': 'W',
  'sequence': 'U A G C W U A U C A G A C U G A U G U U G A'},
 {'score': 0.08448788523674011,
  'token': 9,
  'token_str': 'U',
  'sequence': 'U A G C U U A U C A G A C U G A U G U U G A'},
 {'score': 0.07201363891363144,
  'token': 14,
  'token_str': 'H',
  'sequence': 'U A G C H U A U C A G A C U G A U G U U G A'},
 {'score': 0.06648518145084381,
  'token': 17,
  'token_str': 'M',
  'sequence': 'U A G C M U A U C A G A C U G A U G U U G A'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, SpliceBertModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/splicebert')
model = SpliceBertModel.from_pretrained('multimolecule/splicebert')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, SpliceBertForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/splicebert')
model = SpliceBertForSequencePrediction.from_pretrained('multimolecule/splicebert')

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
from multimolecule import RnaTokenizer, SpliceBertForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/splicebert')
model = SpliceBertForNucleotidePrediction.from_pretrained('multimolecule/splicebert')

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
from multimolecule import RnaTokenizer, SpliceBertForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/splicebert')
model = SpliceBertForContactPrediction.from_pretrained('multimolecule/splicebert')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

SpliceBERT used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The SpliceBERT model was pre-trained on messenger RNA precursor sequences from [UCSC Genome Browser](https://genome.ucsc.edu). UCSC Genome Browser provides visualization, analysis, and download of comprehensive vertebrate genome data with aligned annotation tracks (known genes, predicted genes, ESTs, mRNAs, CpG islands, etc.).

SpliceBERT collected reference genomes and gene annotations from the UCSC Genome Browser for 72 vertebrate species. It applied [bedtools getfasta](https://bedtools.readthedocs.io/en/latest/content/tools/getfasta.html) to extract pre-mRNA sequences from the reference genomes based on the gene annotations. The pre-mRNA sequences are then used to pre-train SpliceBERT. The pre-training data contains 2 million pre-mRNA sequences with a total length of 65 billion nucleotides.

Note [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

SpliceBERT used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

#### PreTraining

The model was trained on 8 NVIDIA V100 GPUs.

- Learning rate: 1e-4
- Learning rate scheduler: ReduceLROnPlateau(patience=3)
- Optimizer: AdamW

SpliceBERT trained model in a two-stage training process:

1. Pre-train with sequences of a fixed length of 510 nucleotides.
2. Pre-train with sequences of a variable length between 64 and 1024 nucleotides.

The intermediate model after the first stage is available as `multimolecule/splicebert.510nt`.

SpliceBERT also pre-trained a model on human data only to validate the contribution of multi-species pre-training. The intermediate model after the first stage is available as `multimolecule/splicebert-human.510nt`.

## Citation

**BibTeX**:

```bibtex
@article {chen2023self,
	author = {Chen, Ken and Zhou, Yue and Ding, Maolin and Wang, Yu and Ren, Zhixiang and Yang, Yuedong},
	title = {Self-supervised learning on millions of pre-mRNA sequences improves sequence-based RNA splicing prediction},
	elocation-id = {2023.01.31.526427},
	year = {2023},
	doi = {10.1101/2023.01.31.526427},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {RNA splicing is an important post-transcriptional process of gene expression in eukaryotic cells. Predicting RNA splicing from primary sequences can facilitate the interpretation of genomic variants. In this study, we developed a novel self-supervised pre-trained language model, SpliceBERT, to improve sequence-based RNA splicing prediction. Pre-training on pre-mRNA sequences from vertebrates enables SpliceBERT to capture evolutionary conservation information and characterize the unique property of splice sites. SpliceBERT also improves zero-shot prediction of variant effects on splicing by considering sequence context information, and achieves superior performance for predicting branchpoint in the human genome and splice sites across species. Our study highlighted the importance of pre-training genomic language models on a diverse range of species and suggested that pre-trained language models were promising for deciphering the sequence logic of RNA splicing.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/05/09/2023.01.31.526427},
	eprint = {https://www.biorxiv.org/content/early/2023/05/09/2023.01.31.526427.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [SpliceBERT paper](https://doi.org/10.1101/2023.01.31.526427) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
