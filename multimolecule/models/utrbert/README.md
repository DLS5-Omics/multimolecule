---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
---

# 3UTRBERT

Pre-trained model on 3’ untranslated region (3’UTR) using a masked language modeling (MLM) objective.

## Disclaimer

This is an UNOFFICIAL implementation of the [Deciphering 3’ UTR mediated gene regulation using interpretable deep representation learning](https://doi.org/10.1101/2023.09.08.556883) by Yuning Yang, Gen Li, et al.

The OFFICIAL repository of 3UTRBERT is at [yangyn533/3UTRBERT](https://github.com/yangyn533/3UTRBERT).

**The team releasing 3UTRBERT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

3UTRBERT is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model pre-trained on a large corpus of 3’ untranslated regions (3’UTRs) in a self-supervised fashion. This means that the model was trained on the raw nucleotides of RNA sequences only, with an automatic process to generate inputs and labels from those texts. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variations

- **[`multimolecule/utrbert-3mer`](https://huggingface.co/multimolecule/utrbert-3mer)**: The 3UTRBERT model pre-trained on 3-mer data.
- **[`multimolecule/utrbert-4mer`](https://huggingface.co/multimolecule/utrbert-4mer)**: The 3UTRBERT model pre-trained on 4-mer data.
- **[`multimolecule/utrbert-5mer`](https://huggingface.co/multimolecule/utrbert-5mer)**: The 3UTRBERT model pre-trained on 5-mer data.
- **[`multimolecule/utrbert-6mer`](https://huggingface.co/multimolecule/utrbert-6mer)**: The 3UTRBERT model pre-trained on 6-mer data.

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
    <td>UTRBERT-3mer</td>
    <td rowspan="4">12</td>
    <td rowspan="4">768</td>
    <td rowspan="4">12</td>
    <td rowspan="4">3072</td>
    <td>86.14</td>
    <td rowspan="4">22.36</td>
    <td rowspan="4">11.17</td>
    <td rowspan="4">512</td>
  </tr>
  <tr>
    <td>UTRBERT-4mer</td>
    <td>86.53</td>
  </tr>
  <tr>
    <td>UTRBERT-5mer</td>
    <td>88.45</td>
  </tr>
  <tr>
    <td>UTRBERT-6mer</td>
    <td>98.05</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.utrbert](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/utrbert)
- **Data**: [GENCODE](https://gencodegenes.org)
- **Paper**: [Deciphering 3’ UTR mediated gene regulation using interpretable deep representation learning](https://doi.org/10.1101/2023.09.08.556883)
- **Developed by**: Yuning Yang, Gen Li, Kuan Pang, Wuxinhao Cao, Xiangtao Li, Zhaolei Zhang
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased) - [FlashAttention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)
- **Original Repository**: [https://github.com/yangyn533/3UTRBERT](https://github.com/yangyn533/3UTRBERT)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

**Note**: Default transformers pipeline does not support K-mer tokenization.

You can use this model directly with a pipeline for masked language modeling:

```python
>>> import multimolecule  # you must import multimolecule to register models
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='multimolecule/utrbert-3mer')
>>> unmasker("uagc<mask><mask><mask>ucagacugauguuga")[1]

[{'score': 0.510771632194519,
  'token': 49,
  'token_str': 'CUU',
  'sequence': '<cls> UAG AGC <mask> CUU <mask> UCA CAG AGA GAC ACU CUG UGA GAU AUG UGU GUU UUG UGA <eos>'},
 {'score': 0.3299057185649872,
  'token': 39,
  'token_str': 'CCU',
  'sequence': '<cls> UAG AGC <mask> CCU <mask> UCA CAG AGA GAC ACU CUG UGA GAU AUG UGU GUU UUG UGA <eos>'},
 {'score': 0.09743840992450714,
  'token': 34,
  'token_str': 'CAU',
  'sequence': '<cls> UAG AGC <mask> CAU <mask> UCA CAG AGA GAC ACU CUG UGA GAU AUG UGU GUU UUG UGA <eos>'},
 {'score': 0.010745460167527199,
  'token': 64,
  'token_str': 'GCU',
  'sequence': '<cls> UAG AGC <mask> GCU <mask> UCA CAG AGA GAC ACU CUG UGA GAU AUG UGU GUU UUG UGA <eos>'},
 {'score': 0.010299043729901314,
  'token': 24,
  'token_str': 'AUU',
  'sequence': '<cls> UAG AGC <mask> AUU <mask> UCA CAG AGA GAC ACU CUG UGA GAU AUG UGU GUU UUG UGA <eos>'}]
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import RnaTokenizer, UtrBertModel


tokenizer = RnaTokenizer.from_pretrained('multimolecule/utrbert-3mer')
model = UtrBertModel.from_pretrained('multimolecule/utrbert-3mer')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')

output = model(**input)
```

#### Sequence Classification / Regression

**Note**: This model is not fine-tuned for any specific task. You will need to fine-tune the model on a downstream task to use it for sequence classification or regression.

Here is how to use this model as backbone to fine-tune for a sequence-level task in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, UtrBertForSequencePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/utrbert-3mer')
model = UtrBertForSequencePrediction.from_pretrained('multimolecule/utrbert-3mer')

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
from multimolecule import RnaTokenizer, UtrBertForNucleotidePrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/utrbert-3mer')
model = UtrBertForNucleotidePrediction.from_pretrained('multimolecule/utrbert-3mer')

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
from multimolecule import RnaTokenizer, UtrBertForContactPrediction


tokenizer = RnaTokenizer.from_pretrained('multimolecule/utrbert')
model = UtrBertForContactPrediction.from_pretrained('multimolecule/utrbert')

text = "UAGCUUAUCAGACUGAUGUUGA"
input = tokenizer(text, return_tensors='pt')
label = torch.randint(2, (len(text), len(text)))

output = model(**input, labels=label)
```

## Training Details

3UTRBERT used Masked Language Modeling (MLM) as the pre-training objective: taking a sequence, the model randomly masks 15% of the tokens in the input then runs the entire masked sentence through the model and has to predict the masked tokens. This is comparable to the Cloze task in language modeling.

### Training Data

The 3UTRBERT model was pre-trained on human mRNA transcript sequences from [GENCODE](https://gencodegenes.org). GENCODE aims to identify all gene features in the human genome using a combination of computational analysis, manual annotation, and experimental validation. The GENCODE release 40 used by this work contains 61,544 genes, and 246,624 transcripts.

3UTRBERT collected the human mRNA transcript sequences from GENCODE, including 108,573 unique mRNA transcripts. Only the longest transcript of each gene was used in the pre-training process. 3UTRBERT only used the 3’ untranslated regions (3’UTRs) of the mRNA transcripts for pre-training to avoid codon constrains in the CDS region, and to reduce increased complexity of the entire mRNA transcripts. The average length of the 3’UTRs was 1,227 nucleotides, while the median length was 631 nucleotides. Each 3’UTR sequence was cut to non-overlapping patches of 510 nucleotides. The remaining sequences were padded to the same length.

Note [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Preprocessing

3UTRBERT used masked language modeling (MLM) as the pre-training objective. The masking procedure is similar to the one used in BERT:

- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

Since 3UTRBERT used k-mer tokenizer, it masks the entire k-mer instead of individual nucleotides to avoid information leakage.

For example, if the k-mer is 3, the sequence `"UAGCGUAU"` will be tokenized as `["UAG", "AGC", "GCG", "CGU", "GUA", "UAU"]`. If the nucleotide `"C"` is masked, the adjacent tokens will also be masked, resulting `["UAG", "<mask>", "<mask>", "<mask>", "GUA", "UAU"]`.

#### PreTraining

The model was trained on 4 NVIDIA Quadro RTX 6000 GPUs with 24GiB memories.

- Batch size: 128
- Learning rate: 3e-4
- Weight decay: 0.01
- Optimizer: AdamW(β1=0.9, β2=0.98, e=1e-6)
- Steps: 200,000
- Learning rate scheduler: Linear
- Learning rate warm-up: 10,000 steps

## Citation

**BibTeX**:

```bibtex
@article {yang2023deciphering,
	author = {Yang, Yuning and Li, Gen and Pang, Kuan and Cao, Wuxinhao and Li, Xiangtao and Zhang, Zhaolei},
	title = {Deciphering 3{\textquoteright} UTR mediated gene regulation using interpretable deep representation learning},
	elocation-id = {2023.09.08.556883},
	year = {2023},
	doi = {10.1101/2023.09.08.556883},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The 3{\textquoteright}untranslated regions (3{\textquoteright}UTRs) of messenger RNAs contain many important cis-regulatory elements that are under functional and evolutionary constraints. We hypothesize that these constraints are similar to grammars and syntaxes in human languages and can be modeled by advanced natural language models such as Transformers, which has been very effective in modeling protein sequence and structures. Here we describe 3UTRBERT, which implements an attention-based language model, i.e., Bidirectional Encoder Representations from Transformers (BERT). 3UTRBERT was pre-trained on aggregated 3{\textquoteright}UTR sequences of human mRNAs in a task-agnostic manner; the pre-trained model was then fine-tuned for specific downstream tasks such as predicting RBP binding sites, m6A RNA modification sites, and predicting RNA sub-cellular localizations. Benchmark results showed that 3UTRBERT generally outperformed other contemporary methods in each of these tasks. We also showed that the self-attention mechanism within 3UTRBERT allows direct visualization of the semantic relationship between sequence elements.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.08.556883},
	eprint = {https://www.biorxiv.org/content/early/2023/09/12/2023.09.08.556883.full.pdf},
	journal = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [3UTRBERT paper](https://doi.org/10.1101/2023.09.08.556883) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
