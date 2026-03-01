---
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/gencode-human
library_name: multimolecule
pipeline_tag: text-generation
---

# HyenaDNA

Pre-trained model on human reference genome using a causal language modeling (CLM) objective with the Hyena operator.

## Disclaimer

This is an UNOFFICIAL implementation of the [HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution](https://doi.org/10.5555/3666122.3667994) by Eric Nguyen, Michael Poli, Marjan Faizi, et al.

The OFFICIAL repository of HyenaDNA is at [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing HyenaDNA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

HyenaDNA is a decoder-only model pre-trained on the human reference genome with single nucleotide tokenization in a self-supervised fashion. This means that the model was trained on the raw nucleotides of DNA sequences only, with an automatic process to generate inputs and labels from those sequences. Please refer to the [Training Details](#training-details) section for more information on the training process.

HyenaDNA replaces the attention mechanism with the Hyena operator — a subquadratic sequence mixer based on implicit long convolutions. This enables O(L log L) complexity for sequence modeling, allowing the model to handle context lengths up to 1 million base pairs at single nucleotide resolution.

The Hyena operator uses:

- **Implicit filters**: MLP-parameterized convolution kernels with learned positional embeddings
- **Element-wise gating**: Multiplicative interactions between projected input channels
- **FFT convolution**: Efficient computation via the Fast Fourier Transform

### Variants

- **[multimolecule/hyenadna-large-1m](https://huggingface.co/multimolecule/hyenadna-large-1m)**: The HyenaDNA model with 8 layers, 256 hidden size, and 1,000,000 context length.
- **[multimolecule/hyenadna-medium-450k](https://huggingface.co/multimolecule/hyenadna-medium-450k)**: The HyenaDNA model with 8 layers, 256 hidden size, and 450,000 context length.
- **[multimolecule/hyenadna-medium-160k](https://huggingface.co/multimolecule/hyenadna-medium-160k)**: The HyenaDNA model with 8 layers, 256 hidden size, and 160,000 context length.
- **[multimolecule/hyenadna-small-32k](https://huggingface.co/multimolecule/hyenadna-small-32k)**: The HyenaDNA model with 4 layers, 256 hidden size, and 32,768 context length.
- **[multimolecule/hyenadna-tiny-16k-d128](https://huggingface.co/multimolecule/hyenadna-tiny-16k-d128)**: The HyenaDNA model with 2 layers, 128 hidden size, and 16,384 context length.
- **[multimolecule/hyenadna-tiny-1k-d256](https://huggingface.co/multimolecule/hyenadna-tiny-1k-d256)**: The HyenaDNA model with 2 layers, 256 hidden size, and 1,024 context length.
- **[multimolecule/hyenadna-tiny-1k](https://huggingface.co/multimolecule/hyenadna-tiny-1k)**: The HyenaDNA model with 2 layers, 128 hidden size, and 1,024 context length.

### Model Specification

<table>
<thead>
  <tr>
    <th>Variants</th>
    <th>Num Layers</th>
    <th>Hidden Size</th>
    <th>Intermediate Size</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
    <th>Max Num Tokens</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>HyenaDNA-large-1m</td>
    <td rowspan="3">8</td>
    <td rowspan="5">256</td>
    <td rowspan="5">1024</td>
    <td rowspan="3">6.62</td>
    <td rowspan="3">6.69</td>
    <td rowspan="3">3.35</td>
    <td>1,000,002</td>
  </tr>
  <tr>
    <td>HyenaDNA-medium-450k</td>
    <td>450,002</td>
  </tr>
  <tr>
    <td>HyenaDNA-medium-160k</td>
    <td>160,002</td>
  </tr>
  <tr>
    <td>HyenaDNA-small-32k</td>
    <td>4</td>
    <td>3.34</td>
    <td>3.35</td>
    <td>1.67</td>
    <td>32,770</td>
  </tr>
  <tr>
    <td>HyenaDNA-tiny-1k-d256</td>
    <td rowspan="3">2</td>
    <td>1.71</td>
    <td>1.67</td>
    <td>0.84</td>
    <td>1,026</td>
  </tr>
  <tr>
    <td>HyenaDNA-tiny-16k-d128</td>
    <td rowspan="2">128</td>
    <td rowspan="2">512</td>
    <td rowspan="2">0.45</td>
    <td rowspan="2">0.44</td>
    <td rowspan="2">0.22</td>
    <td>16,386</td>
  </tr>
  <tr>
    <td>HyenaDNA-tiny-1k</td>
    <td>1,026</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.hyenadna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/hyenadna)
- **Data**: [multimolecule/gencode-human](https://huggingface.co/datasets/multimolecule/gencode-human)
- **Paper**: [HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution](https://doi.org/10.5555/3666122.3667994)
- **Developed by**: Eric Nguyen, Michael Poli, Marjan Faizi, Armin W. Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Christopher Ré, Stephen A. Baccus
- **Model type**: Decoder-only with [Hyena](https://arxiv.org/abs/2302.10866) operator
- **Original Repository**: [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)

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

generator = pipeline("text-generation", model="multimolecule/hyenadna-medium-160k")
output = generator("ATCGATCGATCG", max_new_tokens=50)
```

### Downstream Use

#### Extract Features

Here is how to use this model to get the features of a given sequence in PyTorch:

```python
from multimolecule import DnaTokenizer, HyenaDnaModel


tokenizer = DnaTokenizer.from_pretrained("multimolecule/hyenadna-medium-160k")
model = HyenaDnaModel.from_pretrained("multimolecule/hyenadna-medium-160k")

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
from multimolecule import DnaTokenizer, HyenaDnaForSequencePrediction


tokenizer = DnaTokenizer.from_pretrained("multimolecule/hyenadna-medium-160k")
model = HyenaDnaForSequencePrediction.from_pretrained("multimolecule/hyenadna-medium-160k")

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
from multimolecule import DnaTokenizer, HyenaDnaForTokenPrediction


tokenizer = DnaTokenizer.from_pretrained("multimolecule/hyenadna-medium-160k")
model = HyenaDnaForTokenPrediction.from_pretrained("multimolecule/hyenadna-medium-160k")

text = "ATCGATCGATCGATCG"
input = tokenizer(text, return_tensors="pt")
label = torch.randint(2, (len(text), ))

output = model(**input, labels=label)
```

## Training Details

HyenaDNA used Causal Language Modeling (CLM) as the pre-training objective: given a DNA sequence, the model is trained to predict the next nucleotide token autoregressively.

### Training Data

The HyenaDNA model was pre-trained on the human reference genome (GRCh38). The training data consists of single nucleotide-level DNA sequences from all human chromosomes. Sequences are tokenized at the individual character level (A, C, G, T, N) without k-mer encoding.

The dataset is split into training and test sets by chromosome, with held-out chromosomes used for evaluation.

### Training Procedure

#### Preprocessing

HyenaDNA used causal language modeling (CLM) as the pre-training objective: given a DNA sequence of length L, the model predicts the next nucleotide at each position, i.e., predicting token x\_{t+1} given x_1, ..., x_t.

Single nucleotide tokenization is used with a vocabulary of 12 tokens: A, C, G, T, N, and special tokens (CLS, SEP, BOS, MASK, PAD, RESERVED, UNK).

#### Sequence Length Warm-up

A key training strategy in HyenaDNA is progressive sequence length warm-up. Training begins with short sequences and gradually increases the context length:

1. Training starts with sequences of length L=64.
2. The sequence length is doubled at each warm-up stage (64 → 128 → 256 → ... → target length).
3. This strategy enables stable training at very long context lengths that would be difficult to train from scratch.

#### Pre-training

The model was trained on up to 8 NVIDIA A100 (80GB) GPUs.

- Batch size: 64 -- 256
- Steps: 10,000 -- 20,000
- Optimizer: AdamW
- Learning rate: 1.5e-4 -- 6e-4
- Learning rate scheduler: Cosine
- Weight decay: 0.1

## Citation

```bibtex
@inproceedings{NEURIPS2023_86ab6927,
  author = {Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Armin and Wornow, Michael and Birch-Sykes, Callum and Massaroli, Stefano and Patel, Aman and Rabideau, Clayton and Bengio, Yoshua and Ermon, Stefano and R\'{e}, Christopher and Baccus, Stephen},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
  pages = {43177--43201},
  publisher = {Curran Associates, Inc.},
  title = {HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/86ab6927ee4ae9bde4247793c46797c7-Paper-Conference.pdf},
  volume = {36},
  year = {2023}
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

Please contact the authors of the [HyenaDNA paper](https://doi.org/10.5555/3666122.3667994) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
