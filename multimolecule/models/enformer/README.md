---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/encode
  - multimolecule/fantom5
  - multimolecule/gtex
library_name: multimolecule
---

# Enformer

Transformer-based deep neural network for predicting genomic coverage tracks from long DNA sequences with long-range context.

## Disclaimer

This is an UNOFFICIAL implementation of [Effective gene expression prediction from sequence by integrating long-range interactions](https://doi.org/10.1038/s41592-021-01252-x) by Žiga Avsec, Vikram Agarwal, Daniel Visentin et al.

The OFFICIAL repository of Enformer is at [google-deepmind/deepmind-research/enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Enformer did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Enformer is the successor of Basenji. It replaces Basenji's dilated convolution tower with a convolution stem followed by a Transformer trunk, which lets it model long-range genomic interactions. It consumes a long DNA window (~393 kb), passes it through a convolution + attention-pooling stem that downsamples the sequence by `2 ** 7 = 128x`, processes the binned representation with 11 Transformer blocks using Transformer-XL style relative positional encoding, center-crops to 896 output bins, and applies a pointwise head plus a per-species linear track projection with a softplus activation. The prediction is **binned**: the output has shape `(batch_size, target_length, num_tracks)` where each bin summarizes 128 bp of sequence and `num_tracks` is the number of genomic coverage experiments for the selected species.

### Model Specification

| Input Length | Bin Size | Output Bins | Hidden Size | Layers | Heads | Num Labels | Num Parameters (M) |
| ------------ | -------- | ----------- | ----------- | ------ | ----- | ---------- | ------------------ |
| 393216       | 128      | 896         | 1536        | 11     | 8     | 5313       | 246.2              |

The default table reports the human output head. The mouse head predicts 1643 tracks.

### Links

- **Code**: [multimolecule.enformer](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/enformer)
- **Weights**: [multimolecule/enformer](https://huggingface.co/multimolecule/enformer)
- **Data**: ENCODE, FANTOM5, GTEx CAGE, ChIP-seq, DNase-seq, and related genomic coverage tracks
- **Paper**: [Effective gene expression prediction from sequence by integrating long-range interactions](https://doi.org/10.1038/s41592-021-01252-x)
- **Developed by**: Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam, Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, David R. Kelley
- **Model type**: Convolutional stem followed by Transformer trunk with long-range attention for binned multi-track genomic coverage prediction
- **Original Repository**: [google-deepmind/deepmind-research/enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Genomic Coverage Prediction

You can use this model to predict binned genomic coverage tracks from a DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, EnformerConfig, EnformerForTokenPrediction

>>> config = EnformerConfig(
...     sequence_length=256, hidden_size=12, num_hidden_layers=1, num_attention_heads=2,
...     attention_head_size=4, num_downsamples=3, dim_divisible_by=2, target_length=16,
...     num_labels=4,
... )
>>> model = EnformerForTokenPrediction(config)
>>> output = model(torch.randint(config.vocab_size, (1, 256)))
>>> output.logits.shape
torch.Size([1, 16, 4])
```

The binned positional axis is treated as the "token" axis: each output position corresponds to one
genomic bin rather than a single nucleotide. The `species` configuration option selects the
`human` (5,313 tracks) or `mouse` (1,643 tracks) output head.

### Interface

- **Input length**: fixed 393,216 bp DNA window
- **Output binning**: 128 bp per output bin; 896 output bins per window (after center-cropping the binned representation)
- **Species head**: select `human` (5,313 tracks) or `mouse` (1,643 tracks) via the `species` config option
- **Output**: `(batch_size, target_length, num_tracks)`

## Training Details

Enformer was trained to predict genomic coverage tracks (DNase-seq, ATAC-seq, ChIP-seq and CAGE)
from the human and mouse reference genomes.

### Training Data

The model was trained on a large compendium of functional genomics experiments aligned to the
human (hg38) and mouse (mm10) reference genomes. The genome was divided into overlapping windows;
for each window the per-128-bp coverage of every experiment served as the regression target.

### Training Procedure

#### Pre-training

The model was trained to minimize a Poisson regression loss between predicted and observed
coverage, using a softplus output activation to keep the predicted coverage non-negative.

## Citation

```bibtex
@article{avsec2021effective,
  author    = {Avsec, {\v{Z}}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
  title     = {Effective gene expression prediction from sequence by integrating long-range interactions},
  journal   = {Nature Methods},
  year      = 2021,
  volume    = 18,
  number    = 10,
  pages     = {1196--1203},
  doi       = {10.1038/s41592-021-01252-x},
  publisher = {Nature Publishing Group}
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

Please contact the authors of the [Enformer paper](https://doi.org/10.1038/s41592-021-01252-x) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
