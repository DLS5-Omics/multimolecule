---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-track
---

# Basenji

Deep convolutional neural network for predicting genomic coverage tracks across chromosomes.

## Disclaimer

This is an UNOFFICIAL implementation of [Sequential regulatory activity prediction across chromosomes with deep convolutional and recurrent neural networks](https://doi.org/10.1101/gr.227819.117) by David R. Kelley, Yakir A. Reshef, et al.

The OFFICIAL repository of Basenji is at [calico/basenji](https://github.com/calico/basenji).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Basenji did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Basenji is a deep convolutional neural network trained to predict genomic regulatory activity from long DNA sequences. It consumes a long DNA window (~131 kb), passes it through a convolution + pooling stem that downsamples the sequence, and then through a tower of dilated residual convolutional blocks that expand the receptive field. A pointwise output head predicts a vector of genomic coverage tracks for each output bin. Because the stem downsamples the input, the prediction is **binned**: the output has shape `(batch_size, num_bins, num_tracks)` where each bin summarizes 128 bp of sequence and `num_tracks` is the number of genomic coverage experiments.

### Model Specification

| Input Length | Bin Size | Output Bins | Hidden Size | Dilated Blocks | Num Labels | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ------------ | -------- | ----------- | ----------- | -------------- | ---------- | ------------------ | --------- | -------- | -------------- |
| 131,072      | 128      | 896         | 768         | 11             | 5,313      | 30.09              | 234.85    | 117.19   | 131,072        |

FLOPs and MACs are measured on the canonical 131,072 bp Basenji input window.

### Links

- **Code**: [multimolecule.basenji](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/basenji)
- **Data**: ENCODE, FANTOM5, GTEx, and related genomic coverage tracks aligned to human and mouse genomes
- **Paper**: [Sequential regulatory activity prediction across chromosomes with deep convolutional and recurrent neural networks](https://doi.org/10.1101/gr.227819.117)
- **Developed by**: David R. Kelley, Yakir A. Reshef, Maxwell Bileschi, David Belanger, Cory Y. McLean, Jasper Snoek
- **Model type**: 1D dilated residual CNN with pre-activation blocks for binned multi-track genomic coverage prediction
- **Original Repository**: [calico/basenji](https://github.com/calico/basenji)

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
>>> from multimolecule import DnaTokenizer, BasenjiConfig, BasenjiForTokenPrediction

>>> config = BasenjiConfig(
...     sequence_length=256, stem_channels=8, conv_tower_channels=[8],
...     stem_pool_size=2, head_hidden_size=8, crop_bins=2, num_labels=4,
...     blocks={"num_blocks": 1, "kernel_size": 3, "bottleneck_size": 4},
... )
>>> model = BasenjiForTokenPrediction(config)
>>> output = model(torch.randint(config.vocab_size, (1, 256)))
>>> output.logits.shape
torch.Size([1, 60, 4])
```

The binned positional axis is treated as the "token" axis: each output position corresponds to one
genomic bin rather than a single nucleotide.

### Interface

- **Input length**: fixed 131,072 bp DNA window
- **Output binning**: 128 bp per output bin; 896 output bins per window (after `Cropping1D(64)` on each side)
- **Output**: `(batch_size, num_bins, num_tracks)`; `num_tracks` is 5,313 human coverage experiments

## Training Details

Basenji was trained to predict genomic coverage tracks (DNase-seq, ATAC-seq, ChIP-seq and CAGE) from
the human and mouse reference genomes.

### Training Data

The model was trained on a large compendium of functional genomics experiments aligned to the human
(hg38) and mouse (mm10) reference genomes. The genome was divided into overlapping windows; for each
window the per-128-bp coverage of every experiment served as the regression target.

### Training Procedure

#### Pre-training

The model was trained to minimize a Poisson regression loss between predicted and observed coverage.

## Citation

```bibtex
@article{kelley2018sequential,
  author    = {Kelley, David R. and Reshef, Yakir A. and Bileschi, Maxwell and Belanger, David and McLean, Cory Y. and Snoek, Jasper},
  title     = {Sequential regulatory activity prediction across chromosomes with deep convolutional and recurrent neural networks},
  journal   = {Genome Research},
  year      = 2018,
  volume    = 28,
  number    = 5,
  pages     = {739--750},
  doi       = {10.1101/gr.227819.117},
  publisher = {Cold Spring Harbor Laboratory}
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

Please contact the authors of the [Basenji paper](https://doi.org/10.1101/gr.227819.117) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
