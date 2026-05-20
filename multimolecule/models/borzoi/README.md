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
pipeline_tag: other
pipeline: regulatory-track
---

# Borzoi

Sequence-to-coverage neural network for predicting RNA-seq and chromatin tracks across 524 kb DNA windows at 32 bp resolution.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation](https://doi.org/10.1038/s41588-024-02053-6) by Johannes Linder, Divyanshi Srivastava, Han Yuan, et al.

The OFFICIAL repository of Borzoi is at [calico/borzoi](https://github.com/calico/borzoi).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Borzoi did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Borzoi is the successor of [Enformer](https://huggingface.co/multimolecule/enformer). It extends the Enformer recipe (convolution stem + Transformer trunk + binned multi-track output) to a 524,288 bp input window and 32 bp output bins, and adds a U-Net style upsampling tail so the binned positional axis matches a higher-resolution coverage prediction. A long DNA window of 524 kb is downsampled by a convolution stem and a width-growing residual convolution tower, projected to 1,536 channels by a U-Net bottleneck, processed by 8 Transformer blocks with Transformer-XL style relative positional encoding, then upsampled by two skip-connected U-Net stages with depthwise-separable convolutions, center-cropped to 6,144 bins, and projected to per-species coverage tracks with a softplus activation. The output is **binned**: it has shape `(batch_size, target_length, num_tracks)` where each bin summarizes 32 bp of sequence and `num_tracks` is the number of genomic coverage experiments for the selected species. Borzoi was trained jointly on RNA-seq, CAGE, ATAC-seq, DNase-seq, and ChIP-seq tracks. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

Borzoi releases separate human and mouse checkpoints for the corresponding species track sets.

- **[multimolecule/borzoi-human](https://huggingface.co/multimolecule/borzoi-human)**: human checkpoint with 7,611 human genomic coverage tracks.
- **[multimolecule/borzoi-mouse](https://huggingface.co/multimolecule/borzoi-mouse)**: mouse checkpoint with 2,608 mouse genomic coverage tracks.

### Model Specification

| Input Length | Bin Size | Output Bins | Hidden Size | Layers | Heads | Num Labels | Num Parameters (M) | FLOPs (P) | MACs (P) |
| ------------ | -------- | ----------- | ----------- | ------ | ----- | ---------- | ------------------ | --------- | -------- |
| 524288       | 32       | 6144        | 1536        | 8      | 8     | 7611       | 185.90             | 13.57     | 6.76     |

The table reports the human checkpoint. The mouse checkpoint predicts 2,608 tracks.
FLOPs and MACs are measured on the canonical 524,288 bp Borzoi input window.

### Links

- **Code**: [multimolecule.borzoi](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/borzoi)
- **Data**: ENCODE, GTEx, FANTOM5 RNA-seq / CAGE / ATAC-seq / DNase-seq / ChIP-seq tracks aligned to human and mouse genomes
- **Paper**: [Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation](https://doi.org/10.1038/s41588-024-02053-6)
- **Developed by**: Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley
- **Model type**: Convolutional stem followed by Transformer trunk and U-Net upsampling tail for binned multi-track RNA-seq and chromatin coverage prediction
- **Original Repository**: [calico/borzoi](https://github.com/calico/borzoi)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Genomic Coverage Prediction

You can use this model to predict binned RNA-seq and chromatin coverage tracks from a DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, BorzoiConfig, BorzoiForTokenPrediction

>>> config = BorzoiConfig(
...     sequence_length=512, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
...     attention_head_size=4, attention_value_size=4, num_rel_pos_features=4,
...     stem_channels=8, conv_tower_channels=[12], head_hidden_size=8, target_length=16,
...     num_labels=4,
... )
>>> model = BorzoiForTokenPrediction(config)
>>> output = model(torch.randint(config.vocab_size, (1, 512)))
>>> output.logits.shape
torch.Size([1, 16, 4])
>>> coverage, channels = model.postprocess(output)
>>> coverage.shape
torch.Size([1, 16, 4])
```

The binned positional axis is treated as the "token" axis: each output position corresponds to one
genomic bin rather than a single nucleotide. The `species` configuration option selects the
`human` (7,611 tracks) or `mouse` (2,608 tracks) species track set for the converted checkpoint.

### Interface

- **Input length**: fixed 524,288 bp DNA window
- **Output binning**: 32 bp per output bin; 6,144 output bins per window (after center-cropping the U-Net upsampling tail)
- **Species track set**: select `human` (7,611 tracks) or `mouse` (2,608 tracks) via the `species` config option
- **Output**: raw pre-softplus `logits` of shape `(batch_size, target_length, num_tracks)`; use `postprocess` for non-negative coverage tracks

## Training Details

Borzoi was trained to predict bulk RNA-seq coverage together with chromatin tracks (DNase-seq, ATAC-seq, ChIP-seq) and CAGE from the human and mouse reference genomes.

### Training Data

The model was trained on a large compendium of functional genomics experiments aligned to the human (hg38) and mouse (mm10) reference genomes. The genome was divided into 524 kb windows; for each window the per-32-bp coverage of every experiment served as the regression target. The training set is dominated by RNA-seq coverage (the modality Borzoi extends over Enformer); the remaining tracks include the chromatin and CAGE modalities used by Enformer.

### Training Procedure

#### Pre-training

The model was trained to minimize a Poisson-multinomial regression loss between predicted and observed coverage, using a softplus output activation to keep the predicted coverage non-negative. Training used the Adam optimizer with a warmup schedule and global gradient-norm clipping; reverse-complement and small genomic-shift data augmentations were applied during training.

## Citation

```bibtex
@article{linder2025predicting,
  author    = {Linder, Johannes and Srivastava, Divyanshi and Yuan, Han and Agarwal, Vikram and Kelley, David R.},
  title     = {Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation},
  journal   = {Nature Genetics},
  year      = 2025,
  volume    = 57,
  number    = 4,
  pages     = {949--961},
  doi       = {10.1038/s41588-024-02053-6},
  publisher = {Nature Publishing Group}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If MultiMolecule supports your research, please cite the MultiMolecule project as follows:

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

Please contact the authors of the [Borzoi paper](https://doi.org/10.1038/s41588-024-02053-6) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
