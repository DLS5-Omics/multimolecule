---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: sequence-binding-prediction
---

# RBP-eCLIP

Per-RBP convolutional neural network with spline-transformed positional features for predicting RNA-binding-protein (RBP) binding from sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Modeling positional effects of regulatory sequences with spline transformations increases prediction accuracy of deep neural networks](https://doi.org/10.1093/bioinformatics/btx727) by **Žiga Avsec et al.**.

The OFFICIAL repository of RBP-eCLIP is at [gagneurlab/Manuscript_Avsec_Bioinformatics_2017](https://github.com/gagneurlab/Manuscript_Avsec_Bioinformatics_2017); the per-RBP trained checkpoints are distributed via [Kipoi `rbp_eclip`](https://kipoi.org/models/rbp_eclip/).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RBP-eCLIP did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RBP-eCLIP is a small 1D convolutional neural network that combines an RNA-sequence module with a position module. The sequence module consumes a fixed-length 101 nt one-hot encoded RNA peak window and applies two convolutions (a motif-scoring `ConvDNA` and a 1x1 mixing convolution) followed by global max-pooling. The position module consumes eight scalar genomic-landmark distance features (TSS, poly-A, exon/intron and intron/exon boundaries, start and stop codons, gene start and gene end) that are pre-encoded with a 10-component B-spline basis and projected through per-feature GAM 1x1 convolutions. The pooled sequence features and the position-module scalars are concatenated and projected through a single hidden dense layer before the binding-score head. The spline-based positional encoding lets the network model smooth, position-dependent regulatory effects without spending parameters on raw coordinates. Please refer to the [Training Details](#training-details) section for more information on the training process.

The Avsec et al. eCLIP RBP model family ships one trained checkpoint per RBP. The MultiMolecule release exposes a single model class (`RbpEclipModel`) with one Hub repository per RBP; the architecture is identical across RBPs.

### Variants

The MultiMolecule release ships a small set of representative per-RBP checkpoints from the Kipoi `rbp_eclip` model group. The full upstream family contains 112 trained RBPs; additional checkpoints can be converted via [`convert_checkpoint.py`](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rbpeclip/convert_checkpoint.py).

| RBP    | Hub repository                                                                  |
| ------ | ------------------------------------------------------------------------------- |
| HNRNPK | [`rbpeclip-hnrnpk`](https://huggingface.co/multimolecule/rbpeclip-hnrnpk)       |
| PUM2   | [`rbpeclip-pum2`](https://huggingface.co/multimolecule/rbpeclip-pum2)           |
| U2AF2  | [`rbpeclip-u2af2`](https://huggingface.co/multimolecule/rbpeclip-u2af2)         |

### Model Specification

| Sequence Conv | Position Bases | Hidden Size | Num Parameters | FLOPs (M) | MACs (M) | Max Num Tokens |
| ------------- | -------------- | ----------- | -------------- | --------- | -------- | -------------- |
| 2             | 10             | 100         | 38,021         | 0.30      | 0.14     | 101            |

### Links

- **Code**: [multimolecule.rbpeclip](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/rbpeclip)
- **Weights**: [multimolecule/rbpeclip](https://huggingface.co/multimolecule/rbpeclip)
- **Data**: [ENCODE eCLIP-seq](https://www.encodeproject.org/eclip/) released by Van Nostrand et al. (2016); preprocessed via the [Kipoi `rbp_eclip`](https://kipoi.org/models/rbp_eclip/) dataloader.
- **Paper**: [Modeling positional effects of regulatory sequences with spline transformations increases prediction accuracy of deep neural networks](https://doi.org/10.1093/bioinformatics/btx727)
- **Developed by**: Žiga Avsec, Mohammadamin Barekatain, Jun Cheng, Julien Gagneur
- **Model type**: 1D CNN over a 101 nt RNA peak window with spline-transformed positional features for per-RBP binding-score prediction
- **Original Repository**: [gagneurlab/Manuscript_Avsec_Bioinformatics_2017](https://github.com/gagneurlab/Manuscript_Avsec_Bioinformatics_2017)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RBP Binding-Score Prediction

You can use this model directly to predict the binding score of a 101 nt RNA peak window:

```python
>>> import torch
>>> from multimolecule import RbpEclipForSequencePrediction, RnaTokenizer

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rbpeclip")
>>> model = RbpEclipForSequencePrediction.from_pretrained("multimolecule/rbpeclip")
>>> sequence = "ACGU" * 25 + "A"
>>> input = tokenizer(sequence, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 1])
```

The optional spline-encoded position features can be passed explicitly. When omitted, the position module sees an all-zero basis (matching the Kipoi default dataloader behaviour when no transcript context is available):

```python
>>> position_features = torch.zeros(1, 8, 10)
>>> output = model(**input, position_features=position_features)
```

### Interface

- **Input length**: fixed 101 nt RNA peak window
- **Alphabet**: streamline RNA `ACGUN`
- **Special tokens**: do not add (`add_special_tokens=False` is implied by the fixed peak window)
- **Auxiliary inputs**: `position_features` of shape `(batch_size, 8, 10)`; the eight position features are the B-spline-encoded scalar distances to TSS, poly-A, exon/intron boundary, intron/exon boundary, start codon, stop codon, gene start, and gene end. When omitted, the model uses an all-zero basis.
- **Output**: single binding-score logit per sequence (`postprocess` applies a sigmoid to return a calibrated binding probability).

## Training Details

RBP-eCLIP was trained on per-RBP eCLIP-seq peak windows from the ENCODE eCLIP compendium released by Van Nostrand et al. (2016). The published architecture is shared across all 112 RBPs in the Kipoi `rbp_eclip` model group; one checkpoint is trained per RBP.

### Training Data

Each model was trained on positive and negative 101 nt RNA peak windows derived from the per-RBP eCLIP peaks of [Van Nostrand et al. 2016 (ENCODE)](https://www.encodeproject.org/eclip/). The Kipoi `rbp_eclip` dataloader also produces eight scalar genomic-landmark distance features per window (TSS, poly-A, exon-intron, intron-exon, start codon, stop codon, gene start, gene end) and encodes each one with a 10-component B-spline basis.

### Training Procedure

#### Pre-training

The model was trained with single-task binary cross-entropy on the per-RBP binding label (peak vs background).

- Optimizer: Adam
- Loss: Binary cross-entropy
- Regularization: Dropout on the pooled sequence features and on the hidden dense layer; GAM (second-derivative) smoothness penalty on the position-module spline coefficients.

## Citation

```bibtex
@article{avsec2018modeling,
  author    = {Avsec, {\v{Z}}iga and Barekatain, Mohammadamin and Cheng, Jun and Gagneur, Julien},
  title     = {Modeling positional effects of regulatory sequences with spline transformations increases prediction accuracy of deep neural networks},
  journal   = {Bioinformatics},
  volume    = {34},
  number    = {8},
  pages     = {1261--1269},
  year      = {2018},
  publisher = {Oxford University Press},
  doi       = {10.1093/bioinformatics/btx727}
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

Please contact the authors of the [RBP-eCLIP paper](https://doi.org/10.1093/bioinformatics/btx727) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
