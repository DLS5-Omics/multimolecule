---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-activity
---

# scBasset

Sequence-based convolutional neural network for modeling single-cell ATAC-seq chromatin accessibility from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [scBasset: sequence-based modeling of single-cell ATAC-seq using convolutional neural networks](https://doi.org/10.1038/s41592-022-01562-8) by Han Yuan, et al.

The OFFICIAL repository of scBasset is at [calico/scBasset](https://github.com/calico/scBasset).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing scBasset did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

scBasset is a convolutional neural network (CNN) that predicts per-cell chromatin accessibility of a DNA peak sequence. The model consumes a fixed-length 1344 bp one-hot encoded DNA sequence and applies a pre-activation convolution stem, a reducing convolution tower, a pointwise convolution, and a dense bottleneck before a final cell-embedding layer that produces one accessibility logit per single cell.

scBasset uses a pre-activation block layout: each convolution block applies the activation (the sigmoid approximation of GELU, `sigmoid(1.702 * x) * x`) _before_ the convolution, then batch normalization and max pooling. The dense bottleneck flattens the convolution output in Keras channels-last (length-major) order.

> [!IMPORTANT]
> The final cell-embedding (dense) layer of scBasset is **dataset-specific**: it has one row per single cell in the training atlas. The Buenrostro2018 hematopoiesis tutorial dataset distributed by the scBasset authors has **2034 single cells** (so `num_labels = 2034`). A different scBasset dataset would have a different number of cells and a differently sized cell-embedding layer.

The cell-embedding layer is exposed through the shared [`SequencePredictionHead`][multimolecule.SequencePredictionHead]; the per-cell accessibility task is modeled as a binary problem (`problem_type="binary"`).

### Model Specification

| Num Conv Layers | Hidden Size | Num Cells | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| --------------- | ----------- | --------- | ------------------ | --------- | -------- | -------------- |
| 8               | 32          | 2034      | 4.59               | 0.95      | 0.47     | 1344           |

### Links

- **Code**: [multimolecule.scbasset](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/scbasset)
- **Data**: [scBasset Buenrostro2018 tutorial data](https://storage.googleapis.com/scbasset_tutorial_data/buen_ad_sc.h5ad)
- **Paper**: [scBasset: sequence-based modeling of single-cell ATAC-seq using convolutional neural networks](https://doi.org/10.1038/s41592-022-01562-8)
- **Developed by**: Han Yuan, David R. Kelley
- **Model type**: 1D CNN backbone with learned per-cell embedding head for single-cell ATAC-seq accessibility prediction
- **Original Repository**: [calico/scBasset](https://github.com/calico/scBasset)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Single-Cell Chromatin Accessibility Prediction

You can use this model directly to predict per-cell chromatin accessibility of a DNA peak sequence:

```python
>>> from multimolecule import DnaTokenizer, ScBassetForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/scbasset")
>>> model = ScBassetForSequencePrediction.from_pretrained("multimolecule/scbasset")
>>> input = tokenizer("ACGT" * 336, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 2034])
```

Each of the 2034 logits is a per-cell accessibility score for the Buenrostro2018 hematopoiesis atlas.

### Interface

- **Input length**: fixed 1344 bp DNA peak window
- **Output**: per-cell accessibility logits (2034 cells in the Buenrostro2018 hematopoiesis atlas; cell count is dataset-specific)

## Training Details

scBasset was trained to predict the per-cell chromatin accessibility of DNA peak sequences across a single-cell ATAC-seq atlas.

### Training Data

The scBasset model uses the **Buenrostro2018 hematopoiesis** tutorial model trained on the Buenrostro et al. 2018 single-cell ATAC-seq hematopoiesis dataset (2034 single cells). Each 1344 bp peak is associated with a per-cell binary accessibility vector.

### Training Procedure

#### Pre-training

The model was trained to minimize a per-cell binary cross-entropy loss, comparing its predicted per-cell accessibility probabilities (sigmoid of the cell-embedding logits) against the observed single-cell ATAC-seq accessibility labels.

- Optimizer: Adam
- Loss: Per-cell binary cross-entropy
- Regularization: Batch normalization and dropout

## Citation

```bibtex
@article{yuan2022scbasset,
  author    = {Yuan, Han and Kelley, David R.},
  title     = {scBasset: sequence-based modeling of single-cell ATAC-seq using convolutional neural networks},
  journal   = {Nature Methods},
  volume    = 19,
  number    = 9,
  pages     = {1088--1096},
  year      = 2022,
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41592-022-01562-8}
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

Please contact the authors of the [scBasset paper](https://doi.org/10.1038/s41592-022-01562-8) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
