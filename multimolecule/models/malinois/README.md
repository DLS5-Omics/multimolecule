---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/malinois
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-activity
---

# Malinois

Convolutional neural network for predicting cell-type-targeting cis-regulatory element (CRE) activity from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Machine-guided design of cell-type-targeting cis-regulatory elements](https://doi.org/10.1038/s41586-024-08070-z) by Sager J. Gosai, Rodrigo I. Castro, et al.

The OFFICIAL repository of Malinois is at [sjgosai/boda2](https://github.com/sjgosai/boda2).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Malinois did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Malinois is a deep convolutional neural network (a tuned Basset-style "branched" architecture) trained to quantitatively predict cell-type-informed CRE activity from ~200 bp DNA sequences measured by a massively parallel reporter assay (MPRA). The model emits three regression outputs, one per human cell line: K562, HepG2 and SK-N-SH (in that order).

The architecture consists of three convolutional blocks, one shared fully-connected block, and a branched grouped-linear tower with an independent parameter set per cell line. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 8          | 420         | 4.11               | 332.95    | 165.70   | 600            |

### Links

- **Code**: [multimolecule.malinois](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/malinois)
- **Data**: MPRA libraries across K562, HepG2, and SK-N-SH human cell lines
- **Paper**: [Machine-guided design of cell-type-targeting cis-regulatory elements](https://doi.org/10.1038/s41586-024-08070-z)
- **Developed by**: Sager J. Gosai, Rodrigo I. Castro, Natalia Fuentes, John C. Butts, Kousuke Mouri, Michael Alasoadura, Susan Kales, Thanh Thanh L. Nguyen, Ramil R. Noche, Arya S. Rao, Mary T. Joy, Pardis C. Sabeti, Steven K. Reilly, Ryan Tewhey
- **Model type**: 1D CNN with cell-type-specific grouped-linear output head for MPRA cis-regulatory element activity
- **Original Repository**: [sjgosai/boda2](https://github.com/sjgosai/boda2)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### CRE Activity Prediction

You can use this model directly to predict the cell-type-informed CRE activity (K562, HepG2, SK-N-SH) of a sequence. Malinois pads each ~200 bp candidate to 600 bp with fixed MPRA plasmid flanks before inference; the example below uses a pre-padded 600 bp sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, MalinoisForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/malinois")
>>> model = MalinoisForSequencePrediction.from_pretrained("multimolecule/malinois")
>>> sequence = "ACGT" * 150
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 3])
```

### Interface

- **Input length**: fixed 600 bp window
- **Padding**: each ~200 bp candidate CRE is centered and padded with fixed MPRA plasmid flanks (`MPRA_UPSTREAM` / `MPRA_DOWNSTREAM`); flank padding is part of the data pipeline, not the model
- **Output**: 3 cell-line CRE activity values (K562, HepG2, SK-N-SH)

## Training Details

Malinois was trained to predict quantitative, cell-type-informed CRE activity from DNA sequence.

### Training Data

Malinois was trained on a lentiMPRA dataset measuring the regulatory activity of ~200 bp sequences across three human cell lines (K562, HepG2 and SK-N-SH). Each training example is a sequence with three continuous activity values (log2 fold-change over input), one per cell line. Genomic sequences were split by chromosome into training, validation, and test sets to avoid sequence leakage.

### Training Procedure

#### Pre-training

The model was trained to minimize an L1 + KL-divergence mixed loss between predicted and measured cell-type CRE activities, with the architecture and training hyperparameters selected by Bayesian optimization.

- Optimizer: Adam
- Loss: L1 + KL-divergence mixed loss
- Early stopping on validation loss

## Citation

```bibtex
@article{gosai2024malinois,
  author    = {Gosai, Sager J. and Castro, Rodrigo I. and Fuentes, Natalia and Butts, John C. and Mouri, Kousuke and Alasoadura, Michael and Kales, Susan and Nguyen, Thanh Thanh L. and Noche, Ramil R. and Rao, Arya S. and Joy, Mary T. and Sabeti, Pardis C. and Reilly, Steven K. and Tewhey, Ryan},
  journal   = {Nature},
  month     = oct,
  number    = 8036,
  pages     = {1211--1220},
  publisher = {Springer Science and Business Media LLC},
  title     = {Machine-guided design of cell-type-targeting cis-regulatory elements},
  volume    = 634,
  year      = 2024,
  doi       = {10.1038/s41586-024-08070-z}
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

Please contact the authors of the [Malinois paper](https://doi.org/10.1038/s41586-024-08070-z) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
