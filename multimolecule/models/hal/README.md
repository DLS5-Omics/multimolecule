---
language: dna
tags:
  - Biology
  - DNA
  - RNA
  - Splicing
license: agpl-3.0
library_name: multimolecule
---

# HAL

Hexamer Additive Linear model for predicting alternative splicing from sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Learning the Sequence Determinants of Alternative Splicing from Millions of Random Sequences](https://doi.org/10.1016/j.cell.2015.09.054) by Alexander B. Rosenberg et al.

The OFFICIAL repository of HAL is at [Alex-Rosenberg/cell-2015](https://github.com/Alex-Rosenberg/cell-2015).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing HAL did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

HAL is a linear (additive) model that scores alternative 5' splice-site usage from normalized hexamer (6-mer) frequencies across a 160-nucleotide donor-region window. It was learned from massively parallel reporter assays measuring splicing of millions of random synthetic sequences. The published coefficient table contains a `(4096, 8)` matrix of hexamer effects; the model averages the eight coefficient columns into one effect per hexamer and applies those effects to normalized hexamer frequencies.

### Model Specification

| Window | Published Coefficient Columns | Hexamer Features | Num Parameters | FLOPs | MACs  |
| ------ | ----------------------------- | ---------------- | -------------- | ----- | ----- |
| 160 nt | 8 averaged                    | 4,096            | 4,096          | 8,192 | 4,096 |

### Links

- **Code**: [multimolecule.hal](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/hal)
- **Data**: Rosenberg lab random-library 5' splice-site MPRA
- **Paper**: [Learning the Sequence Determinants of Alternative Splicing from Millions of Random Sequences](https://doi.org/10.1016/j.cell.2015.09.054)
- **Developed by**: Alexander B. Rosenberg, Rupali P. Patwardhan, Jay Shendure, Georg Seelig
- **Model type**: Linear regression over normalized hexamer-frequency features with learned per-hexamer effect coefficients
- **Original Repository**: [Alex-Rosenberg/cell-2015](https://github.com/Alex-Rosenberg/cell-2015)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Alternative Splicing Prediction

You can use this model directly to predict a splicing score for a 160-nucleotide DNA sequence window:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, HalForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/hal")
>>> model = HalForSequencePrediction.from_pretrained("multimolecule/hal")
>>> sequence = "ACGT" * 40
>>> input = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: 160 nt fixed donor-region window
- **Alphabet**: `ACGT` only; any hexamer spanning an unknown / `N` token is ignored
- **Special tokens**: do not add (`add_special_tokens=False`)
- **Output**: single scalar splicing score per window
- **Variant effect**: subtract two window scores and apply sigmoid externally for paired donor comparisons

## Training Details

HAL was learned from massively parallel splicing reporter assays in which millions of random synthetic sequences were inserted into an alternatively spliced reporter minigene. Splicing outcomes were measured by high-throughput sequencing of the resulting mRNA isoforms.

### Training Data

The model was trained on the splicing measurements of millions of degenerate (random) sequences from the reporter library described in the HAL paper. Hexamer coefficients were estimated by regressing the measured splicing index against the hexamer composition of each sequence.

### Training Procedure

#### Pre-training

HAL is a linear regression model. The published hexamer coefficient table is fit to the measured splicing index, and the model prediction is the linear combination of normalized hexamer frequencies with the averaged hexamer effects.

The HAL model uses the published `HAL_mer_scores.npz` hexamer coefficient table from Rosenberg et al. The table stores 4,096 hexamer rows and eight coefficient columns; the eight columns are averaged into the single per-hexamer effect used by the HAL formula.

## Citation

```bibtex
@article{rosenberg2015learning,
  author    = {Rosenberg, Alexander B. and Patwardhan, Rupali P. and Shendure, Jay and Seelig, Georg},
  journal   = {Cell},
  number    = 3,
  pages     = {698--711},
  publisher = {Elsevier BV},
  title     = {Learning the Sequence Determinants of Alternative Splicing from Millions of Random Sequences},
  volume    = 163,
  year      = 2015,
  doi       = {10.1016/j.cell.2015.09.054}
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

Please contact the authors of the [HAL paper](https://doi.org/10.1016/j.cell.2015.09.054) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
