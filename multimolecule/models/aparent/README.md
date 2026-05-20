---
language: rna
tags:
  - Biology
  - RNA
  - 3' UTR
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: polyadenylation
---

# APARENT

Convolutional neural network for predicting human 3'UTR Alternative Polyadenylation (APA) from sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [A Deep Neural Network for Predicting and Engineering Alternative Polyadenylation](https://doi.org/10.1016/j.cell.2019.04.046) by Nicholas Bogard, Johannes Linder, et al.

The OFFICIAL repository of APARENT is at [johli/aparent](https://github.com/johli/aparent).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing APARENT did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

APARENT (APA REgression NeT) is a convolutional neural network trained on more than 3.5 million randomized 3'UTR poly-A signals expressed on mini-gene reporters in HEK293. Given a fixed-length 205 nt 3'UTR/polyA sequence, APARENT predicts the alternative-polyadenylation isoform proportion (a scalar) and a positional cleavage distribution. The model is primarily used to score the impact of genetic variants on APA regulation and to engineer new polyadenylation signals. Please refer to the [Training Details](#training-details) section for more information on the training process.

The base, non-normalised APARENT model is recommended by the original authors for isoform and variant-effect prediction.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 4          | 256         | 6.43               | 0.03      | 0.01     | 205            |

### Links

- **Code**: [multimolecule.aparent](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/aparent)
- **Data**: Massively-parallel polyadenylation MPRA, GEO [GSE113849](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113849)
- **Paper**: [A Deep Neural Network for Predicting and Engineering Alternative Polyadenylation](https://doi.org/10.1016/j.cell.2019.04.046)
- **Developed by**: Nicholas Bogard, Johannes Linder, Alexander B. Rosenberg, Georg Seelig
- **Model type**: 1D CNN for alternative polyadenylation isoform and cleavage prediction from 3'UTR sequence
- **Original Repository**: [johli/aparent](https://github.com/johli/aparent)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### APA Isoform Prediction

You can use this model directly to predict the APA isoform proportion of a 3'UTR/polyA sequence:

```python
>>> from multimolecule import RnaTokenizer, AparentForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/aparent")
>>> model = AparentForSequencePrediction.from_pretrained("multimolecule/aparent")
>>> output = model(**tokenizer("ACGUACGUACGU", return_tensors="pt"))

>>> output.keys()
odict_keys(['logits'])
```

The full APARENT isoform and cleavage outputs are available on the backbone:

```python
>>> from multimolecule import RnaTokenizer, AparentModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/aparent")
>>> model = AparentModel.from_pretrained("multimolecule/aparent")
>>> output = model(**tokenizer("ACGUACGUACGU", return_tensors="pt"))

>>> output.keys()
odict_keys(['pooler_output', 'isoform_logits', 'cleavage_logits'])
```

### Interface

- **Input length**: fixed 205 nt 3'UTR / polyA sequence
- **Output (`AparentModel`)**: `isoform_logits` (scalar APA proportion) + `cleavage_logits` (206-dim positional cleavage distribution)
- **Output (`AparentForSequencePrediction`)**: APA isoform scalar only (`logits`)

## Training Details

APARENT was trained to jointly predict the APA isoform proportion and the positional cleavage distribution of randomized 3'UTR poly-A signals.

### Training Data

APARENT was trained on more than 3.5 million randomized 3'UTR poly-A signal sequences expressed on mini-gene reporters in HEK293 cells (a massively parallel reporter assay, MPRA). The raw sequencing data for the 3'UTR MPRA libraries are available at GEO accession GSE113849.

This APARENT model was trained on all MPRA libraries (no libraries held out) to produce the best general-purpose APA predictor; it differs from the per-library held-out model evaluated in the paper.

### Training Procedure

#### Pre-training

The model was trained to minimize a combined objective: a sigmoid KL-divergence on the isoform proportion and a KL-divergence on the positional cleavage distribution, weighted equally.

## Citation

```bibtex
@article{bogard2019adeep,
  author    = {Bogard, Nicholas and Linder, Johannes and Rosenberg, Alexander B. and Seelig, Georg},
  title     = {A Deep Neural Network for Predicting and Engineering Alternative Polyadenylation},
  journal   = {Cell},
  volume    = {178},
  number    = {1},
  pages     = {91--106.e23},
  year      = {2019},
  publisher = {Elsevier BV},
  doi       = {10.1016/j.cell.2019.04.046}
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

Please contact the authors of the [APARENT paper](https://doi.org/10.1016/j.cell.2019.04.046) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
