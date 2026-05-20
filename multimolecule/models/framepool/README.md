---
language: rna
tags:
  - Biology
  - RNA
  - 5' UTR
  - Translation
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: mean-ribosome-load
---

# Framepool

Frame-aware pooling convolutional network for predicting mean ribosome load from variable-length 5'UTR sequences.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting mean ribosome load for 5'UTR of any length using deep learning](https://doi.org/10.1371/journal.pcbi.1008982) by Alexander Karollus, et al.

The OFFICIAL repository of Framepool is at [Karollus/5UTR](https://github.com/Karollus/5UTR) and the published Kipoi wrapper is at [kipoi/models](https://github.com/kipoi/models/tree/master/Framepool).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Framepool did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Framepool is a small 1D convolutional network that predicts the mean ribosome load (MRL) of a human 5' untranslated region from sequence alone. It extends the fixed-length network of [Sample et al., 2019](https://doi.org/10.1038/s41587-019-0164-5) with a frame-aware pooling layer that reverses the sequence to anchor reading frames at the start codon, slices the convolutional feature map into the three reading frames, and applies global max and masked global average pooling per frame. The pooled representation is length-independent and is consumed by a small dense head followed by a per-sub-library scaling regression that recalibrates the prediction across the two training libraries (`egfp_unmod_1` and `random`). Please refer to the [Training Details](#training-details) section for more information on the training process.

The released `combined_residual` model is recommended by the upstream authors for variant effect scoring.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 4          | 768         | 0.28               | 0.05      | 0.02     | unlimited      |

### Links

- **Code**: [multimolecule.framepool](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/framepool)
- **Data**: eGFP polysome-profiling massively parallel reporter assay (MPRA) from [Sample et al., 2019](https://doi.org/10.1038/s41587-019-0164-5), HEK293T cells, fixed-length (50 nt) and variable-length (25-100 nt) 5'UTR libraries
- **Paper**: [Predicting mean ribosome load for 5'UTR of any length using deep learning](https://doi.org/10.1371/journal.pcbi.1008982)
- **Developed by**: Alexander Karollus, Žiga Avsec, Julien Gagneur
- **Model type**: 1D residual CNN with frame-aware pooling for mean-ribosome-load prediction from 5'UTR sequence
- **Original Repository**: [Karollus/5UTR](https://github.com/Karollus/5UTR)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Mean Ribosome Load Prediction

You can use this model directly to predict the mean ribosome load of a 5'UTR sequence:

```python
>>> from multimolecule import RnaTokenizer, FramepoolForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/framepool")
>>> model = FramepoolForSequencePrediction.from_pretrained("multimolecule/framepool")
>>> output = model(**tokenizer("ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUAC", return_tensors="pt"))

>>> output.keys()
odict_keys(['logits'])
```

### Interface

- **Input length**: variable; the upstream MPRA training data is 25-100 nt 5'UTR but the model accepts any length because of frame-aware pooling
- **Alphabet**: RNA (`A`, `C`, `G`, `U`); [`RnaTokenizer`][multimolecule.RnaTokenizer] converts `T` to `U`; `N` and other non-canonical tokens are encoded as all-zero columns and ignored by the masked pooling
- **Padding**: zero-padding is supported via `attention_mask` and is excluded from pooling
- **Output**: single scalar per sequence — predicted mean ribosome load (`logits`, shape `(batch_size, 1)`)
- **Auxiliary inputs**: optional `library_indicator` (shape `(batch_size, library_size)`) selecting one of the two training sub-libraries for the scaling regression. Defaults to the `random` library, matching the upstream Kipoi variant effect interface

### Variant Effect

Framepool supports paired reference/alternative scoring through the optional `alternative_input_ids` argument:

- **Single sequence (reference only)**: `logits` is the predicted mean ribosome load (one scalar per sequence)
- **Reference + alternative**: `logits` is the `log2` mean ribosome load fold change `log2(MRL_alt / MRL_ref)`, matching the Kipoi `UTRVariantEffectModel.predict_on_batch` `mrl_fold_change` output
- Reference and alternative sequences are scored independently; both must use the same `library_indicator` so that the scaling regression cancels out of the fold change
- For the upstream "shifted-frame" variant effect outputs (`shift_1`, `shift_2`), prepend one or two zero columns (or `N` tokens) to both reference and alternative inputs before scoring, matching the Kipoi loop

## Training Details

Framepool was trained on polysome-profiling MPRA data measuring the mean ribosome load of randomized 5'UTR sequences and uses frame-aware pooling so that a single network can score sequences of arbitrary length.

### Training Data

Framepool was trained on the eGFP polysome-profiling MPRA libraries of [Sample et al., 2019](https://doi.org/10.1038/s41587-019-0164-5) in HEK293T cells: the fixed-length library (`egfp_unmod_1`, 50 nt) and the variable-length library (`random`, 25-100 nt). Approximately 260,000 sequences were used for training, with 20,000 held out for testing; additional validation was performed on endogenous data.

### Training Procedure

#### Pre-training

- **Loss**: mean squared error between the predicted and measured mean ribosome load
- **Optimizer**: Adam with `lr = 1e-3`, `beta_1 = 0.9`, `beta_2 = 0.999`, `epsilon = 1e-8`
- **Epochs**: 6
- **Mini-batch sampling**: the two training libraries are mixed within every batch; a one-hot library indicator is fed to the scaling regression layer so that the network can absorb the library-specific offset

## Citation

```bibtex
@article{karollus2021predicting,
  author    = {Karollus, Alexander and Avsec, {\v Z}iga and Gagneur, Julien},
  title     = {Predicting mean ribosome load for 5{\textquoteright}UTR of any length using deep learning},
  journal   = {PLOS Computational Biology},
  volume    = {17},
  number    = {5},
  pages     = {e1008982},
  year      = {2021},
  publisher = {Public Library of Science},
  doi       = {10.1371/journal.pcbi.1008982}
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

Please contact the authors of the [Framepool paper](https://doi.org/10.1371/journal.pcbi.1008982) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
