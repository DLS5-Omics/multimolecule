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

# Optimus 5-Prime

Convolutional neural network that predicts the mean ribosome load (MRL) of a fixed 50 nt human 5' untranslated region (5'UTR) from sequence alone.

## Disclaimer

This is an UNOFFICIAL implementation of [Human 5' UTR design and variant effect prediction from a massively parallel translation assay](https://doi.org/10.1038/s41587-019-0164-5) by Paul J. Sample, et al.

The OFFICIAL repository of Optimus 5-Prime is at [pjsample/human_5utr_modeling](https://github.com/pjsample/human_5utr_modeling).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Optimus 5-Prime did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Optimus 5-Prime is a simple, fully feed-forward 1D convolutional network trained on a massively parallel polysome-profiling assay of ~280,000 random 50 nt 5'UTRs upstream of an eGFP reporter expressed in HEK293T. The network ingests a fixed 50 nt 5'UTR one-hot tensor, applies three stacked `padding="same"` 1D convolutions (120 filters, kernel 8, ReLU) with dropout between the second/third convolutions, flattens the per-position activations channels-last, and emits a single standardized mean ribosome load (MRL) regression score through a 40-unit fully connected layer and a linear regression head. Please refer to the [Training Details](#training-details) section for more information on the training process.

The MRL scalar is the per-sequence mean of polysome-profile-derived ribosome loading and is used by the original authors both to score natural human 5'UTRs and to engineer new sequences with predictable translation efficiency. Variant-effect scoring is performed externally by computing the MRL difference between the reference and alternative sequences; the model itself takes a single sequence as input.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 4          | 40          | 0.48               | 24.04     | 12.00    | 50             |

### Links

- **Code**: [multimolecule.optimus5prime](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/optimus5prime)
- **Data**: Massively parallel polysome-profiling MRL library on randomized 50 nt 5'UTRs in HEK293T, GEO [GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002)
- **Paper**: [Human 5' UTR design and variant effect prediction from a massively parallel translation assay](https://doi.org/10.1038/s41587-019-0164-5)
- **Developed by**: Paul J. Sample, Ban Wang, David W. Reid, Vlad Presnyak, Iain J. McFadyen, David R. Morris, Georg Seelig
- **Model type**: 1D CNN for mean ribosome load (MRL) regression from a fixed 50 nt 5'UTR sequence
- **Original Repository**: [pjsample/human_5utr_modeling](https://github.com/pjsample/human_5utr_modeling)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Mean Ribosome Load Prediction

You can use this model directly to predict the mean ribosome load (MRL) of a fixed 50 nt 5'UTR sequence:

```python
>>> from multimolecule import RnaTokenizer, Optimus5PrimeForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/optimus5prime")
>>> model = Optimus5PrimeForSequencePrediction.from_pretrained("multimolecule/optimus5prime")
>>> output = model(**tokenizer("GGGACAUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC", return_tensors="pt"))

>>> output.keys()
odict_keys(['logits'])
```

The pre-regression dense representation is exposed on the backbone:

```python
>>> from multimolecule import RnaTokenizer, Optimus5PrimeModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/optimus5prime")
>>> model = Optimus5PrimeModel.from_pretrained("multimolecule/optimus5prime")
>>> output = model(**tokenizer("GGGACAUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC", return_tensors="pt"))

>>> output.keys()
odict_keys(['pooler_output'])
```

### Interface

- **Input length**: fixed 50 nt 5'UTR sequence
- **Padding**: shorter sequences are right-padded with zeros to 50 nt; longer sequences are truncated to the first 50 nt
- **Alphabet**: RNA (`A`, `C`, `G`, `U`); `N` is encoded as an all-zero channel
- **Special tokens**: none added; `input_ids` are consumed positionally as one-hot channels
- **Output**: standardized mean ribosome load score (`logits`) of shape `(batch_size, 1)`; raw-MRL calibration requires the external scaler used by the upstream training workflow

### Variant Effect

Optimus 5-Prime is a single-sequence regression model. To score the effect of a variant on translation, run the reference and alternative 5'UTRs through the model independently and compute the difference between their predicted MRL values:

```python
>>> from multimolecule import RnaTokenizer, Optimus5PrimeForSequencePrediction
>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/optimus5prime")
>>> model = Optimus5PrimeForSequencePrediction.from_pretrained("multimolecule/optimus5prime")
>>> ref = "GGGACAUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGC"
>>> alt = "GGGACAUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGAUAGC"
>>> ref_mrl = model(**tokenizer(ref, return_tensors="pt"))["logits"]
>>> alt_mrl = model(**tokenizer(alt, return_tensors="pt"))["logits"]
>>> delta = (alt_mrl - ref_mrl).item()
```

## Training Details

Optimus 5-Prime was trained to regress the per-sequence mean ribosome load (MRL) derived from polysome profiling on a massively parallel reporter assay.

### Training Data

Optimus 5-Prime was trained on approximately 280,000 randomized 50 nt 5'UTRs placed upstream of an eGFP reporter and expressed in HEK293T cells. Mean ribosome load was computed per sequence from polysome-fractionation read counts. The raw sequencing data are available at GEO accession [GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002).

### Training Procedure

#### Pre-training

The published `main_MRL_model` was trained with mean-squared-error loss against standardized per-sequence MRL values. The optimizer was Adam with learning rate 1e-3, batch size 128, betas (0.9, 0.999), and epsilon 1e-8.

## Citation

```bibtex
@article{sample2019human,
  author    = {Sample, Paul J. and Wang, Ban and Reid, David W. and Presnyak, Vlad and McFadyen, Iain J. and Morris, David R. and Seelig, Georg},
  title     = {Human 5' UTR design and variant effect prediction from a massively parallel translation assay},
  journal   = {Nature Biotechnology},
  volume    = {37},
  number    = {7},
  pages     = {803--809},
  year      = {2019},
  publisher = {Springer Science and Business Media LLC},
  doi       = {10.1038/s41587-019-0164-5}
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

Please contact the authors of the [Optimus 5-Prime paper](https://doi.org/10.1038/s41587-019-0164-5) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
