---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-profile
---

# ChromBPNet

Bias-factorized, base-resolution convolutional neural network for predicting chromatin accessibility (ATAC-seq / DNase-seq) from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants](https://doi.org/10.1101/2024.12.25.630221) by Anusri Pampari, et al.

The OFFICIAL repository of ChromBPNet is at [kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ChromBPNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ChromBPNet is a convolutional neural network (CNN) trained to predict base-resolution chromatin accessibility (ATAC-seq or DNase-seq) from primary DNA sequence with explicit enzyme-bias correction. It builds on the BPNet architecture and internally composes a bias sub-model with an accessibility sub-model. The composed output is factorized into profile and count branches, and the usable base-resolution prediction is reconstructed by `ChromBpNetForProfilePrediction.postprocess`. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Input Length | Profile Length | Num Layers | Hidden Size | Bias Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ------------ | -------------- | ---------- | ----------- | ---------------- | ------------------ | --------- | -------- |
| 2114         | 1000           | 9 + 5      | 512         | 128              | 6.61               | 27.83     | 13.91    |

The accessibility sub-model has 1 stem convolution + 8 dilated residual blocks (512 filters); the bias sub-model has 1 stem convolution + 4 dilated residual blocks (128 filters).
FLOPs and MACs are measured on the canonical 2114 bp ChromBPNet input window.

### Links

- **Code**: [multimolecule.chrombpnet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/chrombpnet)
- **Data**: [RoboATAC ChromBPNet Models](https://zenodo.org/records/16295014)
- **Paper**: [ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants](https://doi.org/10.1101/2024.12.25.630221)
- **Developed by**: Anusri Pampari, Anna Shcherbina, Evgeny Kvon, Michael Kosicki, Surag Nair, Soumya Kundu, Arwa S. Kathiria, Viviana I. Risca, Kristiina Simola, Melissa J. Funk, Eileen E. M. Furlong, Len A. Pennacchio, William J. Greenleaf, Anshul Kundaje
- **Model type**: BPNet-style 1D dilated CNN composed with an enzyme-bias model for bias-corrected chromatin-accessibility prediction
- **Original Repository**: [kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Chromatin Accessibility Profile Prediction

You can use this model directly to predict base-resolution chromatin accessibility of a DNA sequence:

```python
>>> from multimolecule import DnaTokenizer, ChromBpNetForProfilePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/chrombpnet")
>>> model = ChromBpNetForProfilePrediction.from_pretrained("multimolecule/chrombpnet")
>>> output = model(**tokenizer(("ACGT" * 529)[:2114], return_tensors="pt"))

>>> output.keys()
odict_keys(['profile_logits', 'count_logits'])

>>> output["profile_logits"].shape
torch.Size([1, 1000, 1])

>>> output["count_logits"].shape
torch.Size([1, 1])

>>> track = model.postprocess(output)
>>> track.shape
torch.Size([1, 1000, 1])
```

The recombined `track` is the usable, bias-corrected base-resolution accessibility prediction.

### Interface

- **Input length**: 2114 bp DNA window
- **Profile length**: 1000 bp
- **Output**: factorized `(profile_logits, count_logits)`; recombine the bias-corrected base-resolution track via `ChromBpNetForProfilePrediction.postprocess`
- **Composition**: profile logits added across bias + accessibility sub-models; counts combined via `logsumexp`

## Training Details

ChromBPNet was trained to predict base-resolution chromatin accessibility profiles from ATAC-seq / DNase-seq with explicit enzyme-bias correction.

### Training Data

The ChromBPNet model follows the HEK293T GFP-control model from the [RoboATAC ChromBPNet Models](https://zenodo.org/records/16295014) release (an automated ATAC-seq dataset from the Kundaje/Greenleaf labs). The accessibility and scaled-bias sub-models are combined for bias-corrected prediction.

### Training Procedure

#### Pre-training

The model was trained with a composite loss: a multinomial negative log-likelihood on the per-position profile shape plus a mean-squared-error regression on the log total counts.

- Optimizer: Adam

## Citation

```bibtex
@article{pampari2024chrombpnet,
  author    = {Pampari, Anusri and Shcherbina, Anna and Kvon, Evgeny and Kosicki, Michael and Nair, Surag and Kundu, Soumya and Kathiria, Arwa S. and Risca, Viviana I. and Simola, Kristiina and Funk, Melissa J. and Furlong, Eileen E. M. and Pennacchio, Len A. and Greenleaf, William J. and Kundaje, Anshul},
  title     = {ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants},
  journal   = {bioRxiv},
  year      = 2024,
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.1101/2024.12.25.630221},
  note      = {Preprint}
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

Please contact the authors of the [ChromBPNet paper](https://doi.org/10.1101/2024.12.25.630221) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
