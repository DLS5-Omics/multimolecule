---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/encode
library_name: multimolecule
---

# ProCapNet

Base-resolution convolutional neural network for predicting PRO-cap transcription-initiation signal from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Dissecting the cis-regulatory syntax of transcription initiation with deep learning](https://doi.org/10.1101/2024.05.28.596138) by Kelly Cochran et al.

The OFFICIAL repository of ProCapNet is at [kundajelab/ProCapNet](https://github.com/kundajelab/ProCapNet).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing ProCapNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

ProCapNet is a convolutional neural network (CNN) trained to predict base-resolution PRO-cap transcription-initiation signal from primary DNA sequence. Its architecture is largely adapted from Jacob Schreiber's `bpnet-lite` and shares BPNet's dilated-convolution backbone and profile/count factorization. The output is two-stranded (plus / minus strand), mappability-aware, and reconstructed by `ProCapNetForProfilePrediction.postprocess`. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Input Length | Profile Length | Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ------------ | -------------- | ---------- | ----------- | ------------------ | --------- | -------- |
| 2114         | 1000           | 9          | 512         | 6.43               | 27.17     | 13.58    |

FLOPs and MACs are measured on the canonical 2114 bp ProCapNet input window.

### Links

- **Code**: [multimolecule.procapnet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/procapnet)
- **Weights**: [multimolecule/procapnet](https://huggingface.co/multimolecule/procapnet)
- **Data**: [K562 PRO-cap (ENCODE ENCSR261KBX)](https://www.encodeproject.org/experiments/ENCSR261KBX/)
- **Paper**: [Dissecting the cis-regulatory syntax of transcription initiation with deep learning](https://doi.org/10.1101/2024.05.28.596138)
- **Developed by**: Kelly Cochran, Melody Yin, Anika Mantripragada, Jacob Schreiber, Georgi K. Marinov, Sagar R. Shah, Haiyuan Yu, John T. Lis, Anshul Kundaje
- **Model type**: BPNet-derived 1D dilated CNN with two-stranded factorized profile-and-count heads for PRO-cap transcription-initiation prediction
- **Original Repository**: [kundajelab/ProCapNet](https://github.com/kundajelab/ProCapNet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Transcription-Initiation Profile Prediction

You can use this model directly to predict PRO-cap transcription-initiation profiles of a DNA sequence:

```python
>>> from multimolecule import DnaTokenizer, ProCapNetForProfilePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/procapnet")
>>> model = ProCapNetForProfilePrediction.from_pretrained("multimolecule/procapnet")
>>> output = model(**tokenizer(("ACGT" * 529)[:2114], return_tensors="pt"))

>>> output.keys()
odict_keys(['profile_logits', 'count_logits'])

>>> output["profile_logits"].shape
torch.Size([1, 1000, 2])

>>> output["count_logits"].shape
torch.Size([1, 1])

>>> track = model.postprocess(output)
>>> track.shape
torch.Size([1, 1000, 2])
```

The recombined `track` is the usable base-resolution prediction. The last dimension stacks the `num_strands` (plus, minus) PRO-cap signal predictions.

### Interface

- **Input length**: 2114 bp DNA window
- **Profile length**: 1000 bp, two-stranded (plus / minus)
- **Output**: factorized `(profile_logits, count_logits)`; recombine the base-resolution PRO-cap track via `ProCapNetForProfilePrediction.postprocess`

## Training Details

ProCapNet was trained to predict the base-resolution, two-stranded PRO-cap transcription-initiation signal in human cell lines. The default model is the K562 model.

### Training Data

The published ProCapNet models were trained on PRO-cap signal using ~2 kb genomic windows. The default K562 model was trained on K562 PRO-cap experiment [ENCSR261KBX](https://www.encodeproject.org/experiments/ENCSR261KBX/). Training and test regions, observed signal tracks, and contribution scores are distributed through the same ENCODE release.

### Training Procedure

#### Pre-training

The model was trained with a composite loss: a (strand-merged) multinomial negative log-likelihood on the per-position, two-stranded profile shape plus a mean-squared-error regression on `log(count + 1)` total counts.

- Optimizer: Adam
- Training is mappability-aware

## Citation

```bibtex
@article{cochran2024procapnet,
  author    = {Cochran, Kelly and Yin, Melody and Mantripragada, Anika and Schreiber, Jacob and Marinov, Georgi K. and Shah, Sagar R. and Yu, Haiyuan and Lis, John T. and Kundaje, Anshul},
  title     = {Dissecting the cis-regulatory syntax of transcription initiation with deep learning},
  journal   = {bioRxiv},
  year      = 2024,
  doi       = {10.1101/2024.05.28.596138},
  note      = {Preprint}
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

Please contact the authors of the [ProCapNet paper](https://doi.org/10.1101/2024.05.28.596138) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
