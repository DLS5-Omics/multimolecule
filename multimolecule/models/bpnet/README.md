---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/bpnet-oskn
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-profile
---

# BPNet

Base-resolution convolutional neural network for predicting transcription-factor binding profiles from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Base-resolution models of transcription-factor binding reveal soft motif syntax](https://doi.org/10.1038/s41588-021-00782-6) by Žiga Avsec, Melanie Weilert, et al.

The OFFICIAL repository of BPNet is at [kundajelab/bpnet](https://github.com/kundajelab/bpnet).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing BPNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

BPNet is a convolutional neural network (CNN) trained to predict base-resolution transcription-factor binding signal (ChIP-nexus) from primary DNA sequence. It uses a convolutional motif stem followed by a stack of dilated residual convolutions that aggregate ~1 kb of genomic context. The output is factorized into profile and count branches, and the usable base-resolution prediction is reconstructed by `BpNetForProfilePrediction.postprocess`. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ---------- | ----------- | ------------------ | --------- | -------- |
| 10         | 64          | 0.13               | 0.24      | 0.12     |

### Links

- **Code**: [multimolecule.bpnet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/bpnet)
- **Data**: [BPNet manuscript data](https://zenodo.org/records/4294904)
- **Paper**: [Base-resolution models of transcription-factor binding reveal soft motif syntax](https://doi.org/10.1038/s41588-021-00782-6)
- **Developed by**: Žiga Avsec, Melanie Weilert, Avanti Shrikumar, Sabrina Krueger, Amr Alexandari, Khyati Dalal, Robin Fropf, Charles McAnany, Julien Gagneur, Anshul Kundaje, Julia Zeitlinger
- **Model type**: 1D dilated CNN with factorized profile-and-count heads for base-resolution transcription-factor binding prediction
- **Original Repository**: [kundajelab/bpnet](https://github.com/kundajelab/bpnet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Transcription-Factor Binding Profile Prediction

You can use this model directly to predict transcription-factor binding profiles of a DNA sequence:

```python
>>> from multimolecule import DnaTokenizer, BpNetForProfilePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/bpnet")
>>> model = BpNetForProfilePrediction.from_pretrained("multimolecule/bpnet")
>>> output = model(**tokenizer("ACGTNACGTN", return_tensors="pt"))

>>> output.keys()
odict_keys(['profile_logits', 'count_logits'])

>>> output["profile_logits"].shape
torch.Size([1, 10, 8])

>>> output["count_logits"].shape
torch.Size([1, 8])

>>> track = model.postprocess(output)
>>> track.shape
torch.Size([1, 10, 8])
```

The recombined `track` is the usable base-resolution prediction. The last dimension stacks `num_tasks` (Oct4, Sox2, Nanog, Klf4) by `num_strands` (forward, reverse).

### Interface

- **Input length**: 1000 bp DNA window
- **Output**: factorized `(profile_logits, count_logits)`; recombine the usable base-resolution track via `BpNetForProfilePrediction.postprocess`
- **Output shape**: `(batch_size, profile_length, num_tasks × num_strands)`; Oct4 / Sox2 / Nanog / Klf4 × forward / reverse = 8 channels

## Training Details

BPNet was trained to predict the base-resolution ChIP-nexus binding profiles of the pluripotency transcription factors Oct4, Sox2, Nanog and Klf4 in mouse embryonic stem cells.

### Training Data

The published BPNet-OSKN model was trained on ChIP-nexus profiles for Oct4, Sox2, Nanog and Klf4, using 1 kb genomic windows centered on detected binding peaks. The training regions and trained model files are distributed as part of the [BPNet manuscript data](https://zenodo.org/records/4294904).

### Training Procedure

#### Pre-training

The model was trained with a composite loss: a multinomial negative log-likelihood on the per-position profile shape plus a mean-squared-error regression on the log total counts.

- Optimizer: Adam

## Citation

```bibtex
@article{avsec2021baseresolution,
  author    = {Avsec, {\v{Z}}iga and Weilert, Melanie and Shrikumar, Avanti and Krueger, Sabrina and Alexandari, Amr and Dalal, Khyati and Fropf, Robin and McAnany, Charles and Gagneur, Julien and Kundaje, Anshul and Zeitlinger, Julia},
  title     = {Base-resolution models of transcription-factor binding reveal soft motif syntax},
  journal   = {Nature Genetics},
  volume    = 53,
  number    = 3,
  pages     = {354--366},
  year      = 2021,
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41588-021-00782-6}
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

Please contact the authors of the [BPNet paper](https://doi.org/10.1038/s41588-021-00782-6) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
