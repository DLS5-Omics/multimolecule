---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/encode
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-activity
---

# Basset

Deep convolutional neural network for predicting chromatin accessibility (DNase I hypersensitivity) from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks](https://doi.org/10.1101/gr.200535.115) by David R. Kelley, et al.

The OFFICIAL repository of Basset is at [davek44/Basset](https://github.com/davek44/Basset).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Basset did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Basset is a convolutional neural network (CNN) trained to predict the chromatin accessibility (DNase I hypersensitivity) of a DNA sequence across 164 cell types. The model consumes a fixed-length 600 bp one-hot encoded DNA sequence and applies three convolutional blocks (convolution, batch normalization, ReLU, and max pooling) followed by two fully-connected blocks before a multi-label binary classification head. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Conv Layers | Num FC Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| --------------- | ------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 3               | 2             | 1000        | 4.14               | 0.30      | 0.15     | 600            |

### Links

- **Code**: [multimolecule.basset](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/basset)
- **Data**: ENCODE and Roadmap Epigenomics DNase-seq accessibility compendium across 164 cell types
- **Paper**: [Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks](https://doi.org/10.1101/gr.200535.115)
- **Developed by**: David R. Kelley, Jasper Snoek, John L. Rinn
- **Model type**: Three-layer 1D CNN over 600 bp DNA for multi-task chromatin-accessibility prediction
- **Original Repository**: [davek44/Basset](https://github.com/davek44/Basset)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Chromatin Accessibility Prediction

You can use this model directly to predict the DNase I hypersensitivity of a DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, BassetForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/basset")
>>> model = BassetForSequencePrediction.from_pretrained("multimolecule/basset")
>>> input = tokenizer("ACGT" * 150, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 164])
```

### Interface

- **Input length**: fixed 600 bp DNA window
- **Output**: 164 per-cell-type accessibility logits (multi-label binary)

## Training Details

Basset was trained to predict the chromatin accessibility of DNA sequences across a panel of cell types.

### Training Data

Basset was trained on DNase I hypersensitivity peaks from [ENCODE](https://www.encodeproject.org) and the [Roadmap Epigenomics](https://www.roadmapepigenomics.org) project, covering 164 cell types.
Each 600 bp genomic interval is labeled with a binary vector indicating which of the 164 cell types show an accessibility peak overlapping that interval.

### Training Procedure

#### Pre-training

The model was trained to minimize a multi-label binary cross-entropy loss, comparing its predicted per-cell-type accessibility probabilities against the observed DNase I hypersensitivity labels.

- Optimizer: RMSprop
- Loss: Multi-label binary cross-entropy
- Regularization: Batch normalization and dropout

## Citation

```bibtex
@article{kelley2016basset,
  author    = {Kelley, David R. and Snoek, Jasper and Rinn, John L.},
  title     = {Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks},
  journal   = {Genome Research},
  volume    = 26,
  number    = 7,
  pages     = {990--999},
  year      = 2016,
  publisher = {Cold Spring Harbor Laboratory Press},
  doi       = {10.1101/gr.200535.115}
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

Please contact the authors of the [Basset paper](https://doi.org/10.1101/gr.200535.115) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
