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

# Xpresso

Deep convolutional neural network for predicting mRNA abundance directly from genomic promoter sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks](https://doi.org/10.1016/j.celrep.2020.107663) by Vikram Agarwal, et al.

The OFFICIAL repository of Xpresso is at [vagarwal87/Xpresso](https://github.com/vagarwal87/Xpresso).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Xpresso did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Xpresso is a deep convolutional neural network (CNN) that predicts steady-state mRNA expression level directly from genomic sequence. It consumes a promoter window of roughly 10.5 kb centered on the transcription start site (TSS), processes it through a stack of 1D convolution + max-pooling blocks, flattens the result, concatenates a small set of auxiliary numeric mRNA half-life features, and passes the combined representation through fully-connected layers to predict a single scalar expression value. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Input Length | Conv Blocks | Hidden Size | Auxiliary Features | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ------------ | ----------- | ----------- | ------------------ | ------------------ | --------- | -------- | -------------- |
| 10,500       | 2           | 2           | 6                  | 0.11               | 0.11      | 0.05     | 10,500         |

### Links

- **Code**: [multimolecule.xpresso](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/xpresso)
- **Data**: Roadmap Epigenomics gene-expression data with promoter sequence and mRNA half-life features
- **Paper**: [Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks](https://doi.org/10.1016/j.celrep.2020.107663)
- **Developed by**: Vikram Agarwal, Jay Shendure
- **Model type**: 1D CNN over promoter DNA combined with auxiliary mRNA half-life features for mRNA-abundance regression
- **Original Repository**: [vagarwal87/Xpresso](https://github.com/vagarwal87/Xpresso)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### mRNA Expression Prediction

You can use this model directly to predict the mRNA expression of a promoter sequence together with its auxiliary mRNA half-life features:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, XpressoForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/xpresso")
>>> model = XpressoForSequencePrediction.from_pretrained("multimolecule/xpresso")
>>> input = tokenizer("ACGTACGTACGTACGT", return_tensors="pt")
>>> features = torch.randn(1, model.config.num_features)
>>> output = model(**input, features=features)

>>> output.logits.shape
torch.Size([1, 1])
```

The auxiliary half-life features are passed through the `features` argument as a float tensor of shape `(batch_size, num_features)`. Models configured with a non-zero `num_features` require this tensor; models configured with `num_features=0` do not accept it.

### Interface

- **Input length**: fixed 10,500 bp promoter window centered on the TSS
- **Padding**: shorter inputs right-padded; longer inputs center-cropped to `input_length`
- **Auxiliary inputs**: `features` tensor of shape `(batch_size, num_features)` required when `num_features > 0`; not accepted when `num_features = 0`
- **Output**: scalar mRNA expression

## Training Details

Xpresso was trained to predict steady-state mRNA expression levels (median across tissues/cell lines) from genomic promoter sequence.

### Training Data

Xpresso was trained on human and mouse genes, using promoter sequences (~10.5 kb windows centered on the TSS) together with mRNA half-life features derived from gene-body and UTR properties. Expression targets are log-transformed median mRNA levels across tissues.

The Xpresso model follows the published `humanMedian` configuration.

### Training Procedure

#### Pre-training

The model was trained to minimize a mean-squared-error loss between predicted and observed log mRNA expression values.

- Optimizer: Adam
- Loss: Mean squared error

## Citation

```bibtex
@article{agarwal2020predicting,
  author    = {Agarwal, Vikram and Shendure, Jay},
  journal   = {Cell Reports},
  number    = 7,
  pages     = {107663},
  publisher = {Elsevier BV},
  title     = {Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks},
  volume    = 31,
  year      = 2020,
  doi       = {10.1016/j.celrep.2020.107663}
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

Please contact the authors of the [Xpresso paper](https://doi.org/10.1016/j.celrep.2020.107663) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
