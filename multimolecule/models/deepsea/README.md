---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-variant-effect
---

# DeepSEA

Deep convolutional neural network that predicts noncoding chromatin features (DNase I hypersensitivity, transcription-factor binding, and histone marks) from DNA sequence, used to score the regulatory impact of noncoding variants.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting effects of noncoding variants with deep learning-based sequence model](https://doi.org/10.1038/nmeth.3547) by Jian Zhou, et al.

The OFFICIAL repository of DeepSEA is at [jisraeli/DeepSEA](http://deepsea.princeton.edu).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeepSEA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeepSEA is a convolutional neural network (CNN) trained to predict 919 chromatin features—DNase I hypersensitivity peaks, transcription-factor binding peaks, and histone-mark peaks—across multiple human cell types from a fixed-length 1000 bp DNA sequence. The model applies three convolutional blocks (convolution, ReLU, max pooling, and dropout) followed by a single fully-connected layer and a multi-label sigmoid output. The sequence-prediction model averages forward and reverse-complement probabilities. The trained model is then used to score the regulatory impact of noncoding single-nucleotide variants by computing the difference between reference- and alternate-allele predictions. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Conv Layers | Num FC Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| --------------- | ------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 3               | 1             | 925         | 52.84              | 1.10      | 0.55     | 1000           |

### Links

- **Code**: [multimolecule.deepsea](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deepsea)
- **Data**: ENCODE and Roadmap Epigenomics chromatin-feature peak compendium covering 690 transcription-factor binding profiles, 125 DNase I hypersensitivity profiles, and 104 histone-mark profiles (919 chromatin features in total)
- **Paper**: [Predicting effects of noncoding variants with deep learning-based sequence model](https://doi.org/10.1038/nmeth.3547)
- **Developed by**: Jian Zhou, Olga G. Troyanskaya
- **Model type**: Three-layer 1D CNN over 1000 bp DNA for multi-task chromatin-feature prediction
- **Original Repository**: [DeepSEA](http://deepsea.princeton.edu)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Chromatin Feature Prediction

You can use this model directly to predict the chromatin features of a DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, DeepSeaForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepsea")
>>> model = DeepSeaForSequencePrediction.from_pretrained("multimolecule/deepsea")
>>> input = tokenizer("ACGT" * 250, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 919])
```

### Interface

- **Input length**: fixed 1000 bp DNA window
- **Output**: 919 chromatin-feature logits (multi-label binary), covering DNase I hypersensitivity, transcription-factor binding, and histone-mark peaks across multiple cell types

## Training Details

DeepSEA was trained to predict the chromatin features of DNA sequences across a panel of human cell types and then used to score the regulatory impact of noncoding variants.

### Training Data

DeepSEA was trained on chromatin profiling data from [ENCODE](https://www.encodeproject.org) and the [Roadmap Epigenomics](https://www.roadmapepigenomics.org) project, comprising 690 transcription-factor ChIP-seq profiles, 125 DNase I hypersensitivity profiles, and 104 histone-mark ChIP-seq profiles for a total of 919 chromatin features. Each 1000 bp genomic interval centered on a 200 bp bin is labeled with a binary vector indicating which of the 919 chromatin features have a peak overlapping the central bin.

### Training Procedure

#### Pre-training

The model was trained to minimize a multi-label binary cross-entropy loss, comparing its predicted per-feature probabilities against the observed chromatin-feature labels.

- Optimizer: Stochastic gradient descent with momentum
- Loss: Multi-label binary cross-entropy
- Regularization: Dropout (0.2 after the first two convolutions, 0.5 after the third convolution) and L2 weight decay

## Citation

```bibtex
@article{zhou2015deepsea,
  author    = {Zhou, Jian and Troyanskaya, Olga G.},
  title     = {Predicting effects of noncoding variants with deep learning-based sequence model},
  journal   = {Nature Methods},
  volume    = 12,
  number    = 10,
  pages     = {931--934},
  year      = 2015,
  publisher = {Nature Publishing Group},
  doi       = {10.1038/nmeth.3547}
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

Please contact the authors of the [DeepSEA paper](https://doi.org/10.1038/nmeth.3547) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
