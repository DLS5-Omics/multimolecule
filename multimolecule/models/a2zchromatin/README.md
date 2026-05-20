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

# a2z-chromatin

Recurrent convolutional neural network for predicting chromatin accessibility or lack of DNA methylation from angiosperm DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Modeling chromatin state from sequence across angiosperms using recurrent convolutional neural networks](https://doi.org/10.1002/tpg2.20249) by Travis Wrightsman, et al.

The OFFICIAL repository of a2z-chromatin is at [twrightsman/a2z-regulatory](https://github.com/twrightsman/a2z-regulatory).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing a2z-chromatin did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

a2z-chromatin is a recurrent convolutional neural network (CNN+BLSTM, DanQ topology) trained to predict chromatin state from a fixed-length 600 bp one-hot encoded angiosperm DNA sequence. The single convolutional layer applies 320 filters with a kernel size of 26, followed by dropout and a max-pool over 13 positions; the resulting feature sequence is fed to a bidirectional LSTM (320 units per direction) whose final forward and backward hidden states are concatenated, projected through a 925-unit dense layer, and read out as a single per-window probability.

Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Conv Layers | Num LSTM Layers   | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| --------------- | ----------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 1               | 1 (bidirectional) | 925         | 1.23               | 14.61     | 7.30     | 600            |

### Links

- **Code**: [multimolecule.a2zchromatin](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/a2zchromatin)
- **Data**: Leaf ATAC-seq from 12 angiosperm species + unmethylated-region calls from 10 angiosperms
- **Paper**: [Modeling chromatin state from sequence across angiosperms using recurrent convolutional neural networks](https://doi.org/10.1002/tpg2.20249)
- **Developed by**: Travis Wrightsman, Alexandre P. Marand, Peter A. Crisp, Nathan M. Springer, Edward S. Buckler
- **Model type**: 1D CNN + bidirectional LSTM over 600 bp angiosperm DNA for chromatin accessibility / methylation prediction
- **Original Repository**: [twrightsman/a2z-regulatory](https://github.com/twrightsman/a2z-regulatory)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Chromatin State Prediction

You can use this model directly to predict the chromatin accessibility (or lack of DNA methylation, for the methylation variant) of a 600 bp angiosperm DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, A2zChromatinForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/a2zchromatin-accessibility")
>>> model = A2zChromatinForSequencePrediction.from_pretrained("multimolecule/a2zchromatin-accessibility")
>>> input = tokenizer("ACGT" * 150, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: fixed 600 bp DNA window
- **Alphabet**: DNA IUPAC tokens; ambiguous bases use upstream fractional A/C/G/T mixtures, and non-IUPAC tokens map to zero
- **Output**: single per-window logit (binary chromatin accessibility for `a2z-accessibility`, lack of DNA methylation for `a2z-methylation`)

## Training Details

a2z-chromatin was trained to predict per-window chromatin state across angiosperms using a single shared cross-species DanQ topology.

### Training Data

a2z-chromatin was trained on two cross-species data resources:

- **Chromatin accessibility**: leaf ATAC-seq peaks from 12 angiosperm species, with each 600 bp genomic interval labelled as accessible or inaccessible.
- **DNA methylation**: unmethylated-region (UMR) calls from 10 angiosperm species, with each 600 bp genomic interval labelled as unmethylated or methylated. Unmethylated regions overlap significantly with accessible chromatin in plants, so the two tasks share the same architecture.

Each training example is a 600 bp one-hot encoded DNA sequence with a single binary label.

### Training Procedure

#### Pre-training

Each variant was trained to minimize a binary cross-entropy loss between its sigmoid-activated per-window prediction and the observed accessibility / unmethylation label, sweeping cross-species splits to evaluate generalization.

- Optimizer: Adam
- Loss: Binary cross-entropy
- Regularization: Dropout (0.2 after the convolution, 0.5 after the bidirectional LSTM)

## Citation

```bibtex
@article{wrightsman2022a2z,
  author    = {Wrightsman, Travis and Marand, Alexandre P. and Crisp, Peter A. and Springer, Nathan M. and Buckler, Edward S.},
  title     = {Modeling chromatin state from sequence across angiosperms using recurrent convolutional neural networks},
  journal   = {The Plant Genome},
  volume    = 15,
  number    = 3,
  pages     = {e20249},
  year      = 2022,
  publisher = {Wiley},
  doi       = {10.1002/tpg2.20249}
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

Please contact the authors of the [a2z-chromatin paper](https://doi.org/10.1002/tpg2.20249) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
