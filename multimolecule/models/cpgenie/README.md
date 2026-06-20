---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: methylation-prediction
---

# CpGenie

Deep convolutional neural network for predicting CpG methylation level and the impact of non-coding variants on DNA methylation.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting the impact of non-coding variants on DNA methylation](https://doi.org/10.1093/nar/gkx177) by **Haoyang Zeng et al.**

The OFFICIAL repository of CpGenie is at [gifford-lab/CpGenie](https://github.com/gifford-lab/CpGenie).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing CpGenie did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

CpGenie is a convolutional neural network (CNN) trained to predict the methylation state of the central CpG dinucleotide in a fixed-length 1001 bp DNA window. The model consumes a one-hot encoded sequence whose 501st nucleotide is the `C` of the CpG, and applies three convolutional blocks (convolution with same padding, ReLU activation, and valid max pooling with window 5 and stride 3) followed by two fully-connected layers with dropout and a 2-class softmax classifier (unmethylated vs. methylated). Please refer to the [Training Details](#training-details) section for more information on the training process.

The upstream authors distribute CpGenie as an ensemble of 50 single-cell-line models, each trained on a different ENCODE reduced-representation bisulfite sequencing (RRBS) dataset, and use the per-model methylation probabilities to score the impact of non-coding variants on DNA methylation by comparing reference and alternative sequences.

### Model Specification

| Num Conv Layers | Num FC Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| --------------- | ------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 3               | 2             | 64          | 2.01               | 0.26      | 0.13     | 1001           |

### Links

- **Code**: [multimolecule.cpgenie](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/cpgenie)
- **Weights**: [multimolecule/cpgenie](https://huggingface.co/multimolecule/cpgenie)
- **Data**: ENCODE reduced-representation bisulfite sequencing (RRBS) datasets across 50 immortal cell lines
- **Paper**: [Predicting the impact of non-coding variants on DNA methylation](https://doi.org/10.1093/nar/gkx177)
- **Developed by**: Haoyang Zeng, David K. Gifford
- **Model type**: Three-layer 1D CNN over 1001 bp DNA for CpG methylation prediction
- **Original Repository**: [gifford-lab/CpGenie](https://github.com/gifford-lab/CpGenie)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### CpG Methylation Prediction

You can use this model directly to predict the methylation state of the central CpG of a 1001 bp DNA window:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, CpGenieForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/cpgenie")
>>> model = CpGenieForSequencePrediction.from_pretrained("multimolecule/cpgenie")
>>> sequence = "A" * 500 + "CG" + "T" * 499
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 2])
>>> torch.softmax(output.logits, dim=-1).shape
torch.Size([1, 2])
```

### Interface

- **Input length**: fixed 1001 bp DNA window
- **CpG position**: the 501st nucleotide must be the `C` of the central CpG dinucleotide
- **Output**: 2 methylation logits (unmethylated vs. methylated) for the central CpG of a single ENCODE RRBS cell line

## Training Details

CpGenie was trained to predict the methylation state of the central CpG dinucleotide of a 1001 bp DNA window from sequence alone.

### Training Data

CpGenie was trained on reduced-representation bisulfite sequencing (RRBS) datasets from [ENCODE](https://www.encodeproject.org), covering 50 immortal human cell lines.
Each training example is a 1001 bp genomic window centered on a CpG dinucleotide, with a binary label indicating whether the central CpG is methylated in the target cell line.
The upstream authors release 50 per-cell-line models that together form the CpGenie ensemble used for downstream variant-effect scoring.

### Training Procedure

#### Pre-training

The model was trained to minimize a 2-class softmax cross-entropy loss between the predicted methylation distribution and the observed RRBS-derived methylation label.

- Optimizer: RMSprop
- Loss: 2-class softmax cross-entropy
- Regularization: Dropout, max-norm weight constraint

## Citation

```bibtex
@article{zeng2017cpgenie,
  author    = {Zeng, Haoyang and Gifford, David K.},
  title     = {Predicting the impact of non-coding variants on {DNA} methylation},
  journal   = {Nucleic Acids Research},
  volume    = 45,
  number    = 11,
  pages     = {e99},
  year      = 2017,
  publisher = {Oxford University Press},
  doi       = {10.1093/nar/gkx177}
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

Please contact the authors of the [CpGenie paper](https://doi.org/10.1093/nar/gkx177) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
