---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/deepstarr
library_name: multimolecule
pipeline_tag: other
pipeline: regulatory-activity
---

# DeepSTARR

Convolutional neural network for predicting enhancer activity directly from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers](https://doi.org/10.1038/s41588-022-01048-5) by Bernardo P. de Almeida, Franziska Reiter, et al.

The OFFICIAL repository of DeepSTARR is at [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeepSTARR did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeepSTARR is a convolutional neural network (CNN) trained to quantitatively predict enhancer activity from 249 bp DNA sequences. The model was trained on genome-wide STARR-seq data from _Drosophila melanogaster_ S2 cells and predicts two regression outputs: developmental and housekeeping enhancer activity. The architecture consists of four convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPool) followed by two fully-connected layers. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Conv Layers | Num FC Layers | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| --------------- | ------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 4               | 2             | 256         | 0.62               | 21.03     | 10.26    | 249            |

### Links

- **Code**: [multimolecule.deepstarr](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deepstarr)
- **Data**: Drosophila S2 UMI-STARR-seq enhancer-activity data
- **Paper**: [DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers](https://doi.org/10.1038/s41588-022-01048-5)
- **Developed by**: Bernardo P. de Almeida, Franziska Reiter, Michaela Pagani, Alexander Stark
- **Model type**: Four-block 1D CNN over 249 bp DNA for developmental and housekeeping enhancer-activity regression
- **Original Repository**: [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Enhancer Activity Prediction

You can use this model directly to predict the developmental and housekeeping enhancer activity of a 249 bp DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, DeepStarrForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepstarr")
>>> model = DeepStarrForSequencePrediction.from_pretrained("multimolecule/deepstarr")
>>> sequence = "ACGT" * 62 + "A"
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 2])
```

### Interface

- **Input length**: fixed 249 bp DNA window
- **Output**: 2 regression outputs (developmental and housekeeping enhancer activity, log2 enrichment over input)

## Training Details

DeepSTARR was trained to predict quantitative enhancer activity from DNA sequence.

### Training Data

DeepSTARR was trained on genome-wide UMI-STARR-seq data from _Drosophila melanogaster_ S2 cells, measuring enhancer activity under two transcriptional programs: a developmental program (driven by a developmental core promoter) and a housekeeping program (driven by a housekeeping core promoter).

Each training example is a 249 bp genomic sequence with two continuous activity values (developmental and housekeeping, log2 enrichment over input).
Chromosomes were split into training, validation, and test sets to avoid sequence leakage.

### Training Procedure

#### Pre-training

The model was trained to minimize a mean-squared-error loss between predicted and measured enhancer activities.

- Optimizer: Adam
- Learning rate: 2e-3
- Loss: Mean Squared Error
- Early stopping on validation loss

## Citation

```bibtex
@article{deAlmeida2022deepstarr,
  author    = {de Almeida, Bernardo P. and Reiter, Franziska and Pagani, Michaela and Stark, Alexander},
  journal   = {Nature Genetics},
  month     = may,
  number    = 5,
  pages     = {613--624},
  publisher = {Springer Science and Business Media LLC},
  title     = {{DeepSTARR} predicts enhancer activity from {DNA} sequence and enables the de novo design of synthetic enhancers},
  volume    = 54,
  year      = 2022,
  doi       = {10.1038/s41588-022-01048-5}
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

Please contact the authors of the [DeepSTARR paper](https://doi.org/10.1038/s41588-022-01048-5) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
