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

# OptMRL

Convolutional neural network for predicting the mean ribosome load (MRL) of an mRNA from the 50 nucleotides upstream of the coding sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Interpreting Deep Neural Networks for the Prediction of Translation Rates](https://doi.org/10.1101/2023.06.02.543405) by Frederick Korbel, et al.

The OFFICIAL repository of OptMRL is at [ohlerlab/mlcis](https://github.com/ohlerlab/mlcis).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing OptMRL did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

OptMRL is a small 1D convolutional neural network trained to predict the mean ribosome load (MRL), a polysome-profiling-derived translation efficiency proxy, from the 50 nucleotides of 5' untranslated region (5'UTR) sequence immediately upstream of the coding sequence. The model was first pre-trained on roughly 260,000 random 5'UTR reporters and then fine-tuned on roughly 20,000 endogenous human 5'UTRs. Please refer to the [Training Details](#training-details) section for more information on the training process.

The architecture is a stack of three `Conv1D` layers (120 filters, kernel size 8, `same` padding, ReLU activation) followed by a `Flatten`, a 40-unit `Dense` bottleneck with ReLU activation and dropout, and a final scalar `Dense` regression head.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 5          | 40          | 0.476              | 24.04     | 12.00    | 50             |

### Links

- **Code**: [multimolecule.optmrl](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/optmrl)
- **Data**: 260,000 random 5'UTR reporters (pre-training) + 20,000 human 5'UTR reporters (fine-tuning)
- **Paper**: [Interpreting Deep Neural Networks for the Prediction of Translation Rates](https://doi.org/10.1101/2023.06.02.543405)
- **Developed by**: Frederick Korbel, Ekaterina Eroshok, Uwe Ohler
- **Model type**: 1D CNN for mean-ribosome-load regression from 5'UTR sequence
- **Original Repository**: [ohlerlab/mlcis](https://github.com/ohlerlab/mlcis)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Mean Ribosome Load Prediction

You can use this model directly to predict the mean ribosome load of a 50-nucleotide 5'UTR window:

```python
>>> from multimolecule import RnaTokenizer, OptMrlForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/optmrl")
>>> model = OptMrlForSequencePrediction.from_pretrained("multimolecule/optmrl")
>>> sequence = "ACGU" * 12 + "AC"  # 50 nt
>>> input = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: 50 nt fixed 5'UTR window taken immediately upstream of the coding sequence
- **Padding**: shorter sequences are right-padded with zeros to 50 nt; longer sequences are truncated to the first 50 nt
- **Alphabet**: `ACGU` only; unknown / `N` tokens contribute zero one-hot signal
- **Special tokens**: do not add (`add_special_tokens=False`)
- **Output**: single scalar mean-ribosome-load (MRL) score per window

## Training Details

OptMRL was first pre-trained on a large random-5'UTR reporter library and then fine-tuned on a smaller library of endogenous human 5'UTRs.

### Training Data

- **Pre-training**: ~260,000 random 5'UTR reporters paired with polysome-profiling MRL measurements.
- **Fine-tuning**: ~20,000 endogenous human 5'UTR reporters paired with polysome-profiling MRL measurements.

Each reporter contributes a 50-nucleotide 5'UTR window immediately upstream of the coding sequence and a scalar MRL label.

Note [`RnaTokenizer`][multimolecule.RnaTokenizer] will convert "T"s to "U"s for you, you may disable this behaviour by passing `replace_T_with_U=False`.

### Training Procedure

#### Pre-training

The model was first pre-trained as a regression task to predict the measured MRL of each random 5'UTR reporter, then fine-tuned end-to-end on the human-5'UTR reporters using the same regression objective. The published model is the fine-tuned model.

## Citation

```bibtex
@article{korbel2023interpreting,
  author    = {Korbel, Frederick and Eroshok, Ekaterina and Ohler, Uwe},
  title     = {Interpreting Deep Neural Networks for the Prediction of Translation Rates},
  journal   = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  year      = {2023},
  doi       = {10.1101/2023.06.02.543405}
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

Please contact the authors of the [OptMRL paper](https://doi.org/10.1101/2023.06.02.543405) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
