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

# DeepMEL

Convolutional and recurrent neural network for predicting melanoma-specific accessible chromatin regions and chromatin topics directly from DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Cross-species analysis of enhancer logic using deep learning](https://doi.org/10.1101/gr.260844.120) by Liesbeth Minnoye, Ibrahim Ihsan Taskiran, et al.

The OFFICIAL repository of DeepMEL is at [aertslab/DeepMEL](https://github.com/aertslab/DeepMEL).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeepMEL did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeepMEL is a hybrid convolutional / recurrent neural network trained to predict 24 melanoma chromatin topics (a `4-MEL` melanocytic, a `7-MES` mesenchymal-like, and additional accessibility programs) directly from 500 bp DNA sequence. Each input sequence is processed by a shared encoder consisting of a 1D convolution, max pooling, a time-distributed dense projection, and a bidirectional LSTM, followed by a fully-connected layer. The same encoder is applied independently to the forward DNA strand and to its reverse complement; a final 24-way decoder produces a sigmoid probability per topic in each branch, and the two branches' probabilities are averaged into the model's prediction. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Conv Filters | Conv Kernel | BiLSTM Hidden | FC Hidden | Num Topics | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ------------ | ----------- | ------------- | --------- | ---------- | ------------------ | --------- | -------- | -------------- |
| 128          | 20          | 128           | 256       | 24         | 3.44               | 40.76     | 20.19    | 500            |

### Links

- **Code**: [multimolecule.deepmel](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deepmel)
- **Data**: Melanoma cell-line single-cell ATAC-seq topic models
- **Paper**: [Cross-species analysis of enhancer logic using deep learning](https://doi.org/10.1101/gr.260844.120)
- **Developed by**: Liesbeth Minnoye, Ibrahim Ihsan Taskiran, David Mauduit, Maurizio Fazio, Linde Van Aerschot, Gert Hulselmans, Valerie Christiaens, Samira Makhzami, Monika Seltenhammer, Panagiotis Karras, Aline Primot, Edouard Cadieu, Ellen van Rooijen, Jean-Christophe Marine, Giorgia Egidy, Ghanem-Elias Ghanem, Leonard Zon, Jasper Wouters, Stein Aerts
- **Model type**: 1D CNN + BiLSTM over 500 bp DNA with reverse-complement averaging for multi-task chromatin-topic prediction
- **Original Repository**: [aertslab/DeepMEL](https://github.com/aertslab/DeepMEL)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Chromatin Topic Prediction

You can use this model directly to predict the 24 melanoma chromatin-topic activities of a 500 bp DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, DeepMelForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepmel")
>>> model = DeepMelForSequencePrediction.from_pretrained("multimolecule/deepmel")
>>> sequence = "ACGT" * 125
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 24])
```

### Interface

- **Input length**: fixed 500 bp DNA window
- **Alphabet**: `ACGT` (one-hot encoded); the reverse complement is computed internally
- **Output**: 24 chromatin-topic logits (multi-label binary); `postprocess` returns the branch-averaged sigmoid probability per topic

## Training Details

DeepMEL was trained to predict cell-type-specific accessible chromatin topics derived from single-cell ATAC-seq of melanoma cell lines.

### Training Data

DeepMEL was trained on accessible genomic intervals derived from melanoma single-cell ATAC-seq experiments and modeled as 24 chromatin topics (including the `4-MEL` melanocytic-like and `7-MES` mesenchymal-like programs). Each training example is a 500 bp genomic interval labelled with a binary vector indicating which topics are active. Chromosome 2 was held out for validation and testing.

### Training Procedure

#### Pre-training

The model was trained to minimize a multi-label binary cross-entropy loss between the branch-averaged sigmoid probabilities and the observed topic-activity labels.

- Optimizer: Adam
- Loss: Multi-label binary cross-entropy
- Regularization: Dropout (`0.2` after pooling, `0.1` LSTM input and recurrent dropout, `0.2` after the BiLSTM, `0.4` before the prediction head)

## Citation

```bibtex
@article{minnoye2020deepmel,
  author    = {Minnoye, Liesbeth and Taskiran, Ibrahim Ihsan and Mauduit, David and Fazio, Maurizio and Van Aerschot, Linde and Hulselmans, Gert and Christiaens, Valerie and Makhzami, Samira and Seltenhammer, Monika and Karras, Panagiotis and Primot, Aline and Cadieu, Edouard and van Rooijen, Ellen and Marine, Jean-Christophe and Egidy, Giorgia and Ghanem, Ghanem-Elias and Zon, Leonard and Wouters, Jasper and Aerts, Stein},
  title     = {Cross-species analysis of enhancer logic using deep learning},
  journal   = {Genome Research},
  volume    = 30,
  number    = 12,
  pages     = {1815--1834},
  year      = 2020,
  publisher = {Cold Spring Harbor Laboratory Press},
  doi       = {10.1101/gr.260844.120}
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

Please contact the authors of the [DeepMEL paper](https://doi.org/10.1101/gr.260844.120) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
