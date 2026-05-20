---
language: dna
tags:
  - Biology
  - DNA
  - RNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: sequence-binding-prediction
---

# DeepBind

Single-layer convolutional model for predicting the sequence specificities of DNA- and RNA-binding proteins.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning](https://doi.org/10.1038/nbt.3300) by **Babak Alipanahi, Andrew Delong et al.**

The OFFICIAL distribution of DeepBind is at [jisraeli/DeepBind](https://github.com/jisraeli/DeepBind).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeepBind did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeepBind is a single-layer convolutional neural network that predicts how strongly a DNA- or RNA-binding protein binds to a given nucleotide sequence. The model consumes a variable-length one-hot encoded sequence and applies one convolutional motif-detector layer (Conv1D + ReLU) followed by a global pooling stage (max, or concatenated max + average) and one or two fully-connected layers that project the pooled feature vector to a single binding score. The published DeepBind tool ships 538 per-protein checkpoints trained on PBM, RNAcompete, HT-SELEX and ChIP-seq data; every checkpoint shares the same architecture and differs only in training data and per-protein filter / hidden widths. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

Each per-protein DeepBind checkpoint lives in its own Hub repository under `multimolecule/deepbind-<protein>`, all sharing the same `DeepBindModel` class. A small set of representative repositories:

| Hub repository                  | Molecule | Assay      | Protein |
| ------------------------------- | -------- | ---------- | ------- |
| `multimolecule/deepbind-ctcf`   | DNA      | ChIP-seq   | CTCF    |
| `multimolecule/deepbind-max`    | DNA      | ChIP-seq   | MAX     |
| `multimolecule/deepbind-hnrnpk` | RNA      | RNAcompete | HNRNPK  |

### Model Specification

| Num Filters | Kernel Size | Hidden Size | Num Parameters (K) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ----------- | ----------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 16          | 24          | 32          | 2.62               | 0.32      | 0.16     | variable       |

### Links

- **Code**: [multimolecule.deepbind](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deepbind)
- **Weights**: [multimolecule/deepbind](https://huggingface.co/multimolecule/deepbind)
- **Data**: Protein binding microarrays (Mukherjee et al., 2004), RNAcompete (Ray et al., 2009), ChIP-seq (Kharchenko et al., 2008) and HT-SELEX (Jolma et al., 2010)
- **Paper**: [Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning](https://doi.org/10.1038/nbt.3300)
- **Developed by**: Babak Alipanahi, Andrew Delong, Matthew T. Weirauch, Brendan J. Frey
- **Model type**: Single-layer 1D CNN with global max / max+avg pooling and 1-2 fully-connected layers for sequence-level protein-binding prediction
- **Original Repository**: [jisraeli/DeepBind](https://github.com/jisraeli/DeepBind)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Protein-Binding Prediction

You can use this model directly to predict the binding score of a DNA- or RNA-binding protein for a given sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, DeepBindForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/deepbind-ctcf")
>>> model = DeepBindForSequencePrediction.from_pretrained("multimolecule/deepbind-ctcf")
>>> input = tokenizer("ACGT" * 25 + "A", return_tensors="pt", add_special_tokens=False)
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: variable; must be at least `kernel_size` (24) nucleotides
- **Alphabet**: `ACGT` for transcription-factor (DNA) checkpoints; `ACGU` for RNA-binding-protein checkpoints (4 channels in either case)
- **Special tokens**: do not add (`add_special_tokens=False`); DeepBind scores raw nucleotide windows
- **`inputs_embeds`**: supported; shape `(batch_size, sequence_length, vocab_size)`
- **Output**: single scalar TF / RBP binding score per sequence

## Training Details

DeepBind is trained per protein to predict the binding score of a single TF or RBP from sequence input.

### Training Data

DeepBind was trained on a diverse compendium of in vitro and in vivo protein-binding assays: protein binding microarrays (Mukherjee et al., 2004), RNAcompete (Ray et al., 2009), HT-SELEX (Jolma et al., 2010) and ChIP-seq (Kharchenko et al., 2008). Each per-protein checkpoint is trained on a single assay covering a single TF or RBP.

### Training Procedure

#### Pre-training

The model was trained per protein to minimise a calibration loss between the predicted binding score and the experimentally measured probe intensity (or per-peak binding indicator for ChIP-seq).

- Framework: original Lua-Torch (Alipanahi et al., 2015); Kipoi redistribution uses TensorFlow / Keras
- Loss: regression loss against the per-probe / per-peak measurement
- Optimizer: stochastic gradient descent with momentum and weight decay
- Regularization: dropout on the hidden fully-connected layer

## Citation

```bibtex
@article{alipanahi2015predicting,
  author    = {Alipanahi, Babak and Delong, Andrew and Weirauch, Matthew T. and Frey, Brendan J.},
  title     = {Predicting the sequence specificities of {DNA-} and {RNA-}binding proteins by deep learning},
  journal   = {Nature Biotechnology},
  volume    = {33},
  number    = {8},
  pages     = {831--838},
  year      = {2015},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/nbt.3300}
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

Please contact the authors of the [DeepBind paper](https://doi.org/10.1038/nbt.3300) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
