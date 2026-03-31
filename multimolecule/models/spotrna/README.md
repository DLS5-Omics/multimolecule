---
language: rna
tags:
  - Biology
  - RNA
  - Secondary Structure
license: mpl-2.0
datasets:
  - multimolecule/bprna
library_name: multimolecule
pipeline_tag: other
pipeline: rna-secondary-structure
---

# SPOT-RNA

Pre-trained model for RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning.


## Disclaimer

This is an UNOFFICIAL implementation of the [RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning](https://doi.org/10.1038/s41467-019-13395-9) by Jaswinder Singh, et al.

The OFFICIAL repository of SPOT-RNA is at [jaswindersingh2/SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing SPOT-RNA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

SPOT-RNA is a 2D convolutional neural network for predicting RNA secondary structure (base-pair contact maps) from single RNA sequences. It predicts both canonical (Watson-Crick and wobble) and non-canonical base pairs, including pseudoknots and other tertiary interactions.

The model uses:

- **2D residual convolution blocks** with LayerNorm, dropout, and checkpoint-matched ReLU/ELU activations
- **Outer concatenation** of canonical nucleotide features to create an L x L x 8 pairwise feature matrix
- **Optional architecture-specific paths** with either a shared 2D-BLSTM block or dilated convolutions
- **Transfer learning** from a large bpRNA dataset (~13,400 RNAs) to a small PDB dataset (~120 high-resolution structures)
- **The final SPOT-RNA predictor** as published by the paper and original codebase

MultiMolecule provides SPOT-RNA as a single checkpoint, [`multimolecule/spotrna`](https://huggingface.co/multimolecule/spotrna).

### Model Specification

| Num Parameters (M) | FLOPs (G) | MACs (G) |
| ------------------ | --------- | -------- |
| 17.46              | 8642.10   | 4302.16  |

### Links

- **Code**: [multimolecule.spotrna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/spotrna)
- **Weights**: [multimolecule/spotrna](https://huggingface.co/multimolecule/spotrna)
- **Data**: [multimolecule/bprna-spot](https://huggingface.co/datasets/multimolecule/bprna-spot)
- **Paper**: [RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning](https://doi.org/10.1038/s41467-019-13395-9)
- **Developed by**: Jaswinder Singh, Jack Hanson, Kuldip Paliwal, Yaoqi Zhou
- **Original Repository**: [jaswindersingh2/SPOT-RNA](https://github.com/jaswindersingh2/SPOT-RNA)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RNA Secondary Structure Pipeline

You can use SPOT-RNA directly with the MultiMolecule secondary-structure pipeline:

```python
import multimolecule  # you must import multimolecule to register models
from transformers import pipeline

predictor = pipeline("rna-secondary-structure", model="multimolecule/spotrna")
output = predictor("GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCUCA")
```

#### PyTorch Inference

Here is how to use this model to predict RNA secondary structure in PyTorch:

```python
import torch
from multimolecule import RnaTokenizer, SpotRnaModel

tokenizer = RnaTokenizer.from_pretrained("multimolecule/spotrna")
model = SpotRnaModel.from_pretrained("multimolecule/spotrna")

sequence = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCUCA"
input = tokenizer(sequence, return_tensors="pt")

output = model(**input)
contact_map = output.contact_map  # (1, L, L) base-pair probability matrix
```

## Training Details

SPOT-RNA was trained using a two-stage transfer learning approach on RNA secondary structure prediction.

### Training Data

- initial training source: bpRNA-1m (Version 1.0) with 102,348 annotated RNAs.
- initial training filtering: CD-HIT-EST at 80% sequence identity, removal of RNAs with PDB structures, and maximum sequence length of 500 nucleotides.
- initial training corpus: 13,419 RNAs after preprocessing.
- initial training split: TR0 = 10,814, VL0 = 1,300, TS0 = 1,305.
- transfer-learning source: high-resolution PDB RNAs downloaded on March 2, 2019.
- transfer-learning filtering: resolution better than 3.5 A and CD-HIT-EST at 80% sequence identity.
- transfer-learning corpus: 226 nonredundant RNAs after preprocessing.
- transfer-learning split before homology filtering: TR1 = 120, VL1 = 30, TS1 = 76.
- additional TS1 filtering: CD-HIT-EST against the training data at 80% identity, followed by BLAST-N against TR0 and TR1 with e-value cutoff 10.
- final TS1 benchmark: 67 RNAs.
- additional evaluation set: TS2 = 39 NMR-solved RNAs selected from 641 candidates after CD-HIT-EST filtering at 80% identity and BLAST-N filtering against TR0, TR1, and TS1.
- use of TS2: post-training evaluation only.

### Training Procedure

#### Preprocessing

- input representation: one-hot `L x 4` matrix over `A/U/C/G`.
- missing-value handling: invalid or missing residues encoded as `-1` in the original TensorFlow implementation before one-hot conversion.
- pairwise features: outer concatenation from `L x 4` to `L x L x 8`.
- input normalization: standardization to zero mean and unit variance using training-set statistics.
- structure labels: extracted from PDB coordinates with DSSR.
- reference NMR model: model 1.
- pseudoknot and motif definitions: bpRNA definitions from the paper.
- adaptation in MultiMolecule: tokenizer vocabulary is `A/C/G/U/N`; `N` tokens are masked out before constructing canonical 4-base features.

#### Pre-training

The paper states that training was run on Nvidia GTX TITAN X GPUs.

- training split: TR0.
- validation split: VL0.
- optimizer: Adam.
- regularization: 25% dropout before convolution layers and 50% dropout in hidden fully connected layers.
- hyperparameter search over `N_A`: 16 to 32 residual blocks.
- hyperparameter search over `D_RES`: 32 to 72 convolution channels.
- hyperparameter search over `D_BL`: 128 to 256 2D-BLSTM hidden units per direction.
- hyperparameter search over `N_B`: 0 to 4 fully connected blocks.
- hyperparameter search over `D_FC`: 256 to 512 fully connected hidden units.
- model selection: five best models by VL0 performance.

#### Transfer Learning

The pretrained TR0 models were retrained on TR1 with the same architecture and optimization settings.

- initialization: start from the TR0-trained models.
- training split: TR1.
- validation split: VL1.
- frozen layers: none; all weights were updated.
- architecture and optimization settings: same as the TS0-trained models.
- model selection: five best models by VL1 performance.
- decision rule: a single probability threshold chosen to optimize validation performance.
- released model: transfer-learning ensemble, not the direct-training baseline.

## Citation

```bibtex
@article{singh2019rna,
  title     = "{RNA} secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning",
  author    = "Singh, Jaswinder and Hanson, Jack and Paliwal, Kuldip and Zhou, Yaoqi",
  journal   = "Nature Communications",
  doi       = "10.1038/s41467-019-13395-9",
  publisher = "Springer Science and Business Media LLC",
  url       = "https://doi.org/10.1038/s41467-019-13395-9",
  volume    =  10,
  number    =  1,
  pages     = "5407",
  month     =  nov,
  year      =  2019,
  copyright = "https://creativecommons.org/licenses/by/4.0",
  language  = "en"
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

Please contact the authors of the [SPOT-RNA paper](https://doi.org/10.1038/s41467-019-13395-9) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md) and the [CC-BY-NC-4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later AND CC-BY-NC-4.0
```
