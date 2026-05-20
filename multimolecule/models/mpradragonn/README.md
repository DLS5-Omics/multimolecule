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

# MPRA-DragoNN

Convolutional neural network for predicting Sharpr-MPRA reporter activity directly from 145 bp DNA sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays](https://doi.org/10.1371/journal.pone.0218073) by Rajiv Movva, et al.

The OFFICIAL repository of MPRA-DragoNN is at [kundajelab/MPRA-DragoNN](https://github.com/kundajelab/MPRA-DragoNN).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing MPRA-DragoNN did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

MPRA-DragoNN is a convolutional neural network (CNN) trained to quantitatively predict Sharpr-MPRA reporter activity from 145 bp DNA sequences. The released `ConvModel` consists of three convolutional blocks (Conv1D + ReLU + BatchNorm + Dropout, 120 filters of width 5 with valid padding) followed by a flatten and a single fully-connected layer that emits 12 task outputs. Each task corresponds to a (cell line, reporter promoter, replicate) combination from the Sharpr-MPRA experiment: the K562 and HepG2 cell lines, each measured with both a minimal promoter (minP) and the strong SV40 promoter (SV40p), with two individual replicates plus a pooled average per condition. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Conv Layers | Num FC Layers | Hidden Size | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| --------------- | ------------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 3               | 1             | 15960       | 0.34               | 40.40     | 20.05    | 145            |

### Links

- **Code**: [multimolecule.mpradragonn](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/mpradragonn)
- **Data**: [Sharpr-MPRA dataset](http://mitra.stanford.edu/kundaje/projects/mpra/) (Ernst et al. 2016)
- **Paper**: [Deciphering regulatory DNA sequences and noncoding genetic variants using neural network models of massively parallel reporter assays](https://doi.org/10.1371/journal.pone.0218073)
- **Developed by**: Rajiv Movva, Peyton Greenside, Georgi K. Marinov, Surag Nair, Avanti Shrikumar, Anshul Kundaje
- **Model type**: Three-layer 1D CNN over 145 bp DNA for multi-task Sharpr-MPRA activity regression
- **Original Repository**: [kundajelab/MPRA-DragoNN](https://github.com/kundajelab/MPRA-DragoNN)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### MPRA Activity Prediction

You can use this model directly to predict the Sharpr-MPRA activity of a 145 bp DNA sequence:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, MpraDragoNnForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/mpradragonn")
>>> model = MpraDragoNnForSequencePrediction.from_pretrained("multimolecule/mpradragonn")
>>> sequence = "ACGT" * 36 + "A"
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 12])
```

### Interface

- **Input length**: fixed 145 bp DNA window
- **Output**: 12 MPRA activity scalars in the order `k562_minp_{rep1, rep2, avg}`, `k562_sv40p_{rep1, rep2, avg}`, `hepg2_minp_{rep1, rep2, avg}`, `hepg2_sv40p_{rep1, rep2, avg}` (z-scored log2 RNA/DNA ratios)

## Training Details

MPRA-DragoNN was trained to predict quantitative Sharpr-MPRA reporter activity from DNA sequence.

### Training Data

MPRA-DragoNN was trained on the [Sharpr-MPRA dataset](https://www.nature.com/articles/nbt.3678) (Ernst et al. 2016, GEO accession GSE71279) which assays ~487K 145 bp candidate regulatory elements in K562 and HepG2 cell lines under two reporter promoters (a minimal promoter and the strong SV40 promoter) and provides two replicates plus a pooled count per condition (12 tasks total).

Raw counts were preprocessed by (1) computing `log2((RNA + 1) / (DNA + 1))` per task, (2) column-wise z-score normalisation per task, and (3) augmenting with the reverse complement of every sequence. Chromosomes were split with chr8 held out as validation, chr18 held out as test, and all remaining chromosomes used for training (~900K training, ~30K validation, ~20K test sequences after the reverse-complement augmentation).

### Training Procedure

#### Pre-training

The model was trained to minimise a task-wise mean-squared-error loss between predicted and measured MPRA activities and evaluated with Spearman correlation per task.

- Optimizer: Adam
- Loss: Mean Squared Error (task-wise, equally weighted)
- Regularization: Batch normalization and dropout (p=0.1) after every convolutional block
- Validation: chr8 sequences; Test: chr18 sequences

## Citation

```bibtex
@article{movva2019mpradragonn,
  author    = {Movva, Rajiv and Greenside, Peyton and Marinov, Georgi K. and Nair, Surag and Shrikumar, Avanti and Kundaje, Anshul},
  title     = {Deciphering regulatory {DNA} sequences and noncoding genetic variants using neural network models of massively parallel reporter assays},
  journal   = {PLoS ONE},
  volume    = 14,
  number    = 6,
  pages     = {e0218073},
  year      = 2019,
  publisher = {Public Library of Science},
  doi       = {10.1371/journal.pone.0218073}
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

Please contact the authors of the [MPRA-DragoNN paper](https://doi.org/10.1371/journal.pone.0218073) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
