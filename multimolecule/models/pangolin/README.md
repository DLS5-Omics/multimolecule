---
language: rna
tags:
  - Biology
  - RNA
  - Splicing
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
pipeline_tag: other
pipeline: splice-site
---

# Pangolin

Convolutional neural network for predicting tissue-specific splice site strength from pre-mRNA sequences.

## Disclaimer

This is an UNOFFICIAL implementation of [Predicting RNA splicing from DNA sequence using Pangolin](https://doi.org/10.1186/s13059-022-02664-4) by Tony Zeng, et al.

The OFFICIAL repository of Pangolin is at [tkzeng/Pangolin](https://github.com/tkzeng/Pangolin).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing Pangolin did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

Pangolin is a deep convolutional neural network (CNN) that predicts splice site strength from primary pre-mRNA sequence.
It extends the dilated-residual SpliceAI architecture to predict tissue-specific splice site usage, and is trained on splicing measurements derived from RNA-seq data across multiple tissues.
The network processes a one-hot encoded nucleotide sequence and, for each position, predicts a splice-site score and a splice-site usage score per tissue.
Pangolin is typically used to estimate the effect of genetic variants on splicing by scoring reference and alternate sequences and taking the difference.
Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ---------- | ----------- | ------------------ | --------- | -------- |
| 16         | 32          | 8.36               | 168.85    | 84.04    |

### Links

- **Code**: [multimolecule.pangolin](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/pangolin)
- **Data**: Cross-species RNA-seq splice-site usage from human, rhesus, rat, and mouse tissues
- **Paper**: [Predicting RNA splicing from DNA sequence using Pangolin](https://doi.org/10.1186/s13059-022-02664-4)
- **Developed by**: Tony Zeng, Yang I. Li
- **Model type**: Dilated residual 1D CNN ensemble for per-nucleotide multi-tissue splice-site usage prediction
- **Original Repository**: [tkzeng/Pangolin](https://github.com/tkzeng/Pangolin)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RNA Splicing Site Prediction

You can use this model directly to predict per-nucleotide tissue-specific splice-site score and usage channels for a pre-mRNA sequence:

```python
>>> from multimolecule import RnaTokenizer, PangolinModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/pangolin")
>>> model = PangolinModel.from_pretrained("multimolecule/pangolin")
>>> output = model(tokenizer("AGCAGUCAUUAUGGCGAA", return_tensors="pt")["input_ids"])

>>> output.keys()
odict_keys(['last_hidden_state', 'probabilities'])
```

The `probabilities` tensor reproduces the original Pangolin output: for each of the four tissues, two splice-site score channels (softmax) and one splice-site usage channel (sigmoid).

### Downstream Use

#### Token Prediction

You can fine-tune Pangolin for per-nucleotide splice site strength regression with [`PangolinForTokenPrediction`][multimolecule.models.PangolinForTokenPrediction], which adds a shared token prediction head on top of the backbone.

### Interface

- **Input length**: variable pre-mRNA sequence
- **Padding**: flanking context padded with `N` near transcript ends
- **Output**: per-position tissue-specific channels — for each of 4 tissues, 2 splice-site score channels + 1 splice-site usage channel

## Training Details

Pangolin was trained to predict tissue-specific splice site usage from primary pre-mRNA sequence.

### Training Data

Pangolin was trained on splice site usage derived from RNA-seq data in heart, liver, brain, and testis tissues from human and three other species, using gene annotations from [GENCODE](https://multimolecule.danling.org/datasets/gencode).
For each nucleotide whose splicing status was predicted, a sequence window centered on that nucleotide was used, with the flanking context padded with `N` (unknown nucleotide) when near transcript ends.

### Training Procedure

#### Pre-training

The model was trained to minimize a combination of cross-entropy loss over splice-site classification and a regression loss over splice-site usage, comparing predictions against measurements derived from RNA-seq.

- Optimizer: AdamW
- Learning rate scheduler: Step decay

## Citation

```bibtex
@article{zeng2022predicting,
  author    = {Zeng, Tony and Li, Yang I.},
  title     = {Predicting RNA splicing from DNA sequence using Pangolin},
  journal   = {Genome Biology},
  volume    = {23},
  number    = {1},
  pages     = {103},
  year      = {2022},
  doi       = {10.1186/s13059-022-02664-4},
  publisher = {BioMed Central}
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

Please contact the authors of the [Pangolin paper](https://doi.org/10.1186/s13059-022-02664-4) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
