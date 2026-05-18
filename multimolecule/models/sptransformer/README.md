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

# SpTransformer

Transformer network for predicting tissue-specific splicing from pre-mRNA sequences.

## Disclaimer

This is an UNOFFICIAL implementation of [SpliceTransformer predicts tissue-specific splicing linked to human diseases](https://doi.org/10.1038/s41467-024-53088-6) by Ningyuan You, et al.

The OFFICIAL repository of SpliceTransformer (SpTransformer) is at [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing SpTransformer did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

SpTransformer (SpliceTransformer) is a deep neural network that predicts tissue-specific splicing from primary pre-mRNA sequence.
It combines two pretrained SpliceAI-style dilated-residual convolutional feature extractors with a trainable input-projection path; the concatenated features are processed by a Sinkhorn transformer attention block with axial positional embeddings.
For each position the network predicts a 3-channel splice-site score (no-splice / acceptor / donor) and a per-position splice-site usage score across 15 human tissues.
The model uses a fixed flanking context of 4,000 nucleotides on each side of every predicted position.
SpTransformer is typically used to estimate the effect of genetic variants on tissue-specific splicing by scoring reference and alternate sequences and taking the difference.
Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Max Seq Len | Num Parameters (M) | FLOPs (G) | MACs (G) | Context |
| ---------- | ----------- | --------- | ----------------- | ----------- | ------------------ | --------- | -------- | ------- |
| 8          | 256         | 8         | 1024              | 8192        | 17.07              | 290.72    | 144.65   | 4000    |

### Links

- **Code**: [multimolecule.sptransformer](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/sptransformer)
- **Data**: GTEx human RNA-seq across 15 tissues with gene annotations from GENCODE and multi-species sequence data
- **Paper**: [SpliceTransformer predicts tissue-specific splicing linked to human diseases](https://doi.org/10.1038/s41467-024-53088-6)
- **Developed by**: Ningyuan You, Chang Liu, Yuxin Gu, Rong Wang, Hanying Jia, Tianyun Zhang, Song Jiang, Jinsong Shi, Ming Chen, Min-Xin Guan, Siqi Sun, Shanshan Pei, Zhihong Liu, Ning Shen
- **Model type**: Transformer encoder with windowed-local and Sinkhorn sorted-bucket attention for tissue-specific splicing prediction
- **Original Repository**: [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RNA Splicing Site Prediction

You can use this model directly to predict per-nucleotide tissue-specific splicing of a pre-mRNA sequence:

```python
>>> from multimolecule import RnaTokenizer, SpTransformerModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/sptransformer")
>>> model = SpTransformerModel.from_pretrained("multimolecule/sptransformer")
>>> output = model(tokenizer("AGCAGUCAUUAUGGCGAA", return_tensors="pt")["input_ids"])

>>> output.keys()
odict_keys(['last_hidden_state', 'logits'])
```

The `logits` tensor reproduces the original SpTransformer output: a 3-channel splice-site score (no-splice / acceptor / donor) and a per-tissue (15 tissues) splice-site usage score for each position.

### Downstream Use

#### Token Prediction

You can fine-tune SpTransformer for per-nucleotide tissue-specific splicing regression with [`SpTransformerForTokenPrediction`][multimolecule.models.SpTransformerForTokenPrediction], which adds a shared token prediction head on top of the backbone.

### Interface

- **Input length**: variable pre-mRNA sequence
- **Flanking context**: fixed 4,000 nt on each side of every predicted position
- **Padding**: ends padded with `N`
- **Output**: per-position 3-channel splice-site score (`no-splice` / `acceptor` / `donor`) + per-tissue (15 tissues) splice-site usage score

## Training Details

SpTransformer was trained to predict tissue-specific splicing from primary pre-mRNA sequence.

### Training Data

SpTransformer was trained on splicing measurements derived from RNA-seq data across 15 human tissues, using gene annotations from [GENCODE](https://multimolecule.danling.org/datasets/gencode), together with multi-species sequence data.
The two convolutional feature extractors were pre-trained as SpliceAI-style splice-site predictors and remain trainable submodules for downstream fine-tuning.
For each predicted nucleotide, a sequence window centered on that nucleotide was used, with the flanking context padded with `N` (unknown nucleotide) when near transcript ends.

### Training Procedure

#### Pre-training

The model was trained to minimize a combination of cross-entropy loss over splice-site classification and a regression loss over per-tissue splice-site usage, comparing predictions against measurements derived from RNA-seq.

## Citation

```bibtex
@article{You2024,
  author    = {You, Ningyuan and Liu, Chang and Gu, Yuxin and Wang, Rong and Jia, Hanying and Zhang, Tianyun and Jiang, Song and Shi, Jinsong and Chen, Ming and Guan, Min-Xin and Sun, Siqi and Pei, Shanshan and Liu, Zhihong and Shen, Ning},
  title     = {{SpliceTransformer predicts tissue-specific splicing linked to human diseases}},
  journal   = {Nature Communications},
  year      = {2024},
  volume    = {15},
  number    = {1},
  pages     = {9129},
  month     = {oct},
  doi       = {10.1038/s41467-024-53088-6},
  issn      = {2041-1723},
  url       = {https://doi.org/10.1038/s41467-024-53088-6}
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

Please contact the authors of the [SpliceTransformer paper](https://doi.org/10.1038/s41467-024-53088-6) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
