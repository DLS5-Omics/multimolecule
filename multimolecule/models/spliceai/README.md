---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
---

# SpliceAI

Convolutional neural network for predicting mRNA splicing from pre-mRNA sequences.

## Disclaimer

This is an UNOFFICIAL implementation of the [Predicting Splicing from Primary Sequence with Deep Learning](https://doi.org/10.1016/j.cell.2018.12.015) by Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou and Jeremy F. McRae.

The OFFICIAL repository of SpliceAI is at [Illumina/SpliceAI](https://github.com/Illumina/SpliceAI).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing SpliceAI did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

SpliceAI is a convolutional neural network (CNN) trained to predict mRNA splicing site locations (acceptor and donor) from primary pre-mRNA sequences. The model was trained in a supervised manner using annotated splice junctions from human reference transcripts. It processes input RNA sequences and, for each nucleotide, predicts the probability of it being a splice acceptor, a splice donor, or neither. This allows for the identification of canonical splice sites and the prediction of cryptic splice sites potentially activated or inactivated by sequence variants. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ---------- | ----------- | ------------------ | --------- | -------- |
| 16         | 32          | 3.48               | 70.39     | 35.11    |

### Links

- **Code**: [multimolecule.spliceai](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/spliceai)
- **Weights**: [multimolecule/spliceai](https://huggingface.co/multimolecule/spliceai)
- **Paper**: [Predicting Splicing from Primary Sequence with Deep Learning](https://doi.org/10.1016/j.cell.2018.12.015)
- **Developed by**: Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae, Siavash Fazel Darbandi, David Knowles, Yang I. Li, Jack A. Kosmicki, Juan Arbelaez, Wenwu Cui, Grace B. Schwartz, Eric D. Chow, Efstathios Kanterakis, Hong Gao, Amirali Kia, Serafim Batzoglou, Stephan J. Sanders, Kyle Kai-How Farh
- **Original Repository**: [Illumina/SpliceAI](https://github.com/Illumina/SpliceAI)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

You can use this model directly to predict the splicing sites of an RNA sequence:

```python
>>> from multimolecule import RnaTokenizer, SpliceAiModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/spliceai")
>>> model = SpliceAiModel.from_pretrained("multimolecule/spliceai")
>>> output = model(tokenizer("agcagucauuauggcgaa", return_tensors="pt")["input_ids"])

>>> output.keys()
odict_keys(['logits'])

>>> output.logits.squeeze()
tensor([[ 8.5123, -4.9607, -7.6787],
        [ 8.6559, -4.4936, -8.6357],
        [ 5.8514, -1.9375, -6.8030],
        [ 7.3739, -5.3444, -5.2559],
        [ 8.6336, -5.3187, -7.5741],
        [ 6.1947, -1.5497, -7.6286],
        [ 9.0482, -6.1002, -7.1229],
        [ 7.9647, -5.6973, -6.5327],
        [ 8.8795, -6.3714, -7.0204],
        [ 7.9459, -5.4744, -6.0865],
        [ 8.4272, -5.2556, -7.9027],
        [ 7.7523, -5.8517, -6.9109],
        [ 7.3027, -4.6946, -5.9420],
        [ 8.1432, -4.3085, -7.7892],
        [ 7.9060, -4.9454, -7.0091],
        [ 8.9770, -5.3971, -7.3313],
        [ 8.4292, -5.7455, -6.7811],
        [ 8.2709, -6.1388, -6.6784]], grad_fn=<SqueezeBackward0>)
```

## Training Details

SpliceAI was trained to predict the location of splice donor and acceptor sites from primary DNA sequence.

### Training Data

The SpliceAI model was trained on human reference transcripts obtained from [GENCODE](https://multimolecule.danling.org/datasets/gencode) (release 24, GRCh38).
This dataset comprises both protein-coding and non-protein-coding transcripts.

For training, a sequence window of 10,000 base pairs (bp) was used for each nucleotide whose splicing status was to be predicted, including 5,000 bp upstream and 5,000 bp downstream.
Sequences near transcript ends were padded with 'N' (unknown nucleotide) characters to maintain a consistent input length.
Annotated splice donor and acceptor sites from GENCODE served as positive labels for their respective classes.
All other intronic and exonic positions within these transcripts were considered negative (non-splice site) labels.

The data was partitioned by chromosome:
Chromosomes 1-19, X, and Y were designated for the training set.
Chromosome 20 was reserved as a test set.
A validation set, comprising 5% of transcripts from each training chromosome, was used for model selection and to monitor for overfitting.
Positions within 50 bp of a masked interval (an interval of >10 'N's) or within 50 bp of a transcript end were excluded from the training and validation datasets.

To address class imbalance, training examples were weighted such that the total loss contribution from positive examples (acceptor or donor sites) equaled that from negative examples (non-splice sites).
Within positive examples, acceptor and donor sites were weighted equally.

### Training Procedure

#### Pre-training

The model was trained to minimize a cross-entropy loss, comparing its predicted splice site probabilities against the ground truth labels from GENCODE.

- Batch Size:64
- Epochs: 4
- Optimizer: Adam
- Learning rate: 1e-3
- Learning rate scheduler: Exponential
- Minimum learning rate: 1e-5

## Citation

**BibTeX**:

```bibtex
@article{jaganathan2019the,
  abstract  = {The splicing of pre-mRNAs into mature transcripts is remarkable for its precision, but the mechanisms by which the cellular machinery achieves such specificity are incompletely understood. Here, we describe a deep neural network that accurately predicts splice junctions from an arbitrary pre-mRNA transcript sequence, enabling precise prediction of noncoding genetic variants that cause cryptic splicing. Synonymous and intronic mutations with predicted splice-altering consequence validate at a high rate on RNA-seq and are strongly deleterious in the human population. De novo mutations with predicted splice-altering consequence are significantly enriched in patients with autism and intellectual disability compared to healthy controls and validate against RNA-seq in 21 out of 28 of these patients. We estimate that 9\%-11\% of pathogenic mutations in patients with rare genetic disorders are caused by this previously underappreciated class of disease variation.},
  author    = {Jaganathan, Kishore and Kyriazopoulou Panagiotopoulou, Sofia and McRae, Jeremy F and Darbandi, Siavash Fazel and Knowles, David and Li, Yang I and Kosmicki, Jack A and Arbelaez, Juan and Cui, Wenwu and Schwartz, Grace B and Chow, Eric D and Kanterakis, Efstathios and Gao, Hong and Kia, Amirali and Batzoglou, Serafim and Sanders, Stephan J and Farh, Kyle Kai-How},
  copyright = {http://www.elsevier.com/open-access/userlicense/1.0/},
  journal   = {Cell},
  keywords  = {artificial intelligence; deep learning; genetics; splicing},
  language  = {en},
  month     = jan,
  number    = 3,
  pages     = {535--548.e24},
  publisher = {Elsevier BV},
  title     = {Predicting splicing from primary sequence with deep learning},
  volume    = 176,
  year      = 2019
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [SpliceAI paper](https://doi.org/10.1016/j.cell.2018.12.015) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) and the [CC-BY-NC-4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later AND CC-BY-NC-4.0
```
