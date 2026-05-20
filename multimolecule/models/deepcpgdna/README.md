---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
---

# DeepCpG-DNA

DNA-only convolutional neural network from DeepCpG for predicting per-cell single-cell DNA methylation states from a CpG-centered sequence window.

## Disclaimer

This is an UNOFFICIAL implementation of [DeepCpG: accurate prediction of single-cell DNA methylation states using deep learning](https://doi.org/10.1186/s13059-017-1189-z) by Christof Angermueller, et al.

The OFFICIAL repository of DeepCpG is at [cangermueller/deepcpg](https://github.com/cangermueller/deepcpg).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeepCpG-DNA did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeepCpG-DNA is the DNA submodule of the DeepCpG joint model. It is a 1D convolutional neural network that predicts the per-cell methylation state of a CpG site from a fixed-length 1001 bp DNA window centered on the site. The model consumes a one-hot encoded sequence and applies `valid`-padded convolutional blocks (Conv1D + ReLU + MaxPool) followed by a dense bottleneck and one binary classification head per single cell in the training dataset. Please refer to the [Training Details](#training-details) section for more information on the training process.

The full DeepCpG model combines this DNA submodule with a recurrent CpG-context submodule and a joint head; this model card covers the DNA submodule only.

### Variants

The DeepCpG-DNA module is trained per single-cell dataset, so each variant predicts a different number of output cells.

| Dataset                   | Architecture | Cells | Hub repository                                                                                          |
| ------------------------- | ------------ | ----- | ------------------------------------------------------------------------------------------------------- |
| Smallwood 2014 serum mESC | CnnL2h128    | 18    | [`deepcpgdna-smallwood2014-serum`](https://huggingface.co/multimolecule/deepcpgdna-smallwood2014-serum) |
| Smallwood 2014 2i mESC    | CnnL3h128    | 12    | [`deepcpgdna-smallwood2014-2i`](https://huggingface.co/multimolecule/deepcpgdna-smallwood2014-2i)       |
| Hou 2016 HCC              | CnnL2h128    | 25    | [`deepcpgdna-hou2016-hcc`](https://huggingface.co/multimolecule/deepcpgdna-hou2016-hcc)                 |
| Hou 2016 HepG2            | CnnL3h128    | 6     | [`deepcpgdna-hou2016-hepg2`](https://huggingface.co/multimolecule/deepcpgdna-hou2016-hepg2)             |
| Hou 2016 mESC             | CnnL2h128    | 6     | [`deepcpgdna-hou2016-mesc`](https://huggingface.co/multimolecule/deepcpgdna-hou2016-mesc)               |

### Model Specification

<table>
<thead>
  <tr>
    <th>Architecture</th>
    <th>Num Conv Layers</th>
    <th>Hidden Size</th>
    <th>Num Cells</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (M)</th>
    <th>MACs (M)</th>
    <th>Max Num Tokens</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CnnL2h128</td>
    <td>2</td>
    <td rowspan="2">128</td>
    <td>18</td>
    <td>4.11</td>
    <td>70.63</td>
    <td>35.06</td>
    <td rowspan="2">1001</td>
  </tr>
  <tr>
    <td>CnnL3h128</td>
    <td>3</td>
    <td>12</td>
    <td>4.43</td>
    <td>165.02</td>
    <td>82.18</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.deepcpgdna](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deepcpgdna)
- **Data**: scBS-seq (Smallwood 2014) and scRRBS-seq (Hou 2016) single-cell bisulfite sequencing datasets
- **Paper**: [DeepCpG: accurate prediction of single-cell DNA methylation states using deep learning](https://doi.org/10.1186/s13059-017-1189-z)
- **Developed by**: Christof Angermueller, Heather J. Lee, Wolf Reik, Oliver Stegle
- **Model type**: Two- or three-layer 1D CNN over a 1001 bp CpG-centered DNA window for per-cell binary methylation prediction
- **Original Repository**: [cangermueller/deepcpg](https://github.com/cangermueller/deepcpg)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Single-Cell Methylation Prediction

You can use this model directly to predict the per-cell methylation state of a 1001 bp DNA window centered on a CpG site:

```python
>>> from multimolecule import DnaTokenizer, DeepCpgDnaForSequencePrediction

>>> model_id = "multimolecule/deepcpgdna-smallwood2014-serum"
>>> tokenizer = DnaTokenizer.from_pretrained(model_id)
>>> model = DeepCpgDnaForSequencePrediction.from_pretrained(model_id)
>>> input = tokenizer("ACGT" * 250 + "A", return_tensors="pt")
>>> output = model(**input)

>>> output.logits.shape
torch.Size([1, 18])
```

Each logit is a per-cell methylation score for one of the single cells in the chosen training dataset; apply a sigmoid to obtain methylation probabilities.

### Interface

- **Input length**: fixed 1001 bp DNA window centered on a CpG site
- **Padding**: not supported; pad or crop genomic windows so they match `sequence_length` exactly
- **Alphabet**: DNA (`A`, `C`, `G`, `T`); `N` is encoded as an all-zero channel
- **Output**: per-cell methylation logits; the number of cells is dataset-specific (see Variants table)

## Training Details

DeepCpG-DNA was trained to predict the per-cell methylation state of CpG sites from their flanking DNA context.

### Training Data

DeepCpG-DNA was trained on single-cell bisulfite sequencing datasets:

- **Smallwood 2014**: scBS-seq profiles of mouse embryonic stem cells, with 18 serum and 12 2i mESCs (excluding two serum cells whose methylation pattern deviated strongly from the remainder).
- **Hou 2016**: scRRBS-seq profiles of 25 human hepatocellular carcinoma (HCC) cells, 6 human heptoplastoma-derived (HepG2) cells, and 6 mESCs, restricted to CpG sites covered by at least four reads.

Each training example is a 1001 bp DNA window centered on a CpG site, with a per-cell binary methylation label (methylated, unmethylated, or missing). Chromosomes were split into training, validation, and test sets to avoid sequence leakage.

### Training Procedure

#### Pre-training

The model was trained to minimize a per-cell binary cross-entropy loss, comparing its predicted per-cell methylation probabilities (sigmoid of the per-cell logits) against the observed single-cell bisulfite labels. Missing labels are masked out during training.

- Optimizer: Adam
- Loss: Per-cell binary cross-entropy
- Regularization: Dropout and L2 weight decay

## Citation

```bibtex
@article{angermueller2017deepcpg,
  author    = {Angermueller, Christof and Lee, Heather J. and Reik, Wolf and Stegle, Oliver},
  title     = {{DeepCpG}: accurate prediction of single-cell {DNA} methylation states using deep learning},
  journal   = {Genome Biology},
  volume    = 18,
  number    = 1,
  pages     = {67},
  year      = 2017,
  publisher = {BioMed Central},
  doi       = {10.1186/s13059-017-1189-z}
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

Please contact the authors of the [DeepCpG paper](https://doi.org/10.1186/s13059-017-1189-z) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
