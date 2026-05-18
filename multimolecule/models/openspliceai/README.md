---
language: rna
tags:
  - Biology
  - Genomics
  - RNA
  - Splicing
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
pipeline_tag: other
pipeline: splice-site
---

# OpenSpliceAI

Modular native-PyTorch reimplementation of SpliceAI for predicting pre-mRNA splice sites from nucleotide sequence.

## Disclaimer

This is an UNOFFICIAL implementation of [OpenSpliceAI: An efficient, modular implementation of SpliceAI enabling easy retraining on non-human species](https://doi.org/10.7554/eLife.107454.3) by Kuan-Hao Chao, Alan Mao, et al.

The OFFICIAL repository of OpenSpliceAI is at [Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing OpenSpliceAI did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

OpenSpliceAI is a deep dilated residual convolutional neural network that reimplements the SpliceAI architecture in native PyTorch. It predicts, for each nucleotide of a pre-mRNA transcript, whether the position is a splice acceptor, a splice donor, or neither. The model stacks dilated residual units with increasing kernel size and atrous rate so that a wide genomic context window contributes to each per-nucleotide prediction, while skip connections aggregate multi-scale features. OpenSpliceAI reproduces the predictive behavior of SpliceAI while providing an efficient, modular training pipeline that can be retrained on non-human species.

### Variants

OpenSpliceAI ships trained model families for human MANE and four non-human species. Each family provides four
flanking-context sizes.

| Family        | 80 nt                                                                                             | 400 nt                                                                                              | 2,000 nt                                                                                              | 10,000 nt                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| MANE / human  | [`openspliceai-mane.80`](https://huggingface.co/multimolecule/openspliceai-mane.80)               | [`openspliceai-mane.400`](https://huggingface.co/multimolecule/openspliceai-mane.400)               | [`openspliceai-mane.2000`](https://huggingface.co/multimolecule/openspliceai-mane.2000)               | [`openspliceai-mane.10000`](https://huggingface.co/multimolecule/openspliceai-mane.10000)               |
| Mouse         | [`openspliceai-mouse.80`](https://huggingface.co/multimolecule/openspliceai-mouse.80)             | [`openspliceai-mouse.400`](https://huggingface.co/multimolecule/openspliceai-mouse.400)             | [`openspliceai-mouse.2000`](https://huggingface.co/multimolecule/openspliceai-mouse.2000)             | [`openspliceai-mouse.10000`](https://huggingface.co/multimolecule/openspliceai-mouse.10000)             |
| Zebrafish     | [`openspliceai-zebrafish.80`](https://huggingface.co/multimolecule/openspliceai-zebrafish.80)     | [`openspliceai-zebrafish.400`](https://huggingface.co/multimolecule/openspliceai-zebrafish.400)     | [`openspliceai-zebrafish.2000`](https://huggingface.co/multimolecule/openspliceai-zebrafish.2000)     | [`openspliceai-zebrafish.10000`](https://huggingface.co/multimolecule/openspliceai-zebrafish.10000)     |
| Honeybee      | [`openspliceai-honeybee.80`](https://huggingface.co/multimolecule/openspliceai-honeybee.80)       | [`openspliceai-honeybee.400`](https://huggingface.co/multimolecule/openspliceai-honeybee.400)       | [`openspliceai-honeybee.2000`](https://huggingface.co/multimolecule/openspliceai-honeybee.2000)       | [`openspliceai-honeybee.10000`](https://huggingface.co/multimolecule/openspliceai-honeybee.10000)       |
| _Arabidopsis_ | [`openspliceai-arabidopsis.80`](https://huggingface.co/multimolecule/openspliceai-arabidopsis.80) | [`openspliceai-arabidopsis.400`](https://huggingface.co/multimolecule/openspliceai-arabidopsis.400) | [`openspliceai-arabidopsis.2000`](https://huggingface.co/multimolecule/openspliceai-arabidopsis.2000) | [`openspliceai-arabidopsis.10000`](https://huggingface.co/multimolecule/openspliceai-arabidopsis.10000) |

### Model Specification

<table>
<thead>
  <tr>
    <th>Flanking Context</th>
    <th>Residual Blocks</th>
    <th>Hidden Size</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>80 nt</td>
    <td>4</td>
    <td rowspan="4">32</td>
    <td>0.09</td>
    <td>0.95</td>
    <td>0.47</td>
  </tr>
  <tr>
    <td>400 nt</td>
    <td>8</td>
    <td>0.19</td>
    <td>2.00</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td>2,000 nt</td>
    <td>12</td>
    <td>0.36</td>
    <td>5.03</td>
    <td>2.50</td>
  </tr>
  <tr>
    <td>10,000 nt</td>
    <td>16</td>
    <td>0.70</td>
    <td>20.90</td>
    <td>10.40</td>
  </tr>
</tbody>
</table>

Model size is determined by flanking context and is shared across species for the same context. FLOPs and MACs are
reported for a single 5,000-nucleotide output sequence.

### Links

- **Code**: [multimolecule.openspliceai](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/openspliceai)
- **Data**: Human MANE/GENCODE for the MANE variants; species annotations follow the original OpenSpliceAI release.
- **Paper**: [OpenSpliceAI: An efficient, modular implementation of SpliceAI enabling easy retraining on non-human species](https://doi.org/10.7554/eLife.107454.3)
- **Developed by**: Kuan-Hao Chao, Alan Mao, Anqi Liu, Steven L. Salzberg, Mihaela Pertea
- **Model type**: Dilated residual 1D CNN over pre-mRNA sequence for per-nucleotide three-class splice-site classification
- **Original Repository**: [Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RNA Splicing Site Prediction

You can use this model directly to predict the splice sites of a pre-mRNA sequence:

```python
>>> from multimolecule import RnaTokenizer, OpenSpliceAiForTokenPrediction

>>> model_id = "multimolecule/openspliceai-mane.10000"
>>> tokenizer = RnaTokenizer.from_pretrained(model_id)
>>> model = OpenSpliceAiForTokenPrediction.from_pretrained(model_id)
>>> output = model(tokenizer("AGCAGUCAUUAUGGCGAA", return_tensors="pt")["input_ids"])

>>> output.keys()
odict_keys(['logits'])
```

Each output position carries three logits corresponding to _neither_, _acceptor_, and _donor_.

### Interface

- **Input length**: variable pre-mRNA sequence
- **Flanking context**: 80 / 400 / 2,000 / 10,000 nt per variant family, split evenly on both sides of every predicted position
- **Padding**: sequence ends padded with `N`
- **Output**: per-position 3-class logits (`neither`, `acceptor`, `donor`)

## Training Details

OpenSpliceAI was trained to predict the location of splice donor and acceptor sites from nucleotide sequence, following the SpliceAI training methodology.

### Training Data

The MANE variants were trained on transcripts from the [GENCODE](https://multimolecule.danling.org/datasets/gencode)/MANE human reference annotation. The non-human variants use the species annotations released by OpenSpliceAI for mouse, zebrafish, honeybee, and _Arabidopsis_. For each predicted nucleotide, the model receives a flanking context of 80, 400, 2,000, or 10,000 nucleotides, split evenly across the two sides of the output sequence, with sequence ends padded with `N`. Annotated splice donor and acceptor sites serve as positive labels; all other positions are negative.

### Training Procedure

#### Pre-training

The model was trained to minimize a cross-entropy loss between predicted splice-site probabilities and the reference annotation.

- Optimizer: Adam
- Loss: cross-entropy

Please refer to the [OpenSpliceAI paper](https://doi.org/10.7554/eLife.107454.3) for the full training protocol and hardware details.

## Citation

```bibtex
@article{chao2025openspliceai,
  author    = {Chao, Kuan-Hao and Mao, Alan and Liu, Anqi and Salzberg, Steven L and Pertea, Mihaela},
  title     = {OpenSpliceAI: An efficient, modular implementation of SpliceAI enabling easy retraining on non-human species},
  journal   = {eLife},
  volume    = 14,
  pages     = {RP107454},
  year      = 2025,
  doi       = {10.7554/eLife.107454.3},
  publisher = {eLife Sciences Publications, Ltd}
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

Please contact the authors of the [OpenSpliceAI paper](https://doi.org/10.7554/eLife.107454.3) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
