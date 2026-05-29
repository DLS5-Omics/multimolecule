---
language: rna
tags:
  - Biology
  - RNA
  - Splicing
license: agpl-3.0
datasets:
  - multimolecule/deltasplice
library_name: multimolecule
pipeline_tag: other
pipeline: splice-site
---

# DeltaSplice

Reference-informed prediction of alternative splicing and splicing-altering mutations from sequences.

## Disclaimer

This is an UNOFFICIAL implementation of [Reference-informed prediction of alternative splicing and splicing-altering mutations from sequences](https://doi.org/10.1101/gr.279044.124) by Chencheng Xu, Suying Bao, et al.

The OFFICIAL repository of DeltaSplice is at [chaolinzhanglab/DeltaSplice](https://github.com/chaolinzhanglab/DeltaSplice).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing DeltaSplice did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

DeltaSplice predicts splice-site usage (SSU) and splicing-altering mutation effects from sequence. The model uses a valid-convolution dilated residual encoder and three prediction modules: splice-site usage, reference-informed delta-SSU, and an auxiliary splice-site head. The official package uses the average prediction of five checkpoints for SSU and delta-SSU prediction; MultiMolecule stores the five seed checkpoints of each released data variant as internal ensemble members and returns their average prediction.

### Variants

- **[multimolecule/deltasplice](https://huggingface.co/multimolecule/deltasplice)**: DeltaSplice trained on the multi-species training set used by the upstream default checkpoints.
- **[multimolecule/deltasplice-human](https://huggingface.co/multimolecule/deltasplice-human)**: DeltaSplice human-only comparison checkpoint set released by the upstream project.

### Model Specification

| Variant           | Num Layers | Hidden Size | Context | Ensemble Members | Num Parameters (M) | FLOPs (M)  | MACs (M)  |
| ----------------- | ---------- | ----------- | ------- | ---------------- | ------------------ | ---------- | --------- |
| DeltaSplice       | 24         | 64          | 30000   | 5                | 40.376             | 1642965.72 | 820284.36 |
| DeltaSplice-Human | 24         | 64          | 30000   | 5                | 40.376             | 1642965.72 | 820284.36 |

(FLOPs and MACs measured on one requested output nucleotide with the default 30 kb padded context.)

### Links

- **Code**: [multimolecule.deltasplice](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/deltasplice)
- **Paper**: [Reference-informed prediction of alternative splicing and splicing-altering mutations from sequences](https://doi.org/10.1101/gr.279044.124)
- **Developed by**: Chencheng Xu, Suying Bao, Ye Wang, Wenxing Li, Hao Chen, Yufeng Shen, Tao Jiang, Chaolin Zhang
- **Model type**: Dilated residual 1D CNN ensemble for splice-site usage and delta-SSU prediction
- **Original Repository**: [chaolinzhanglab/DeltaSplice](https://github.com/chaolinzhanglab/DeltaSplice)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Splice-Site Usage

```python
>>> from multimolecule import RnaTokenizer
>>> from multimolecule.models.deltasplice import DeltaSpliceModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/deltasplice")
>>> model = DeltaSpliceModel.from_pretrained("multimolecule/deltasplice")
>>> inputs = tokenizer("AGCAGUCAUUAUGGCGAAUCUGGCAAGUA", return_tensors="pt")
>>> output = model(**inputs)
>>> output["probabilities"].shape
torch.Size([1, 30, 3])
```

#### Variant Effect

```python
>>> from multimolecule import RnaTokenizer
>>> from multimolecule.models.deltasplice import DeltaSpliceModel

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/deltasplice")
>>> model = DeltaSpliceModel.from_pretrained("multimolecule/deltasplice")
>>> reference = tokenizer("AGCAGUCAUUAUGGCGAAUCUGGCAAGUA", return_tensors="pt")
>>> alternative = tokenizer("AGCAGUCAUUAUGGCUAAUCUGGCAAGUA", return_tensors="pt")
>>> output = model(reference["input_ids"], alternative_input_ids=alternative["input_ids"], use_reference=True)
>>> output["delta"].shape
torch.Size([1, 30, 3])
```

### Interface

- **Input**: RNA sequence tokenized with `RnaTokenizer`; `N` is encoded as zero nucleotide channels
- **Output channels**: `no_splice`, `acceptor`, `donor`
- **Reference-only call**: returns per-position splice-site usage probabilities in `probabilities`
- **Reference + alternative call**: pass the reference sequence as `input_ids` and the alternate sequence as `alternative_input_ids`
- **Reference usage**: pass `reference_usage` with shape `(batch_size, sequence_length, 3)` or omit it to use the model's own reference usage as the reference signal

## Training Details

DeltaSplice was trained to predict splice-site usage from gene sequence and to improve mutation-effect prediction by incorporating reference splice-site usage.

### Training Data

The upstream repository describes training from `gene_dataset.tsu.txt`, which contains splice-site usage in adult brains of eight mammalian species.

### Training Procedure

The official release provides five seed checkpoints with the same architecture and data split. MultiMolecule represents these seed checkpoints as internal ensemble members rather than public model variants.

## Citation

```bibtex
@article{xu2024deltasplice,
  title     = {Reference-informed prediction of alternative splicing and splicing-altering mutations from sequences},
  author    = {Xu, Chencheng and Bao, Suying and Wang, Ye and Li, Wenxing and Chen, Hao and Shen, Yufeng and Jiang, Tao and Zhang, Chaolin},
  journal   = {Genome Research},
  volume    = {34},
  number    = {7},
  pages     = {1052--1065},
  year      = {2024},
  doi       = {10.1101/gr.279044.124}
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

Please contact the authors of the [DeltaSplice paper](https://doi.org/10.1101/gr.279044.124) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
