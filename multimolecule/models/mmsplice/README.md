---
language: rna
tags:
  - Biology
  - RNA
  - Splicing
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: splice-variant-effect
---

# MMSplice

Modular modeling of the effects of genetic variants on splicing.

## Disclaimer

This is an UNOFFICIAL implementation of the [MMSplice: modular modeling improves the predictions of genetic variant effects on splicing](https://doi.org/10.1186/s13059-019-1653-z) by Jun Cheng, et al.

The OFFICIAL repository of MMSplice is at [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing MMSplice did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

MMSplice is a _modular_ neural network for predicting the effect of genetic variants on pre-mRNA splicing. It decomposes an exon together with its flanking introns into five regions and scores each region with an independent small convolutional sub-network. For variant-effect estimation, the model is run on both the reference and the alternative sequence, and the per-module score deltas are combined by a fixed linear model into a delta-logit-PSI splicing-effect score. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Modules | Num Parameters (M) | FLOPs (M) | MACs (M) |
| ----------- | ------------------ | --------- | -------- |
| 5           | 0.057              | 5.71      | 2.79     |

(FLOPs and MACs measured on a 220 bp exon-with-flanks input.)

### Links

- **Code**: [multimolecule.mmsplice](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/mmsplice)
- **Data**: Human splice-site and exon data with MPRA exon-skipping variant-effect measurements
- **Paper**: [MMSplice: modular modeling improves the predictions of genetic variant effects on splicing](https://doi.org/10.1186/s13059-019-1653-z)
- **Developed by**: Jun Cheng, Thi Yen Duong Nguyen, Kamil J. Cygan, Muhammed Hasan Çelik, William G. Fairbrother, Žiga Avsec, Julien Gagneur
- **Model type**: Modular 1D CNN with five region-specific sub-networks for splice variant-effect prediction
- **Original Repository**: [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Module Scores

```python
>>> import torch
>>> from multimolecule import RnaTokenizer, MmSpliceForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/mmsplice")
>>> model = MmSpliceForSequencePrediction.from_pretrained("multimolecule/mmsplice")
>>> _ = model.eval()
>>> left_intron = "A" * 100
>>> exon = "C" * 20
>>> right_intron = "G" * 100
>>> reference = tokenizer(left_intron + exon + right_intron, add_special_tokens=False, return_tensors="pt")
>>> output = model.model(**reference)
>>> output["logits"].shape
torch.Size([1, 5])
```

#### Variant Effect

```python
>>> import torch
>>> from multimolecule import RnaTokenizer, MmSpliceForSequencePrediction

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/mmsplice")
>>> model = MmSpliceForSequencePrediction.from_pretrained("multimolecule/mmsplice")
>>> _ = model.eval()
>>> left_intron = "A" * 100
>>> exon = "C" * 20
>>> right_intron = "G" * 100
>>> reference = tokenizer(left_intron + exon + right_intron, add_special_tokens=False, return_tensors="pt")
>>> alternative_exon = exon[:10] + "U" + exon[11:]
>>> alternative = tokenizer(left_intron + alternative_exon + right_intron, add_special_tokens=False, return_tensors="pt")
>>> output = model(
...     reference["input_ids"],
...     alternative_input_ids=alternative["input_ids"],
... )
>>> output["logits"].shape
torch.Size([1, 1])
```

### Interface

- **Input length**: exon sequence with 100 nt upstream intronic context + 100 nt downstream intronic context
- **Tokenization**: disable special tokens; the embedding layer maps `A/C/G/U` ids to the four upstream channels and maps `N`, padding, special, and unknown tokens to all-zero columns
- **Output (reference-only call, `input_ids` / `inputs_embeds`)**: per-module score vector `logits` of shape `(batch_size, 5)`

### Variant Effect

- **Reference + alternative call** (also pass `alternative_input_ids` / `alternative_inputs_embeds`): additionally returns `alternative_logits` and per-module `delta_logits = alternative_logits - logits`
- **`MmSpliceForSequencePrediction`**: requires both reference and alternative; returns the combined scalar delta-logit-PSI score of shape `(batch_size, 1)`

## Training Details

MMSplice was trained as five independent modules on splicing data and the modules were combined with a linear model to predict variant effects on percent-spliced-in (PSI).

### Training Data

The acceptor, donor, exon, and intron modules were trained on splice-site and exon data derived from human reference transcripts. The combining linear model was fit against a massively parallel reporter assay (MPRA) of exon-skipping variants.

### Training Procedure

#### Pre-training

Each module was trained with a sequence-to-scalar objective scoring its region. The module scores (and their reference/alternative deltas) were then combined by a fixed linear model into a delta-logit-PSI splicing-effect score.

## Citation

```bibtex
@article{cheng2019mmsplice,
  title     = {MMSplice: modular modeling improves the predictions of genetic variant effects on splicing},
  author    = {Cheng, Jun and Nguyen, Thi Yen Duong and Cygan, Kamil J and {\c{C}}elik, Muhammed Hasan and Fairbrother, William G and Avsec, {\v{Z}}iga and Gagneur, Julien},
  journal   = {Genome Biology},
  volume    = 20,
  number    = 1,
  pages     = {48},
  year      = 2019,
  publisher = {Springer},
  doi       = {10.1186/s13059-019-1653-z}
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

Please contact the authors of the [MMSplice paper](https://doi.org/10.1186/s13059-019-1653-z) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
