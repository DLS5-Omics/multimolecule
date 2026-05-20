---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: polyadenylation
---

# APARENT2

Deep residual neural network for predicting human 3' UTR Alternative Polyadenylation (APA) and cleavage magnitude at nucleotide resolution, and for deciphering the impact of genetic variants on polyadenylation.

## Disclaimer

This is an UNOFFICIAL implementation of [Deciphering the impact of genetic variation on human polyadenylation using APARENT2](https://doi.org/10.1186/s13059-022-02799-4) by Johannes Linder, Samantha E. Koplik, et al.

The OFFICIAL repository of APARENT2 is at [johli/aparent-resnet](https://github.com/johli/aparent-resnet).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing APARENT2 did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

APARENT2 is a residual convolutional neural network (a ResNet successor to the original [APARENT](https://github.com/johli/aparent)) trained on a 3' UTR massively parallel reporter assay (MPRA). Given a fixed 205 nt polyadenylation signal (PAS) sequence, it predicts a nucleotide-resolution cleavage probability distribution as well as the overall isoform abundance. It is primarily used to score the effect of genetic variants on polyadenylation by comparing the predictions for a reference and an alternate sequence.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) | Max Num Tokens |
| ---------- | ----------- | ------------------ | --------- | -------- | -------------- |
| 28         | 32          | 0.19               | 0.08      | 0.04     | 205            |

### Links

- **Code**: [multimolecule.aparent2](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/aparent2)
- **Data**: Massively-parallel polyadenylation MPRA with variant-effect evaluation data
- **Paper**: [Deciphering the impact of genetic variation on human polyadenylation using APARENT2](https://doi.org/10.1186/s13059-022-02799-4)
- **Developed by**: Johannes Linder, Samantha E. Koplik, Anshul Kundaje, Georg Seelig
- **Model type**: 1D residual CNN successor to APARENT for polyadenylation isoform, cleavage, and variant-effect prediction
- **Original Repository**: [johli/aparent-resnet](https://github.com/johli/aparent-resnet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Polyadenylation Cleavage Prediction

You can use this model directly to predict the cleavage distribution of a 205 nt polyadenylation signal sequence (core hexamer starting at position 70):

```python
>>> import torch
>>> from multimolecule import RnaTokenizer, Aparent2Model

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/aparent2")
>>> model = Aparent2Model.from_pretrained("multimolecule/aparent2")
>>> sequence = "A" * 70 + "AAUAAA" + "A" * 129
>>> output = model(**tokenizer(sequence, return_tensors="pt"))

>>> output.logits.shape
torch.Size([1, 206])
```

#### Variant Effect Scoring

Score a reference and an alternate sequence separately, then compare:

```python
>>> import torch
>>> ref = "A" * 70 + "AAUAAA" + "A" * 129
>>> alt = "A" * 70 + "AAUACA" + "A" * 129
>>> ref_prob = torch.softmax(model(**tokenizer(ref, return_tensors="pt")).logits, dim=-1)
>>> alt_prob = torch.softmax(model(**tokenizer(alt, return_tensors="pt")).logits, dim=-1)
>>> ref_iso = ref_prob[:, 77:127].sum(dim=-1)
>>> alt_iso = alt_prob[:, 77:127].sum(dim=-1)
>>> delta_logodds = torch.log(alt_iso / (1 - alt_iso)) - torch.log(ref_iso / (1 - ref_iso))
```

### Interface

- **Input length**: fixed 205 nt window
- **Hexamer position**: core hexamer (e.g., `AAUAAA`) at position 70 (0-indexed) of the 205 nt window
- **Output**: 206-dim cleavage distribution (one score per input position + trailing "no cleavage in window" bucket)

### Variant Effect

- Score reference and alternate sequences separately and compare their cleavage / isoform predictions
- There is no separate ref/alt output dataclass

## Training Details

APARENT2 was trained to predict nucleotide-resolution cleavage and isoform abundance from 3' UTR MPRA measurements.

### Training Data

The model was trained on the 3' UTR MPRA library used by the original APARENT, re-processed with additional improvements (exact cleavage positions for the Alien1 Random sublibrary and a 20 nt random barcode upstream of the USE in the Alien1 sublibrary). The measured variant data and processed data repository are available at the original [APARENT GitHub](https://github.com/johli/aparent).

### Training Procedure

#### Pre-training

The model minimizes a combination of a sigmoid KL-divergence isoform loss and a KL-divergence cleavage loss, weighted equally. The released inference model corresponds to the residual-network model trained for 5 epochs on all sublibraries (excluding ClinVar wild-type sequences), with dropout disabled for inference.

## Citation

```bibtex
@article{linder2022deciphering,
  author    = {Linder, Johannes and Koplik, Samantha E. and Kundaje, Anshul and Seelig, Georg},
  title     = {Deciphering the impact of genetic variation on human polyadenylation using APARENT2},
  journal   = {Genome Biology},
  volume    = {23},
  number    = {1},
  pages     = {232},
  year      = {2022},
  doi       = {10.1186/s13059-022-02799-4},
  publisher = {Springer Science and Business Media LLC}
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

Please contact the authors of the [APARENT2 paper](https://doi.org/10.1186/s13059-022-02799-4) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
