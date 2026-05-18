---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
datasets:
  - multimolecule/gtex
library_name: multimolecule
---

# MTSplice

Tissue-specific modeling of the effects of genetic variants on splicing.

## Disclaimer

This is an UNOFFICIAL implementation of the [MTSplice predicts effects of genetic variants on tissue-specific splicing](https://doi.org/10.1186/s13059-021-02273-7) by Jun Cheng et al.

The OFFICIAL repository of MTSplice is at [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing MTSplice did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

MTSplice is the tissue-specific second generation of MMSplice. It predicts the effect of genetic variants on cassette-exon splicing across 56 GTEx tissues. The cassette exon together with its flanking introns is fed into two parallel sequence towers whose outputs are combined into a per-tissue delta-logit-PSI splicing-effect vector. Please refer to the [Training Details](#training-details) section for more information on the training process.

MTSplice is distributed as a deep four-member ensemble (`mtsplice_deep0..3`) and an earlier eight-member ensemble (`mtsplice0..7`). The default deep-family model is represented as a single deterministic model based on `mtsplice_deep0`.

### Model Specification

| Num Blocks | Hidden Size | Num Tissues | Num Parameters | FLOPs (M) | MACs (M) |
| ---------- | ----------- | ----------- | -------------- | --------- | -------- |
| 8          | 64          | 56          | 210,840        | 164.36    | 80.90    |

(Num Blocks is per tower; FLOPs and MACs measured on an 800 bp cassette-exon-with-flanks input.)

### Links

- **Code**: [multimolecule.mtsplice](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/mtsplice)
- **Weights**: [multimolecule/mtsplice](https://huggingface.co/multimolecule/mtsplice)
- **Data**: GTEx cassette-exon PSI quantifications across 56 tissues
- **Paper**: [MTSplice predicts effects of genetic variants on tissue-specific splicing](https://doi.org/10.1186/s13059-021-02273-7)
- **Developed by**: Jun Cheng, Muhammed Hasan Çelik, Anshul Kundaje, Julien Gagneur
- **Model type**: Two parallel dilated 1D CNN towers with positional B-spline re-weighting for tissue-specific delta-logit-PSI prediction
- **Original Repository**: [gagneurlab/MMSplice_MTSplice](https://github.com/gagneurlab/MMSplice_MTSplice)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Tissue Scores

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, MtSpliceModel

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/mtsplice")
>>> model = MtSpliceModel.from_pretrained("multimolecule/mtsplice")
>>> reference = tokenizer("agcagtcattatggcgaatctggcaagta", return_tensors="pt")
>>> output = model(**reference)
>>> output["logits"].shape
torch.Size([1, 56])
```

#### Variant Effect

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, MtSpliceForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/mtsplice")
>>> model = MtSpliceForSequencePrediction.from_pretrained("multimolecule/mtsplice")
>>> reference = tokenizer("agcagtcattatggcgaatctggcaagta", return_tensors="pt")
>>> alternative = tokenizer("agcagtcattatggctaatctggcaagta", return_tensors="pt")
>>> output = model(
...     reference["input_ids"],
...     alternative_input_ids=alternative["input_ids"],
... )
>>> output["logits"].shape
torch.Size([1, 56])
```

### Interface

- **Input length**: cassette exon with flanking intronic context (typical ~800 bp)
- **Output (reference-only call, `input_ids` / `inputs_embeds`)**: per-tissue score vector `logits` of shape `(batch_size, 56)`

### Variant Effect

- **Reference + alternative call** (also pass `alternative_input_ids` / `alternative_inputs_embeds`): additionally returns `alternative_logits` and per-tissue `delta_logits = alternative_logits - logits`
- **`MtSpliceForSequencePrediction`**: returns per-tissue deltas (or per-tissue scores when no alternative is supplied); applies standard regression loss when labels are provided

## Training Details

MTSplice was trained to predict tissue-specific percent-spliced-in (PSI) of cassette exons across GTEx tissues, building on the MMSplice modular splicing model with an added tissue-specific neural module.

### Training Data

MTSplice was trained on cassette-exon PSI quantifications across 56 GTEx tissues, together with the human reference splice-site and exon sequence context. The variant-effect predictions were validated against tissue-specific splicing quantitative trait loci (sQTL) and MPRA exon-skipping data.

### Training Procedure

#### Pre-training

The two sequence towers consume one-hot encoded DNA. A dilated-convolution stack with positional B-spline re-weighting extracts splicing features, which a dense head maps to per-tissue delta-logit-PSI. The tissue-resolved predictions are formed from the reference/alternative score deltas.

## Citation

```bibtex
@article{cheng2021mtsplice,
  title     = {MTSplice predicts effects of genetic variants on tissue-specific splicing},
  author    = {Cheng, Jun and {\c{C}}elik, Muhammed Hasan and Kundaje, Anshul and Gagneur, Julien},
  journal   = {Genome Biology},
  volume    = 22,
  number    = 1,
  pages     = {94},
  year      = 2021,
  publisher = {Springer},
  doi       = {10.1186/s13059-021-02273-7}
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

Please contact the authors of the [MTSplice paper](https://doi.org/10.1186/s13059-021-02273-7) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
