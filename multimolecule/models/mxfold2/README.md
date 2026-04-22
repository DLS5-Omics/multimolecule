---
language: rna
tags:
  - Biology
  - Secondary Structure
  - RNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: rna-secondary-structure
---

# MXfold2

Deep neural network for RNA secondary structure prediction with thermodynamic integration.

## Disclaimer

This is an UNOFFICIAL implementation of the [RNA secondary structure prediction using deep learning with thermodynamic integration](https://doi.org/10.1038/s41467-021-21194-4) by Kengo Sato, et al.

The OFFICIAL repository of MXfold2 is at [mxfold/mxfold2](https://github.com/mxfold/mxfold2).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoint reproduce the published `TrainSetAB` MXfold2 predictions.

**The team releasing MXfold2 did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

MXfold2 combines a learned positional scoring network with Turner nearest-neighbor thermodynamic parameters and a dynamic-programming decoder. MultiMolecule provides the published `TrainSetAB` checkpoint as a single RNA secondary structure model at [`multimolecule/mxfold2`](https://huggingface.co/multimolecule/mxfold2).

### Model Specification

| Num Parameters (M) |
| ------------------ |
| 0.80               |

FLOPs and MACs are not listed here because MXfold2 includes a native dynamic-programming decoder whose runtime is sequence-length dependent and not directly comparable to feed-forward MM models.

### Links

- **Code**: [multimolecule.mxfold2](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/mxfold2)
- **Weights**: [multimolecule/mxfold2](https://huggingface.co/multimolecule/mxfold2)
- **Paper**: [RNA secondary structure prediction using deep learning with thermodynamic integration](https://doi.org/10.1038/s41467-021-21194-4)
- **Developed by**: Kengo Sato, Michiaki Akiyama, Yasubumi Sakakibara
- **Original Repository**: [mxfold/mxfold2](https://github.com/mxfold/mxfold2)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library and the original MXfold2 backend. You can install them using pip:

```bash
pip install multimolecule mxfold2
```

### Pipeline

You can use MXfold2 directly with the MultiMolecule secondary-structure pipeline:

```python
>>> from transformers import pipeline

>>> predictor = pipeline("rna-secondary-structure", model="multimolecule/mxfold2")
>>> predictor("UAGCUUAUCAGACUGAUGUUG")
{'sequence': 'UAGCUUAUCAGACUGAUGUUG', 'secondary_structure': '((((..((((...))))))))'}
```

### Direct Use

Here is how to use this model to predict RNA secondary structure in PyTorch:

```python
>>> from multimolecule import RnaTokenizer, Mxfold2Model

>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/mxfold2")
>>> model = Mxfold2Model.from_pretrained("multimolecule/mxfold2")
>>> output = model(**tokenizer("UAGCUUAUCAGACUGAUGUUG", return_tensors="pt"))

>>> output.secondary_structure[0]
'((((..((((...))))))))'

>>> output.contact_map.shape
torch.Size([1, 21, 21])
```

> [!NOTE]
> MXfold2 uses the original backend package for dynamic-programming decoding. Inside this repository, MultiMolecule will also use the local `mxfold2-code` tree when it is available.

## Training Details

The released checkpoint corresponds to the authors' `TrainSetAB` model from the official MXfold2 codebase.

### Training Data

- training sets: TrainSetA and TrainSetB from the MXfold2 release.
- released checkpoint: `TrainSetAB`.

### Training Procedure

- objective: learn positional folding scores integrated with Turner nearest-neighbor thermodynamics.
- decoder: Zuker-style dynamic programming with thermodynamic integration.
- released architecture: `MixC` with learned positional scores plus Turner parameters.

## Citation

```bibtex
@article{sato2021rna,
  author  = {Sato, Kengo and Akiyama, Michiaki and Sakakibara, Yasubumi},
  title   = {RNA secondary structure prediction using deep learning with thermodynamic integration},
  journal = {Nature Communications},
  volume  = {12},
  pages   = {941},
  year    = {2021},
  doi     = {10.1038/s41467-021-21194-4}
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

Please contact the authors of the [MXfold2 paper](https://doi.org/10.1038/s41467-021-21194-4) for questions or comments on the paper/model.

## License

This model is licensed under the [GNU Affero General Public License](license.md) and the [MIT License](https://github.com/mxfold/mxfold2/blob/master/LICENSE).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later AND MIT
```
