---
language: rna
tags:
  - Biology
  - RNA
license:
  - agpl-3.0
size_categories:
  - 1K<n<10K
source_datasets:
  - multimolecule/rfam
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: bpRNA-new
library_name: multimolecule
---

# bpRNA-new

bpRNA-new is a database of single molecule secondary structures annotated using bpRNA.

bpRNA-new is a dataset of RNA families from Rfam 14.2, designed for cross-family validation to assess generalization capability.
It focuses on families distinct from those in [bpRNA-1m](../bprna), providing a robust benchmark for evaluating model performance on unseen RNA families.

## Disclaimer

This is an UNOFFICIAL release of the bpRNA-new by Kengo Sato, Manato Akiyama and Yasubumi Sakakibara.

**The team releasing bpRNA-new did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/bprna-new
- **datasets**: https://huggingface.co/datasets/multimolecule/bprna-new
- **Point of Contact**: [Kengo Sato](mailto:satoken@bio.keio.ac.jp)

## Related Datasets

- [bpRNA-1m](https://huggingface.co/datasets/multimolecule/bprna): A database of single molecule secondary structures annotated using bpRNA.
- [bpRNA-spot](https://huggingface.co/datasets/multimolecule/bprna-spot): A subset of bpRNA-1m that applies [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) to remove sequences with more than 80% sequence similarity from bpRNA-1m.

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{sato2021rna,
  author    = {Sato, Kengo and Akiyama, Manato and Sakakibara, Yasubumi},
  journal   = {Nature Communications},
  month     = feb,
  number    = 1,
  pages     = {941},
  publisher = {Springer Science and Business Media LLC},
  title     = {{RNA} secondary structure prediction using deep learning with thermodynamic integration},
  volume    = 12,
  year      = 2021
}
```
