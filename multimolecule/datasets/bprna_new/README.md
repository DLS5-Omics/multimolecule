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

This is an UNOFFICIAL release of the bpRNA-new by Kengo Sato, et al.

**The team releasing bpRNA-new did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/bprna-new
- **datasets**: https://huggingface.co/datasets/multimolecule/bprna-new
- **Point of Contact**: [Kengo Sato](mailto:satoken@bio.keio.ac.jp)

## Related Datasets

- [bpRNA-1m](https://huggingface.co/datasets/multimolecule/bprna): A database of single molecule secondary structures annotated using bpRNA.
- [bpRNA-spot](https://huggingface.co/datasets/multimolecule/bprna-spot): A subset of bpRNA-1m that applies [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) to remove sequences with more than 80% sequence similarity from bpRNA-1m.
- [ArchiveII](https://huggingface.co/datasets/multimolecule/archiveii): A database of RNA secondary with the same families as RNAStrAlign, usually used for testing.

## License

This dataset is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

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
