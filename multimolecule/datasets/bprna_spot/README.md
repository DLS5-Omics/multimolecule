---
language: rna
tags:
  - Biology
  - RNA
license:
  - agpl-3.0
size_categories:
  - 10K<n<100K
source_datasets:
  - multimolecule/bprna
  - multimolecule/pdb
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: bpRNA-spot
library_name: multimolecule
---

# bpRNA-spot

bpRNA-spot is a database of single molecule secondary structures annotated using bpRNA.

bpRNA-spot is a subset of [bpRNA-1m](../bprna).
It applies [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) to remove sequences with more than 80% sequence similarity from bpRNA-1m.
It further randomly splits the remaining sequences into training, validation, and test sets with a ratio of apprxiately 8:1:1.

## Disclaimer

This is an UNOFFICIAL release of the bpRNA-spot by Jaswinder Singh, Jack Hanson, Kuldip Paliwal and Yaoqi Zhou.

**The team releasing bpRNA-spot did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/bprna-spot
- **datasets**: https://huggingface.co/datasets/multimolecule/bprna-spot
- **Point of Contact**: [Kuldip Paliwal](mailto:k.paliwal@griffith.edu.au) and [Yaoqi Zhou](mailto:yaoqi.zhou@griffith.edu.au)

## Related Datasets

- [bpRNA-1m](https://huggingface.co/datasets/multimolecule/bprna): A database of single molecule secondary structures annotated using bpRNA.
- [bpRNA-new](https://huggingface.co/datasets/multimolecule/bprna-new): A dataset of newly discovered RNA families from Rfam 14.2, designed for cross-family validation to assess generalization capability.

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{singh2019rna,
  author    = {Singh, Jaswinder and Hanson, Jack and Paliwal, Kuldip and Zhou, Yaoqi},
  journal   = {Nature Communications},
  month     = nov,
  number    = 1,
  pages     = {5407},
  publisher = {Springer Science and Business Media LLC},
  title     = {{RNA} secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning},
  volume    = 10,
  year      = 2019
}

@article{darty2009varna,
  author    = {Darty, K{\'e}vin and Denise, Alain and Ponty, Yann},
  journal   = {Bioinformatics},
  month     = aug,
  number    = 15,
  pages     = {1974--1975},
  publisher = {Oxford University Press (OUP)},
  title     = {{VARNA}: Interactive drawing and editing of the {RNA} secondary structure},
  volume    = 25,
  year      = 2009
}

@article{berman2000protein,
  author    = {Berman, H M and Westbrook, J and Feng, Z and Gilliland, G and Bhat, T N and Weissig, H and Shindyalov, I N and Bourne, P E},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = 1,
  pages     = {235--242},
  publisher = {Oxford University Press (OUP)},
  title     = {The Protein Data Bank},
  volume    = 28,
  year      = 2000
}
```
