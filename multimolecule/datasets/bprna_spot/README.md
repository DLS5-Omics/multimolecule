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

bpRNA-spot is a collection of the datasets used by SPOT-RNA for RNA secondary structure prediction.

The dataset is released as a composite repository, `bpRNA-spot`, and three numbered component repositories:

- `bpRNA-spot-0`: the initial bpRNA split, `TR0`, `VL0`, and `TS0`.
- `bpRNA-spot-1`: the PDB transfer-learning split, `TR1`, `VL1`, and `TS1`.
- `bpRNA-spot-2`: the NMR-only evaluation split, `TS2`.

`bpRNA-spot` concatenates the components in order:

- `train`: `TR0 + TR1`
- `validation`: `VL0 + VL1`
- `test`: `TS0 + TS1 + TS2`

The `TR0`/`VL0`/`TS0` split is a subset of [bpRNA-1m](../bprna).
It applies [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) to remove sequences with more than 80% sequence similarity from bpRNA-1m.
It further randomly splits the remaining sequences into training, validation, and test sets with a ratio of approximately 8:1:1.

The `TR1`/`VL1`/`TS1` split contains high-resolution PDB RNAs used for transfer learning.
`TS2` contains 39 RNAs solved by NMR and is used for post-training evaluation.

All secondary structures are stored as dot-bracket notation.
For the sequence/label splits, base pairs that would make a nucleotide pair with multiple partners are removed before converting to dot-bracket notation.
Non-`A/C/G/U` symbols in those sequence files are normalized to `N`.

## Schema

| Column | Description |
| --- | --- |
| `id` | Identifier of the sequence. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Secondary structure in dot-bracket notation. Pseudoknots may use bracket tiers beyond `()`. |
| `structural_annotation` | bpRNA-style structural annotation generated from the stored dot-bracket structure. |
| `functional_annotation` | bpRNA-style functional annotation generated from the stored dot-bracket structure. |

## Disclaimer

This is an UNOFFICIAL release of the bpRNA-spot by Jaswinder Singh, et al.

**The team releasing bpRNA-spot did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/bprna-spot
- **datasets**: https://huggingface.co/datasets/multimolecule/bprna-spot
- **Point of Contact**: [Kuldip Paliwal](mailto:k.paliwal@griffith.edu.au) and [Yaoqi Zhou](mailto:yaoqi.zhou@griffith.edu.au)

## Related Datasets

- [bpRNA-1m](https://huggingface.co/datasets/multimolecule/bprna): A database of single molecule secondary structures annotated using bpRNA.
- [bpRNA-new](https://huggingface.co/datasets/multimolecule/bprna-new): A dataset of newly discovered RNA families from Rfam 14.2, designed for cross-family validation to assess generalization capability.
- [RNAStrAlign](https://huggingface.co/datasets/multimolecule/rnastralign): A database of RNA secondary with the same families as ArchiveII, usually used for training.

## License

This dataset is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

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
