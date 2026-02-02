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
pretty_name: ArchiveII
library_name: multimolecule
---

# ArchiveII

ArchiveII is a dataset of RNA sequences and their secondary structures, widely used in RNA secondary structure prediction benchmarks.

ArchiveII contains 2975 RNA samples across 10 RNA families, with sequence lengths ranging from 28 to 2968 nucleotides.
This dataset is frequently used to evaluate RNA secondary structure prediction methods, including those that handle both pseudoknotted and non-pseudoknotted structures.

It is considered complementary to the [RNAStrAlign](./rnastralign) dataset.

## Disclaimer

This is an UNOFFICIAL release of the ArchiveII by Mehdi Saman Booy, et al.

**The team releasing ArchiveII did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/archiveii
- **datasets**: https://huggingface.co/datasets/multimolecule/archiveii
- **Point of Contact**: [Mehdi Saman Booy](mailto:mehdi.samanbooy@aalto.fi)

## Example Entry

| id                  | sequence                            | secondary_structure                  | family     |
| ------------------- | ----------------------------------- | ------------------------------------ | ---------- |
| 16S_rRNA-A.fulgidus | AUUCUGGUUGAUCCUGCCAGAGGCCGCUGCUA... | ...(((((...(((.))))).((((((((((....  | 16S_rRNA   |

## Column Description

- **id**:
    A unique identifier for each RNA entry. This ID is derived from the family and the original `.sta` file name, and serves as a reference to the specific RNA structure within the dataset.

- **sequence**:
    The nucleotide sequence of the RNA molecule, represented using the standard RNA bases:

    - **A**: Adenine
    - **C**: Cytosine
    - **G**: Guanine
    - **U**: Uracil

- **secondary_structure**:
    The secondary structure of the RNA represented in dot-bracket notation, using up to three types of symbols to indicate base pairing and unpaired regions, as per bpRNA's standard:

    - **Dots (`.`)**: Represent unpaired nucleotides.
    - **Parentheses (`(` and `)`)**: Represent base pairs in standard stems (page 1).

- **family**:
    The RNA family to which the sequence belongs, such as 16S rRNA, 5S rRNA, etc.

## Variants

This dataset is available in two additional variants:

- [archiveii](https://huggingface.co/datasets/multimolecule/archiveii): The main ArchiveII dataset.
- [archiveii.512](https://huggingface.co/datasets/multimolecule/archiveii.512): ArchiveII dataset with sequences no longer than 512 nucleotides.
- [archiveii.1024](https://huggingface.co/datasets/multimolecule/archiveii.1024): ArchiveII dataset with sequences no longer than 1024 nucleotides.

## Related Datasets

- [RNAStrAlign](https://huggingface.co/datasets/multimolecule/rnastralign): A database of RNA secondary with the same families as ArchiveII, usually used for training.
- [bpRNA-spot](https://huggingface.co/datasets/multimolecule/bprna-spot): Another commonly used database in RNA secondary structures prediction.

## License

This dataset is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
## Citation

```bibtex
@article{samanbooy2022rna,
  author    = {Saman Booy, Mehdi and Ilin, Alexander and Orponen, Pekka},
  journal   = {BMC Bioinformatics},
  keywords  = {Deep learning; Pseudoknotted structures; RNA structure prediction},
  month     = feb,
  number    = 1,
  pages     = {58},
  publisher = {Springer Science and Business Media LLC},
  title     = {{RNA} secondary structure prediction with convolutional neural networks},
  volume    = 23,
  year      = 2022
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
