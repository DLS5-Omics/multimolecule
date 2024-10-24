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
pretty_name: RNAStrAlign
library_name: multimolecule
---

# RNAStrAlign

RNAStrAlign is a comprehensive dataset of RNA sequences and their secondary structures.

RNAStrAlign aggregates data from multiple established RNA structure repositories, covering diverse RNA families such as 5S ribosomal RNA, tRNA, and group I introns.

It is considered complementary to the [ArchiveII](./archiveii) dataset.

## Disclaimer

This is an UNOFFICIAL release of the RNAStrAlign by Zhen Tan, et al.

**The team releasing RNAStrAlign did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/rnastralign
- **datasets**: https://huggingface.co/datasets/multimolecule/rnastralign
- **Point of Contact**: [David H. Mathews](mailto:David_Mathews@urmc.rochester.edu) and [Gaurav Sharma](mailto:gaurav.sharma@rochester.edu)

## Example Entry

| id                               | sequence                            | secondary_structure                  | family     | subfamily      |
| -------------------------------- | ----------------------------------- | ------------------------------------ | ---------- | -------------- |
| 16S_rRNA-Actinobacteria-AB002635 | ACACAUGCAAGCGAACGUGAUCUCCAGCUUGC... | .(((.(((..((..((((.(((((.((....)...  | 16S_rRNA   | Actinobacteria |

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

- **subfamily**:
    A more specific subfamily within the family, such as Actinobacteria for 16S rRNA.

    Not all families have subfamilies, in which case this field will be `None`.

## Variations

This dataset is available in two additional variants:

- [rnastralign](https://huggingface.co/datasets/multimolecule/rnastralign): The main RNAStrAlign dataset.
- [rnastralign.512](https://huggingface.co/datasets/multimolecule/rnastralign.512): RNAStrAlign dataset with sequences no longer than 512 nucleotides.
- [rnastralign.1024](https://huggingface.co/datasets/multimolecule/rnastralign.1024): RNAStrAlign dataset with sequences no longer than 1024 nucleotides.

## Related Datasets

- [ArchiveII](https://huggingface.co/datasets/multimolecule/archiveii): A database of RNA secondary with the same families as RNAStrAlign, usually used for testing.
- [bpRNA-spot](https://huggingface.co/datasets/multimolecule/bprna-spot): Another commonly used database in RNA secondary structures prediction.

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{ran2017turbofold,
  author   = {Tan, Zhen and Fu, Yinghan and Sharma, Gaurav and Mathews, David H},
  journal  = {Nucleic Acids Research},
  month    = nov,
  number   = 20,
  pages    = {11570--11581},
  title    = {{TurboFold} {II}: {RNA} structural alignment and secondary structure prediction informed by multiple homologs},
  volume   = 45,
  year     = 2017
}
```
