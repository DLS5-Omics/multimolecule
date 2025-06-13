---
language: rna
tags:
  - Biology
  - RNA
license:
  - agpl-3.0
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: RIVAS
library_name: multimolecule
---

# RIVAS

The RIVAS dataset is a curated collection of RNA sequences and their secondary structures, designed for training and evaluating RNA secondary structure prediction methods.
The dataset combines sequences from published studies and databases like Rfam, covering diverse RNA families such as tRNA, SRP RNA, and ribozymes.
The secondary structure data is obtained from experimentally verified structures and consensus structures from Rfam alignments, ensuring high-quality annotations for model training and evaluation.

## Disclaimer

This is an UNOFFICIAL release of the RIVAS dataset by Elena Rivas, et al.

**The team releasing RIVAS did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/rivas
- **Point of Contact**: [Elena Rivas](mailto:elenarivas@fas.harvard.edu)

## Example Entry

| id                      | sequence                            | secondary_structure                 |
| ----------------------- | ----------------------------------- | ----------------------------------- |
| AACY020454584.1_604-676 | ACUGGUUGCGGCCAGUAUAAAUAGUCUUUAAG... | ((((........)))).........((........ |

## Column Description

The converted dataset consists of the following columns, each providing specific information about the RNA secondary structures, consistent with the bpRNA standard:

- **id**:
    A unique identifier for each RNA entry. This ID is derived from the original `.sta` file name and serves as a reference to the specific RNA structure within the dataset.

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
    - **Square Brackets (`[` and `]`)**: Represent base pairs in pseudoknots (page 2).
    - **Curly Braces (`{` and `}`)**: Represent base pairs in additional pseudoknots (page 3).

## Variants

This dataset is available in three variants:

- [RIVAS](https://huggingface.co/datasets/multimolecule/rivas): Includes TrainSetA (3166 sequences) for training, TestSetA (697 sequences) for validation and TestSetB (430 sequences) for testing.
- [RIVAS-A](https://huggingface.co/datasets/multimolecule/rivas-a): Includes TrainSetA (3166 sequences) and TestSetA (697 sequences), emphasizing sequence diversity while minimizing overlap between training and test sets. Suitable for evaluating RNA secondary structure prediction models on diverse RNA families.
- [RIVAS-B](https://huggingface.co/datasets/multimolecule/rivas-b): Consists of TrainSetB (1094 sequences) and TestSetB (430 sequences) derived from Rfam alignments, offering additional structural diversity and RNA types not present in RIVAS-A. Designed for testing the generalization capability of models trained on different types of RNA structures.

## Related Datasets

- [bpRNA-spot](https://huggingface.co/datasets/multimolecule/bprna-spot): A subset of RIVAS that applies [CD-HIT (CD-HIT-EST)](https://sites.google.com/view/cd-hit) to remove sequences with more than 80% sequence similarity from RIVAS.
- [RNAStrAlign](https://huggingface.co/datasets/multimolecule/rnastralign): A database of RNA secondary with the same families as ArchiveII, usually used for training.

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{rivas2012a,
  author    = {Rivas, Elena and Lang, Raymond and Eddy, Sean R},
  journal   = {RNA},
  month     = feb,
  number    = 2,
  pages     = {193--212},
  publisher = {Cold Spring Harbor Laboratory},
  title     = {A range of complex probabilistic models for {RNA} secondary structure prediction that includes the nearest-neighbor model and more},
  volume    = 18,
  year      = 2012
}
```
