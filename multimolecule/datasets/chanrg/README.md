---
language: rna
tags:
  - Biology
  - RNA
license:
  - agpl-3.0
size_categories:
  - 100K<n<1M
source_datasets:
  - multimolecule/rfam
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: CHANRG
library_name: multimolecule
---

# Comprehensive Hierarchical Annotation of Non-coding RNA Groups (CHANRG)

CHANRG is a database of non-coding RNA families and secondary structures.

> [!WARNING]
> This dataset is currently in a **Development Preview** state. It is not yet ready for production use and may contain incomplete or experimental features. The MultiMolecule team is actively working on improving this dataset, and we welcome feedback from the community.
>
> DO NOT DISTRIBUTE, COPY, OR USE THIS DATASET WITHOUT PRIOR WRITTEN PERMISSION FROM THE MULTIMOLECULE TEAM.
> NO ACADEMIC, INDUSTRIAL, OR COMMERCIAL USE IS ALLOWED WITHOUT PRIOR WRITTEN PERMISSION FROM THE MULTIMOLECULE TEAM.

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/chanrg
- **datasets**: https://huggingface.co/datasets/multimolecule/chanrg
- **Point of Contact**: [Zhiyuan Chen](mailto:this@zyc.ai)

## Example Entry

| id              | sequence                            | secondary_structure              | structural_annotation               | functional_annotation               |
| --------------- | ----------------------------------- | -------------------------------- | ----------------------------------- | ----------------------------------- |
| bpRNA_RFAM_1016 | AUUGCUUCUCGGCCUUUUGGCUAACAUCAAGU... | ......(((.((((....)))).)))...... | EEEEEESSSISSSSHHHHSSSSISSSXXXXXX... | NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN... |

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

- **structural_annotation**:
    Structural annotations categorizing different regions of the RNA based on their roles within the secondary structure, consistent with bpRNA standards:

    - **E**: **External Loop** – Regions that are unpaired and external to any loop or helix.
    - **S**: **Stem** – Paired regions forming helical structures.
    - **H**: **Hairpin Loop** – Unpaired regions at the end of a stem, forming a loop.
    - **I**: **Internal Loop** – Unpaired regions between two stems.
    - **M**: **Multi-loop** – Junctions where three or more stems converge.
    - **B**: **Bulge** – Unpaired nucleotides on one side of a stem.
    - **X**: **Ambiguous** or **Undetermined** – Regions where the structure is unknown or cannot be classified.
    - **K**: **Pseudoknot** – Regions involved in pseudoknots, where base pairs cross each other.

- **functional_annotation**:
    Functional annotations indicating specific functional elements or regions within the RNA sequence, as defined by bpRNA:

    - **N**: **None** – No specific functional annotation is assigned.
    - **K**: **Pseudoknot** – Marks nucleotides involved in pseudoknot structures, which can be functionally significant.

## License

This dataset is licensed under the MultiMolecule Development Preview License.

DO NOT DISTRIBUTE, COPY, OR USE THIS DATASET WITHOUT PRIOR WRITTEN PERMISSION FROM THE MULTIMOLECULE TEAM.
NO ACADEMIC, INDUSTRIAL, OR COMMERCIAL USE IS ALLOWED WITHOUT PRIOR WRITTEN PERMISSION FROM THE MULTIMOLECULE TEAM.
