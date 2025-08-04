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

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/chanrg
- **datasets**: https://huggingface.co/datasets/multimolecule/chanrg
- **Point of Contact**: [Zhiyuan Chen](mailto:this@zyc.ai)

## Example Entry

| id                       | sequence                            | secondary_structure                 | structural_annotation               | functional_annotation               | family  | clan    | architecture | super_family | split |
| ------------------------ | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ------- | ------- | ------------ | ------------ | ----- |
| AAAA02037454.1_2001-2135 | GGATGCGATCATACCAGCACTAAAGCACCGGA... | (((((((((....((.(((((...((..((((... | SSSSSSSSSMMMMSSISSSSSIIISSIISSSS... | NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN... | RF00001 | CL00113 | rRNA         | 5S_rRNA      | Train |

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

- **family**:
  The Rfam family accession for the RNA entry, such as `RF00001`. Entries with the same family belong to the same Rfam family model and share the same family-level annotation.

- **clan**:
  The Rfam clan accession for the family, such as `CL00113`. Clans group related Rfam families into a broader category. This field can be `None` when a family has no clan assignment in Rfam.

The `architecture` and `super_family` fields follow the RNArchitecture hierarchy described by [Boccaletto et al. (2018)](https://doi.org/10.1093/nar/gkx966).

- **architecture**:
  A coarse-grained structural architecture label following the hierarchical RNA classification described in the RNArchitecture paper, such as `rRNA`, `hairpin`, `3WJ`, or `complex unclassified`. This field captures the high-level structural organization shared by structurally similar RNA families.

- **super_family**:
  The super-family label following the same RNArchitecture classification, such as `5S_rRNA`. It groups evolutionarily related RNA families within the broader `architecture`. This field can be `None` when no super-family annotation is available.

- **split**:
  The CHANRG split label assigned during dataset conversion.
  - **Train**: Used for model training.
  - **Validation**: Held out for model selection.
  - **Test**: Standard held-out evaluation set.
  - **GenF**: Entire families held out because they cannot be split without accession leakage, used for family-level generalization evaluation.
  - **GenC**: Families from clans without a super-family assignment, held out for clan-level generalization evaluation.
  - **GenA**: Families with `architecture` equal to `complex unclassified`, held out for architecture-level generalization evaluation.

## License

This dataset is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{chen2026fair,
  title         = "Fair splits flip the leaderboard: {CHANRG} reveals limited generalization in {RNA} secondary-structure prediction",
  author        = "Chen, Zhiyuan and Deng, Zhenfeng and Deng, Pan and Liao, Yue and Su, Xiu and Ye, Peng and Liu, Xihui",
  month         =  mar,
  year          =  2026,
  copyright     = "http://creativecommons.org/licenses/by-nc-sa/4.0/",
  archivePrefix = "arXiv",
  primaryClass  = "q-bio.BM",
  eprint        = "2603.22330"
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
