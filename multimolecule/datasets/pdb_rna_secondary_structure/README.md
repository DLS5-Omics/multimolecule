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
  - multimolecule/pdb
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: pdb-rna_secondary_structure
library_name: multimolecule
---

# pdb-rna_secondary_structure

> [!IMPORTANT]
> The pdb-rna_secondary_structure dataset is in beta test.
> This dataset card may not accurately reflects the data content.
> The data content and this dataset card may subject to change.
>
> Please contact the MultiMolecule team on GitHub issues should you have any feedback.

> [!WARNING]
> This dataset is converted from the dataset released by the authors of SPOT-RNA.
> The MultiMolecule is aware of a potential issue in data quality.
> We are working on cleaning the dataset.

![pdb-rna_secondary_structure](https://cdn.rcsb.org/rcsb-pdb/v2/common/images/rcsb_logo.png)

RCSB Protein Data Bank (RCSB PDB) enables breakthroughs in science and education by providing access and tools for exploration, visualization, and analysis of:

- Experimentally-determined 3D structures from the Protein Data Bank (PDB) archive
- Computed Structure Models (CSM) from AlphaFold DB and ModelArchive

The PDB-rna_secondary_structure is a subset of the PDB dataset focusing on the secondary structure of RNA.

## Disclaimer

This is an UNOFFICIAL release of the RNA Secondary Structure data in the [Protein Data Bank](https://www.rcsb.org).

**The team releasing Protein Data Bank did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/pdb-rna_secondary_structure
- **datasets**: https://huggingface.co/datasets/multimolecule/pdb-rna_secondary_structure
- **Point of Contact**: [RCSB PDB](https://www.rcsb.org/pages/contactus)

## Example Entry

| id       | sequence                            | secondary_structure              |
| -------- | ----------------------------------- | -------------------------------- |
| 1c0a-1-B | GGAGCGGUAGUUCAGUCGGUUAGAAUACCUGC... | (((((((([{{[[)<..AB).)]]}}(((... |

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
