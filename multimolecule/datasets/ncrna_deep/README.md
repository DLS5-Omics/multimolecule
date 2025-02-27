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
  - multimolecule/rfam
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: ncRNA_deep-Rfam
library_name: multimolecule
---

# ncRNA_deep-Rfam

`ncRNA_deep-Rfam` is a subset of the `ncRNA_deep` dataset focused on small non-coding RNA sequences sourced from the Rfam database.
This subset includes 148,530 sequences after extensive data processing.

This subset is derived from Rfam Version 14.0, with the following processing pipeline:

1. Extracted 650,790 ncRNA sequences distributed across 2,570 functional families.
2. Removed sequences with non-canonical bases (e.g., letters other than A, C, G, and U).
3. Excluded classes corresponding to long non-coding RNAs or classes with average sequence lengths >200 nucleotides.
4. Filtered out classes strongly dependent on sequence length, identified using a C5.0 decision tree model, to ensure robustness against trivial classification signals.
5. Removed underrepresented classes with <400 sequences, resulting in 306,016 sequences across 88 Rfam functional families.

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of ncRNA-deep.
>
> The original paper of ncRNA-deep reports a dataset with 306,016 sequences, but the provided dataset contains only 148,530 sequences.
>
> This is being tracked in [issue #10](https://github.com/bioinformatics-sannio/ncrna-deep/issues/2).

## Dataset Description

- **Homepage**: [GitHub - ncRNA_deep](https://github.com/bioinformatics-sannio/ncrna-deep)
- **Documentation**: [ncRNA_deep Documentation](https://github.com/bioinformatics-sannio/ncrna-deep)
- **Point of Contact**: [Luigi Cerulo](mailto:lcerulo@unisannio.it)

## Variations

This dataset is available in the following subset:

- **[ncRNA_deep-Rfam](https://huggingface.co/datasets/multimolecule/ncrna_deep_rfam)**: The Rfam subset contains 306,016 ncRNA sequences across 88 functional families derived from Rfam Version 14.0.

## Example Entry

| index      | sequence                  | class    | source     |
|------------|---------------------------|----------|------------|
| RF00001-1  | GCAAGUGGAGUUUGGGGUAC...  | rRNA     | ncRNA_deep-Rfam |

## Column Description

- **index**:
    A unique identifier for each RNA sequence in the dataset, formatted as `RF` IDs.

- **sequence**:
    The nucleotide sequence of the RNA.

- **class**:
    The functional class assigned to each ncRNA sequence (e.g., rRNA, miRNA, snoRNA).

- **source**:
    The subset of the dataset where the sequence originates (`ncRNA_deep-Rfam`).

## Use Cases

- Functional classification of small ncRNA sequences.
- Benchmarking machine learning models for ncRNA classification tasks.

## License

This dataset is licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

```spdx
SPDX-License-Identifier: CC-BY-4.0
