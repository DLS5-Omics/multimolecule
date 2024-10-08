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
pretty_name: EternaBench-External
library_name: multimolecule
---

# EternaBench-External

![EternaBench-External](https://eternagame.org/sites/default/files/thumb_eternabench_paper.png)

EternaBench-External consists of 31 independent RNA datasets from various biological sources, including viral genomes, mRNAs, and synthetic RNAs.
These sequences were probed using techniques such as SHAPE-CE, SHAPE-MaP, and DMS-MaP-seq to understand RNA secondary structures under different experimental and biological conditions.
This dataset serves as a benchmark for evaluating RNA structure prediction models, with a particular focus on generalization to natural RNA molecules.

## Disclaimer

This is an UNOFFICIAL release of the [EternaBench-External](https://github.com/eternagame/EternaBench) by Hannah K. Wayment-Steele, et al.

**The team releasing EternaBench-External did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/eternabench_external
- **Point of Contact**: [Rhiju Das](https://biochemistry.stanford.edu/people/rhiju-das/)

This dataset includes RNA sequences from various biological origins, including viral genomes and mRNAs, and covers a wide range of probing methods like SHAPE-CE and icSHAPE.
Each dataset entry provides sequence information, reactivity profiles, and RNA secondary structure data.
This dataset can be used to examine how RNA structures vary under different conditions and to validate structural predictions for diverse RNA types.

## Example Entry

| name                                                        | sequence                | reactivity                       | seqpos               | class      | dataset        |
| ----------------------------------------------------------- | ----------------------- | -------------------------------- | -------------------- | ---------- | -------------- |
| Dadonaite,2019 Influenza genome SHAPE(1M7) SSII-Mn(2+) Mut. | TTTACCCACAGCTGTGAATT... | [0.639309,0.813297,0.622869,...] | [7425,7426,7427,...] | viral_gRNA | Dadonaite,2019 |

## Column Description

- **name**:
  The name of the dataset entry, typically including the experimental setup and biological source.

- **sequence**:
    The nucleotide sequence of the RNA molecule, represented using the standard RNA bases:

    - **A**: Adenine
    - **C**: Cytosine
    - **G**: Guanine
    - **U**: Uracil

- **reactivity**:
    A list of normalized reactivity values for each nucleotide, representing the likelihood that a nucleotide is unpaired.
    High reactivity indicates high flexibility (unpaired regions), and low reactivity corresponds to paired or structured regions.

- **seqpos**:
  A list of sequence positions corresponding to each nucleotide in the **sequence**.

- **class**:
  The type of RNA sequence, can be one of the following:

    - SARS-CoV-2_gRNA
    - mRNA
    - rRNA
    - small RNAs
    - viral_gRNA

- **dataset**:
    The source or reference for the dataset entry, indicating its origin.

## Variations

This dataset is available in four variants:

- [eternabench-external.1200](https://huggingface.co/datasets/multimolecule/eternabench-external.1200): EternaBench-External dataset with maximum sequence length of 1200 nucleotides.
- [eternabench-external.900](https://huggingface.co/datasets/multimolecule/eternabench-external.900): EternaBench-External dataset with maximum sequence length of 900 nucleotides.
- [eternabench-external.600](https://huggingface.co/datasets/multimolecule/eternabench-external.600): EternaBench-External dataset with maximum sequence length of 600 nucleotides.
- [eternabench-external.300](https://huggingface.co/datasets/multimolecule/eternabench-external.300): EternaBench-External dataset with maximum sequence length of 300 nucleotides.

## Related Datasets

- [eternabench-cm](https://huggingface.co/datasets/multimolecule/eternabench-cm)
- [eternabench-switch](https://huggingface.co/datasets/multimolecule/eternabench-switch)

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{waymentsteele2022rna,
  author    = {Wayment-Steele, Hannah K and Kladwang, Wipapat and Strom, Alexandra I and Lee, Jeehyung and Treuille, Adrien and Becka, Alex and {Eterna Participants} and Das, Rhiju},
  journal   = {Nature Methods},
  month     = oct,
  number    = 10,
  pages     = {1234--1242},
  publisher = {Springer Science and Business Media LLC},
  title     = {{RNA} secondary structure packages evaluated and improved by high-throughput experiments},
  volume    = 19,
  year      = 2022
}
```
