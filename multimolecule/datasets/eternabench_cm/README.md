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
pretty_name: EternaBench-CM
library_name: multimolecule
---

# EternaBench-CM

![EternaBench-CM](https://eternagame.org/sites/default/files/thumb_eternabench_paper.png)

EternaBench-CM is a synthetic RNA dataset comprising 12,711 RNA constructs that have been chemically mapped using SHAPE and MAP-seq methods.
These RNA sequences are probed to obtain experimental data on their nucleotide reactivity, which indicates whether specific regions of the RNA are flexible or structured.
The dataset provides high-resolution, large-scale data that can be used for studying RNA folding and stability.

## Disclaimer

This is an UNOFFICIAL release of the [EternaBench-CM](https://github.com/eternagame/EternaBench) by Hannah K. Wayment-Steele, et al.

**The team releasing EternaBench-CM did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/eternabench_cm
- **datasets**: https://huggingface.co/datasets/multimolecule/eternabench-cm
- **Point of Contact**: [Rhiju Das](https://biochemistry.stanford.edu/people/rhiju-das/)

The dataset includes a large set of synthetic RNA sequences with experimental chemical mapping data, which provides a quantitative readout of RNA nucleotide reactivity. These data are ensemble-averaged and serve as a critical benchmark for evaluating secondary structure prediction algorithms in their ability to model RNA folding dynamics.

## Example Entry

| index    | design                 | sequence         | secondary_structure | reactivity                 | errors                      | signal_to_noise |
| -------- | ---------------------- | ---------------- | ------------------- | -------------------------- | --------------------------- | --------------- |
| 769337-1 | d+m plots weaker again | GGAAAAAAAAAAA... | ................    | [0.642,1.4853,0.1629, ...] | [0.3181,0.4221,0.1823, ...] | 3.227           |

## Column Description

- **id**:
    A unique identifier for each RNA sequence entry.

- **design**:
    The name given to each RNA design by contributors, used for easy reference.

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

- **reactivity**:
    A list of normalized reactivity values for each nucleotide, representing the likelihood that a nucleotide is unpaired.
    High reactivity indicates high flexibility (unpaired regions), and low reactivity corresponds to paired or structured regions.

- **errors**:
    Arrays of floating-point numbers indicating the experimental errors corresponding to the measurements in the **reactivity**.
    These values help quantify the uncertainty in the degradation rates and reactivity measurements.

- **signal_to_noise**:
    The signal-to-noise ratio calculated from the reactivity and error values, providing a measure of data quality.

## Related Datasets

- [eternabench-switch](https://huggingface.co/datasets/multimolecule/eternabench-switch)
- [eternabench-external.1200](https://huggingface.co/datasets/multimolecule/eternabench-external.1200): EternaBench-External dataset with maximum sequence length of 1200 nucleotides.

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
