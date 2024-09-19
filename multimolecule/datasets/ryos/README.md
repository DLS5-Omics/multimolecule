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
pretty_name: RYOS
library_name: multimolecule
---

# RYOS

![RYOS](https://eternagame.org/sites/default/files/hero-covid.jpg)

RYOS is a database of RNA backbone stability in aqueous solution.

RYOS focuses on exploring the stability of mRNA molecules for vaccine applications.
This dataset is part of a broader effort to address one of the key challenges of mRNA vaccines: degradation during shipping and storage.

## Statement

_Deep learning models for predicting RNA degradation via dual crowdsourcing_ is published in [Nature Machine Intelligence](https://doi.org/10.1038/s42256-022-00571-8), which is a Closed Access / Author-Fee journal.

> Machine learning has been at the forefront of the movement for free and open access to research.
>
> We see no role for closed access or author-fee publication in the future of machine learning research and believe the adoption of these journals as an outlet of record for the machine learning community would be a retrograde step.

The MultiMolecule team is committed to the principles of open access and open science.

We do NOT endorse the publication of manuscripts in Closed Access / Author-Fee journals and encourage the community to support Open Access journals and conferences.

Please consider signing the [Statement on Nature Machine Intelligence](https://openaccess.engineering.oregonstate.edu).

## Disclaimer

This is an UNOFFICIAL release of the [RYOS](https://www.kaggle.com/competitions/stanford-covid-vaccine) by Hannah K. Wayment-Steele, et al.

**The team releasing RYOS did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/ryos
- **Point of Contact**: [Rhiju Das](https://biochemistry.stanford.edu/people/rhiju-das/)
- **Kaggle Challenge**: https://www.kaggle.com/competitions/stanford-covid-vaccine
- **Eterna Round 1**: https://eternagame.org/labs/9830365
- **Eterna Round 2**: https://eternagame.org/labs/10207059

## Example Entry

| id      | design  | sequence      | secondary_structure | reactivity                    | errors_reactivity            | signal_to_noise_reactivity | deg_pH10                      | errors_deg_pH10              | signal_to_noise_deg_pH10 | deg_50C                     | errors_deg_50C | signal_to_noise_deg_50C            | deg_Mg_pH10                   | errors_deg_Mg_pH10           | signal_to_noise_deg_Mg_pH10 | deg_Mg_50C                  | errors_deg_Mg_50C            | signal_to_noise_deg_Mg_50C | SN_filter |
| ------- | ------- | ------------- | ------------------- | ----------------------------- | ---------------------------- | -------------------------- | ----------------------------- | ---------------------------- | ------------------------ | --------------------------- | -------------- | ---------------------------------- | ----------------------------- | ---------------------------- | --------------------------- | --------------------------- | ---------------------------- | -------------------------- | --------- |
| 9830366 | testing | GGAAAUUUGC... | .......(((...       | [0.4167, 1.5941, 1.2359, ...] | [0.1689, 0.2323, 0.193, ...] | 5.326                      | [1.5966, 2.6482, 1.3761, ...] | [0.3058, 0.3294, 0.233, ...] | 4.198                    | [0.7885, 1.93, 2.0423, ...] |                | 3.746 [0.2773, 0.328, 0.3048, ...] | [1.5966, 2.6482, 1.3761, ...] | [0.3058, 0.3294, 0.233, ...] | 4.198                       | [0.7885, 1.93, 2.0423, ...] | [0.2773, 0.328, 0.3048, ...] | 3.746                      | True      |

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
    A list of floating-point values that provide an estimate of the likelihood of the RNA backbone being cut at each nucleotide position.
    These values help determine the stability of the RNA structure under various experimental conditions.

- **deg_pH10** and **deg_Mg_pH10**:
    Arrays of degradation rates observed under two conditions: incubation at pH 10 without and with magnesium, respectively.
    These values provide insight into how different conditions affect the stability of RNA molecules.

- **deg_50C** and **deg_Mg_50C**:
    Arrays of degradation rates after incubation at 50°C, without and with magnesium.
    These values capture how RNA sequences respond to elevated temperatures, which is relevant for storage and transportation conditions.

- **\*\_error\_\* Columns**:
    Arrays of floating-point numbers indicating the experimental errors corresponding to the measurements in the **reactivity** and **deg\_** columns.
    These values help quantify the uncertainty in the degradation rates and reactivity measurements.

- **SN_filter**:
    A filter applied to the dataset based on the signal-to-noise ratio, indicating whether a specific sequence meets the dataset’s quality criteria.

    If the SN_filter is `True`, the sequence meets the quality criteria; otherwise, it does not.

Note that due to technical limitations, the ground truth measurements are not available for the final bases of each RNA sequence, resulting in a shorter length for the provided labels compared to the full sequence.

## Variations

This dataset is available in two subsets:

- [RYOS-1](https://huggingface.co/datasets/multimolecule/ryos-1): The RYOS dataset from round 1 of the Eterna RYOS lab. The sequence length for RYOS-1 is 107, and the label length is 68.
- [RYOS-2](https://huggingface.co/datasets/multimolecule/ryos-2): The RYOS dataset from round 2 of the Eterna RYOS lab. The sequence length for RYOS-2 is 130, and the label length is 102.

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Citation

```bibtex
@article{waymentsteele2021deep,
  author  = {Wayment-Steele, Hannah K and Kladwang, Wipapat and Watkins, Andrew M and Kim, Do Soon and Tunguz, Bojan and Reade, Walter and Demkin, Maggie and Romano, Jonathan and Wellington-Oguri, Roger and Nicol, John J and Gao, Jiayang and Onodera, Kazuki and Fujikawa, Kazuki and Mao, Hanfei and Vandewiele, Gilles and Tinti, Michele and Steenwinckel, Bram and Ito, Takuya and Noumi, Taiga and He, Shujun and Ishi, Keiichiro and Lee, Youhan and {\"O}zt{\"u}rk, Fatih and Chiu, Anthony and {\"O}zt{\"u}rk, Emin and Amer, Karim and Fares, Mohamed and Participants, Eterna and Das, Rhiju},
  journal = {ArXiv},
  month   = oct,
  title   = {Deep learning models for predicting {RNA} degradation via dual crowdsourcing},
  year    = 2021
}
```
