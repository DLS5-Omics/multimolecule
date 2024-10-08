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
pretty_name: EternaBench-Switch
library_name: multimolecule
---

# EternaBench-Switch

![EternaBench-Switch](https://eternagame.org/sites/default/files/thumb_eternabench_paper.png)

EternaBench-Switch is a synthetic RNA dataset consisting of 7,228 riboswitch constructs, designed to explore the structural behavior of RNA molecules that change conformation upon binding to ligands such as FMN, theophylline, or tryptophan.
These riboswitches exhibit different structural states in the presence or absence of their ligands, and the dataset includes detailed measurements of binding affinities (dissociation constants), activation ratios, and RNA folding properties.

## Disclaimer

This is an UNOFFICIAL release of the [EternaBench-Switch](https://github.com/eternagame/EternaBench) by Hannah K. Wayment-Steele, et al.

**The team releasing EternaBench-Switch did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/eternabench_switch
- **datasets**: https://huggingface.co/datasets/multimolecule/eternabench-switch
- **Point of Contact**: [Rhiju Das](https://biochemistry.stanford.edu/people/rhiju-das/)

The dataset includes synthetic RNA sequences designed to act as riboswitches. These molecules can adopt different structural states in response to ligand binding, and the dataset provides detailed information on the binding affinities for various ligands, along with metrics on the RNA’s ability to switch between conformations. With over 7,000 entries, this dataset is highly useful for studying RNA folding, ligand interaction, and RNA structural dynamics.

## Example Entry

| id  | design | sequence           | activation_ratio | ligand | switch | kd_off  | kd_on  | kd_fmn | kd_no_fmn | min_kd_val | ms2_aptamer                   | lig_aptamer        | ms2_lig_aptamer    | log_kd_nolig | log_kd_lig | log_kd_nolig_scaled | log_kd_lig_scaled | log_AR | folding_subscore | num_clusters |
| --- | ------ | ------------------ | ---------------- | ------ | ------ | ------- | ------ | ------ | --------- | ---------- | ----------------------------- | ------------------ | ------------------ | ------------ | ---------- | ------------------- | ----------------- | ------ | ---------------- | ------------ |
| 286 | null   | AGGAAACAUGAGGAU... | 0.8824621522     | FMN    | OFF    | 13.3115 | 15.084 | null   | null      | 3.0082     | .....(((((x((xxxx)))))))..... | .................. | .....(((((x((xx... | 2.7137       | 2.5886     | 1.6123              | 1.4873            | -0.125 | null             | null         |

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

- **activation_ratio**:
    The ratio reflecting the RNA molecule’s structural change between two states (e.g., ON and OFF) upon ligand binding.

- **ligand**:
    The small molecule ligand (e.g., FMN, theophylline) that the RNA is designed to bind to, inducing the switch.

- **switch**:
    A binary or categorical value indicating whether the RNA demonstrates switching behavior.

- **kd_off**:
    The dissociation constant (KD) when the RNA is in the "OFF" state (without ligand), representing its binding affinity.

- **kd_on**:
    The dissociation constant (KD) when the RNA is in the "ON" state (with ligand), indicating its affinity after activation.

- **kd_fmn**:
    The dissociation constant for the RNA binding to the FMN ligand.

- **kd_no_fmn**:
    The dissociation constant when no FMN ligand is present, indicating the RNA's behavior in a ligand-free state.

- **min_kd_val**:
    The minimum KD value observed across different ligand-binding conditions.

- **ms2_aptamer**:
    Indicates whether the RNA contains an MS2 aptamer, a motif that binds the MS2 viral coat protein.

- **lig_aptamer**:
    A flag showing the presence of an aptamer that binds the ligand (e.g., FMN), demonstrating ligand-specific binding capability.

- **ms2_lig_aptamer**:
    Indicates if the RNA contains both an MS2 aptamer and a ligand-binding aptamer, potentially allowing for multifaceted binding behavior.

- **log_kd_nolig**:
    The logarithmic value of the dissociation constant without the ligand.

- **log_kd_lig**:
    The logarithmic value of the dissociation constant with the ligand present.

- **log_kd_nolig_scaled**:
    A normalized and scaled version of **log_kd_nolig** for easier comparison across conditions.

- **log_kd_lig_scaled**:
    A normalized and scaled version of **log_kd_lig** for consistency in data comparisons.

- **log_AR**:
    The logarithmic scale of the activation ratio, offering a standardized measure of activation strength.

- **folding_subscore**:
    A numerical score indicating how well the RNA molecule folds into the predicted structure.

- **num_clusters**:
    The number of distinct structural clusters or conformations predicted for the RNA, reflecting the complexity of the folding landscape.

## Related Datasets

- [eternabench-cm](https://huggingface.co/datasets/multimolecule/eternabench-cm)
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
