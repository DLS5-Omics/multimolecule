---
language: rna
tags:
  - Biology
  - RNA
  - Secondary Structures
license:
  - cc-by-4.0
source_datasets:
  - multimolecule/bprna
  - multimolecule/rfam
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: IPknot++
library_name: multimolecule
---

# IPknot++

IPknot++ is a benchmark dataset released with the IPknot++ RNA secondary structure prediction paper.
It is intended as an external evaluation set rather than a training split.
A common use is to train or tune RNA secondary structure prediction models on SPOT-RNA-style splits such as [bpRNA-spot](../bprna_spot), then evaluate generalization on IPknot++.

The original release does not define train and validation partitions, so every variant is released as a single `test` split.

IPknot++ is released as three repositories that share the same RNA targets but differ in whether and how a multiple sequence alignment is provided:

- `ipknot_plus_plus`: single-sequence structures with no alignment, covering both the bpRNA-1m and Rfam 14.5 targets.
- `ipknot_plus_plus-ref`: Rfam 14.5 targets with the Rfam reference alignment, for common-structure prediction.
- `ipknot_plus_plus-mafft`: Rfam 14.5 targets with a MAFFT alignment, for common-structure prediction.

The `-ref` and `-mafft` variants cover the same set of Rfam 14.5 targets, with identical `id`, `sequence`, and `secondary_structure`; only `aligned_sequences` differs (the alignment members in `aligned_ids` are also identical). These targets are a subset of the single sequences in the no-alignment `ipknot_plus_plus` variant.

For the alignment variants, `sequence` and `secondary_structure` are taken from the corresponding single-sequence BPSEQ file, and the full alignment is stored in `aligned_ids` and `aligned_sequences`.

## Schema

The `ipknot_plus_plus` variant contains:

| Column | Description |
| --- | --- |
| `id` | Identifier of the benchmark entry. |
| `sequence` | Ungapped RNA sequence for the benchmark target. |
| `secondary_structure` | Target secondary structure in dot-bracket notation. |

The `ipknot_plus_plus-ref` and `ipknot_plus_plus-mafft` variants add two alignment columns:

| Column | Description |
| --- | --- |
| `aligned_ids` | Sequence IDs in the alignment. The first entry is the benchmark target itself. |
| `aligned_sequences` | Aligned sequences, preserving gap characters. |

## Disclaimer

This is an UNOFFICIAL release of the IPknot++ benchmark dataset by Kengo Sato and Yuki Kato.

**The team releasing IPknot++ did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://www.sato-lab.org/en/publication/sato-2022-qk/
- **datasets**: https://huggingface.co/datasets/multimolecule/ipknot_plus_plus
- **Original URL**: https://zenodo.org/records/4923158

## License

The original IPknot++ benchmark dataset is licensed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license.

```spdx
SPDX-License-Identifier: CC-BY-4.0
```

## Citation

```bibtex
@article{sato2022prediction,
  author    = {Sato, Kengo and Kato, Yuki},
  journal   = {Briefings in Bioinformatics},
  month     = jan,
  number    = 1,
  title     = {Prediction of {RNA} secondary structure including pseudoknots for long sequences},
  volume    = 23,
  year      = 2022
}
```

```bibtex
@dataset{sato2021ipknot,
  author    = {Sato, Kengo and Kato, Yuki},
  title     = {{IPknot++} benchmark dataset},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4923158},
  year      = 2021
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If MultiMolecule supports your research, please cite the MultiMolecule project as follows:

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
