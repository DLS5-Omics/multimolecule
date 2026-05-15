---
language: rna
tags:
  - Biology
  - RNA
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

The original release does not define train, validation, and test partitions.
The converted dataset follows the paper's benchmark blocks as four splits:

- `bprna_1m`: single-sequence structures from bpRNA-1m.
- `rfam_14_5`: single-sequence structures from Rfam 14.5.
- `rfam_14_5_ref`: Rfam 14.5 reference alignments for common-structure prediction.
- `rfam_14_5_mafft`: Rfam 14.5 MAFFT alignments for common-structure prediction.

For alignment entries, `sequence` and `secondary_structure` are taken from the corresponding single-sequence BPSEQ file, and the full alignment is stored in `aligned_ids` and `aligned_sequences`.

## Schema

| Column | Description |
| --- | --- |
| `id` | Identifier of the benchmark entry. |
| `sequence` | Ungapped RNA sequence for the benchmark target. |
| `secondary_structure` | Target secondary structure in dot-bracket notation. |
| `aligned_ids` | Sequence IDs in the alignment. Single-sequence entries contain only their own ID. |
| `aligned_sequences` | Aligned sequences, preserving gap characters. Single-sequence entries contain only the ungapped sequence. |

## Disclaimer

This is an UNOFFICIAL release of the IPknot++ benchmark dataset by Kengo Sato and Yuki Kato.

**The team releasing IPknot++ did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://www.sato-lab.org/en/publication/sato-2022-qk/
- **datasets**: https://huggingface.co/datasets/multimolecule/ipknot-plus-plus
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
