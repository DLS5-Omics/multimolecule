---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# IO

`multimolecule.io` provides lightweight readers and writers for sequence and RNA secondary-structure files used by MultiMolecule datasets, pipelines, and Spaces.

| Format      | Extensions                                              | Record                        |
| ----------- | ------------------------------------------------------- | ----------------------------- |
| FASTA       | `.fa`, `.fas`, `.fasta`, `.fna`, `.ffn`, `.frn`, `.faa` | `SequenceRecord`              |
| Dot-bracket | `.db`, `.dbn`                                           | `RnaSecondaryStructureRecord` |
| BPSEQ       | `.bpseq`                                                | `RnaSecondaryStructureRecord` |
| bpRNA ST    | `.st`, `.sta`                                           | `BpRnaRecord`                 |
