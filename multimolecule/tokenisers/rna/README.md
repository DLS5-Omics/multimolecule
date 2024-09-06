---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# RnaTokenizer

RnaTokenizer is smart, it tokenizes raw RNA nucleotides into tokens, no matter if the input is in uppercase or lowercase, uses U (Uracil) or U (Thymine), and with or without special tokens.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, `RnaTokenizer` uses the [standard alphabet](#standard-alphabet).
If `nmers` is greater than `1`, or `codon` is set to `True`, it will instead use the [streamline alphabet](#streamline-alphabet).
