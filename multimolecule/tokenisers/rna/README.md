---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# RnaTokenizer

RnaTokenizer is smart, it tokenizes raw RNA nucleotides into tokens, no matter if the input is in uppercase or lowercase, uses U (Uracil) or T (Thymine), and with or without special tokens.
By default, T is read as U; pass `replace_T_with_U=False` to disable this.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, `RnaTokenizer` uses the [standard alphabet](#standard-alphabet).
If `nmers` is greater than `1`, or `codon` is set to `True`, it will instead use the [streamline alphabet](#streamline-alphabet).
