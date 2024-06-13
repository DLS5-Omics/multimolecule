---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# DnaTokenizer

DnaTokenizer is smart, it tokenizes raw DNA nucleotides into tokens, no matter if the input is in uppercase or lowercase, uses T (Thymine) or U (Uracil), and with or without special tokens.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, `DnaTokenizer` uses the [standard alphabet](alphabet#standard-alphabet).
If `kmers` is greater than `1`, or `codon` is set to `True`, it will instead use the [streamline alphabet](alphabet#streamline-alphabet).
