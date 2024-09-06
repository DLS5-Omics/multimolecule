---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# DotBracketTokenizer

DotBracketTokenizer provides a simple way to tokenize secondary structure in dot-bracket notation.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, `DotBracketTokenizer` uses the [standard alphabet](#standard-alphabet).
If `nmers` is greater than `1`, or `codon` is set to `True`, it will instead use the [streamline alphabet](#streamline-alphabet).
