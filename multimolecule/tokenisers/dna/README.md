---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# DnaTokenizer

DnaTokenizer is smart, it tokenizes raw DNA nucleotides into tokens, no matter if the input is in uppercase or lowercase, uses T (Thymine) or U (Uracil), and with or without special tokens.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, DnaTokenizer uses an extended version of the [IUPAC nucleotide code](https://www.bioinformatics.org/sms2/iupac.html).
This extension includes two additional tokens, `X` and `*`.

- `X`: represents Any base, it is slightly different from `N` which represents Unknown base.
  In automatic word embedding conversion, the `X` will be initialized as the mean of `A`, `C`, `G`, and `T`, while `N` will not be further processed.
- `*`: is not used in MultiMolecule and is reserved for future use.

If `kmers` is greater than `1`, or `codon` is set to `True`, the tokenizer will use a minimal alphabet that includes only the five canonical nucleotides `A`, `C`, `G`, `T`, and `N`.
