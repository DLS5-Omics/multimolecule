---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# RnaTokenizer

RnaTokenizer is smart, it tokenizes raw RNA nucleotides into tokens, no matter if the input is in uppercase or lowercase, uses U (Uracil) or T (Thymine), and with or without special tokens.
It also supports tokenization into nmers and codons, so you don't have to write complex code to preprocess your data.

By default, RnaTokenizer uses an extended version of the [IUPAC nucleotide code](https://www.bioinformatics.org/sms2/iupac.html).
This extension includes three additional tokens, `I`, `X` and `*`.

- `I` represent Inosine, which is a post-trancriptional modification that is not a standard RNA base.
  Inosine is the result of a deamination reaction of adenines that is catalyzed by adenosine deaminases acting on tRNAs (ADATs)
- `X`: represents Any base, it is slightly different from `N` which represents Unknown base.
  In automatic word embedding conversion, the `X` will be initialized as the mean of `A`, `C`, `G`, and `U`, while `N` will not be further processed.
- `*`: is not used in MultiMolecule and is reserved for future use.

If `kmers` is greater than `1`, or `codon` is set to `True`, the tokenizer will use a minimal alphabet that includes only the five canonical nucleotides `A`, `C`, `G`, `U`, and `N`.
