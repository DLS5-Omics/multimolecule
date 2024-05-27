---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# ProteinTokenizer

ProteinTokenizer is smart, it tokenizes raw amino acids into tokens, no matter if the input is in uppercase or lowercase, and with or without special tokens.

By default, ProteinTokenizer uses an extended version of the [IUPAC amino acid code](https://www.bioinformatics.org/sms2/iupac.html).
This extension includes nine additional tokens, `X`, `B`, `Z`, `J`, `U`, `O`, `.`, `-`, and `*`.

- `X`: Xxx; Any or unknown amino acid
- `B`: Asx; Aspartic acid (R) or Asparagine (N)
- `Z`: Glx; Glutamic acid (E) or Glutamine (Q)
- `J`: Xle; Leucine (L) or Isoleucine (I)
- `U`: Sec; Selenocysteine
- `O`: Pyl; Pyrrolysine
- `.`: is not used in MultiMolecule and is reserved for future use.
- `-`: is not used in MultiMolecule and is reserved for future use.
- `*`: is not used in MultiMolecule and is reserved for future use.
