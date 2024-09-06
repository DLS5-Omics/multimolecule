---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Alphabet

MultiMolecule provides a set of predefined alphabets for tokenization.

## Standard Alphabet

The standard alphabet is an extended version of the [IUPAC alphabet](#iupac-alphabet).
This extension includes six additional symbols to the [IUPAC alphabet](#iupac-alphabet), `J`, `U`, `O`, `.`, `-`, and `*`.

- `J`: Xle; Leucine (L) or Isoleucine (I)
- `U`: Sec; Selenocysteine
- `O`: Pyl; Pyrrolysine
- `.`: is not used in MultiMolecule and is reserved for future use.
- `-`: is not used in MultiMolecule and is reserved for future use.
- `*`: is not used in MultiMolecule and is reserved for future use.

| Amino Acid Code | Three letter Code | Amino Acid                          |
| --------------- | ----------------- | ----------------------------------- |
| A               | Ala               | Alanine                             |
| C               | Cys               | Cysteine                            |
| D               | Asp               | Aspartic Acid                       |
| E               | Glu               | Glutamic Acid                       |
| F               | Phe               | Phenylalanine                       |
| G               | Gly               | Glycine                             |
| H               | His               | Histidine                           |
| I               | Ile               | Isoleucine                          |
| K               | Lys               | Lysine                              |
| L               | Leu               | Leucine                             |
| M               | Met               | Methionine                          |
| N               | Asn               | Asparagine                          |
| P               | Pro               | Proline                             |
| Q               | Gln               | Glutamine                           |
| R               | Arg               | Arginine                            |
| S               | Ser               | Serine                              |
| T               | Thr               | Threonine                           |
| V               | Val               | Valine                              |
| W               | Trp               | Tryptophan                          |
| Y               | Tyr               | Tyrosine                            |
| X               | Xaa               | Any amino acid                      |
| Z               | Glx               | Glutamine (Q) or Glutamic acid (E)  |
| B               | Asx               | Aspartic acid (D) or Asparagine (N) |
| J               | Xle               | Leucine (L) or Isoleucine (I)       |
| U               | Sec               | Selenocysteine                      |
| O               | Pyl               | Pyrrolysine                         |
| .               | ...               | Not Used                            |
| \*              | \*\*\*            | Not Used                            |
| -               | ---               | Not Used                            |

## IUPAC Alphabet

[IUPAC amino acid code](https://www.bioinformatics.org/sms2/iupac.html) is a standard amino acid code proposed by the International Union of Pure and Applied Chemistry (IUPAC) to represent Protein sequences.

The IUPAC amino acid code consists of three additional symbols to [Streamline Alphabet](#streamline-alphabet), `B`, `Z`, and `X`.

| Amino Acid Code | Three letter Code | Amino Acid                          |
| --------------- | ----------------- | ----------------------------------- |
| A               | Ala               | Alanine                             |
| B               | Asx               | Aspartic acid (D) or Asparagine (N) |
| C               | Cys               | Cysteine                            |
| D               | Asp               | Aspartic Acid                       |
| E               | Glu               | Glutamic Acid                       |
| F               | Phe               | Phenylalanine                       |
| G               | Gly               | Glycine                             |
| H               | His               | Histidine                           |
| I               | Ile               | Isoleucine                          |
| K               | Lys               | Lysine                              |
| L               | Leu               | Leucine                             |
| M               | Met               | Methionine                          |
| N               | Asn               | Asparagine                          |
| P               | Pro               | Proline                             |
| Q               | Gln               | Glutamine                           |
| R               | Arg               | Arginine                            |
| S               | Ser               | Serine                              |
| T               | Thr               | Threonine                           |
| V               | Val               | Valine                              |
| W               | Trp               | Tryptophan                          |
| Y               | Tyr               | Tyrosine                            |
| X               | Xaa               | Any amino acid                      |
| Z               | Glx               | Glutamine (Q) or Glutamic acid (E)  |

## Streamline Alphabet

The streamline alphabet is a simplified version of the standard alphabet.

| Amino Acid Code | Three letter Code | Amino Acid     |
| --------------- | ----------------- | -------------- |
| A               | Ala               | Alanine        |
| C               | Cys               | Cysteine       |
| D               | Asp               | Aspartic Acid  |
| E               | Glu               | Glutamic Acid  |
| F               | Phe               | Phenylalanine  |
| G               | Gly               | Glycine        |
| H               | His               | Histidine      |
| I               | Ile               | Isoleucine     |
| K               | Lys               | Lysine         |
| L               | Leu               | Leucine        |
| M               | Met               | Methionine     |
| N               | Asn               | Asparagine     |
| P               | Pro               | Proline        |
| Q               | Gln               | Glutamine      |
| R               | Arg               | Arginine       |
| S               | Ser               | Serine         |
| T               | Thr               | Threonine      |
| V               | Val               | Valine         |
| W               | Trp               | Tryptophan     |
| Y               | Tyr               | Tyrosine       |
| X               | Xaa               | Any amino acid |
