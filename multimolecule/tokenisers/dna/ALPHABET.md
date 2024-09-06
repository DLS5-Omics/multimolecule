---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Alphabet

MultiMolecule provides a set of predefined alphabets for tokenization.

## Standard Alphabet

The standard alphabet is an extended version of the [IUPAC alphabet](#iupac-alphabet).
This extension includes two additional symbols to the [IUPAC alphabet](#iupac-alphabet), `X` and `*`.

- `X`: Any base; is slightly different from `N` which represents Unknown base.
  In automatic word embedding conversion, the `X` will be initialized as the mean of `A`, `C`, `G`, and `T`, while `N` will not be further processed.
- `*`: is not used in MultiMolecule and is reserved for future use.

!!! tip "gap"

    Note that we use `.` to represent a gap in the sequence.

    While `-` exists in the standard alphabet, it is not used in MultiMolecule and is reserved for future use.

| Code | Represents |
| ---- | ---------- |
| A    | Adenine    |
| C    | Cytosine   |
| G    | Guanine    |
| T    | Thymine    |
| N    | Unknown    |
| R    | A or G     |
| Y    | C or T     |
| S    | C or G     |
| W    | A or T     |
| K    | G or T     |
| M    | A or C     |
| B    | C, G, or T |
| D    | A, G, or T |
| H    | A, C, or T |
| V    | A, C, or G |
| .    | Gap        |
| X    | Any        |
| \*   | Not Used   |
| -    | Not Used   |

## IUPAC Alphabet

[IUPAC nucleotide code](https://www.bioinformatics.org/sms2/iupac.html) is a standard nucleotide code proposed by the International Union of Pure and Applied Chemistry (IUPAC) to represent DNA sequences.

It consists of 10 symbols that represent ambiguity in the nucleotide sequence and 1 symbol that represents a gap in addition to the [streamline alphabet](#streamline-alphabet).

| Code | Represents    |
| ---- | ------------- |
| A    | Adenine       |
| C    | Cytosine      |
| G    | Guanine       |
| T    | Thymine       |
| R    | A or G        |
| Y    | C or T        |
| S    | C or G        |
| W    | A or T        |
| K    | G or T        |
| M    | A or C        |
| B    | C, G, or T    |
| D    | A, G, or T    |
| H    | A, C, or T    |
| V    | A, C, or G    |
| N    | A, C, G, or T |
| .    | Gap           |

Note that we use `.` to represent a gap in the sequence.

## Streamline Alphabet

The streamline alphabet includes one additional symbol to the [nucleobase alphabet](#nucleobase-alphabet), `N` to represent unknown nucleobase.

| Code | Nucleotide |
| ---- | ---------- |
| A    | Adenine    |
| C    | Cytosine   |
| G    | Guanine    |
| T    | Thymine    |
| N    | Unknown    |

## Nucleobase Alphabet

The nucleobase alphabet is a minimal version of the DNA alphabet that includes only the four canonical nucleotides `A`, `C`, `G`, and `T`.

| Code | Nucleotide |
| ---- | ---------- |
| A    | Adenine    |
| C    | Cytosine   |
| G    | Guanine    |
| T    | Thymine    |
