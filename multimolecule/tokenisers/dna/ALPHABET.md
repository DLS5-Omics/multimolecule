---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Alphabet

MultiMolecule provides a set of predefined alphabets for tokenization.

## Standard Alphabet

The standard alphabet is an extended version of the [IUPAC alphabet](#iupac-alphabet).
This extension includes two additional nucleotide symbols to the [IUPAC alphabet](#iupac-alphabet), `I` and `X`,
and five sequence symbols, `|`, `.`, `*`, `-`, and `?`.

- `I`: Deoxyinosine; a non-canonical nucleotide, the product of adenine deamination.
- `X`: Unknown or other nucleotide; a nucleotide that cannot be mapped to any other symbol of the alphabet.
  It differs from `N`, which represents any one of `A`, `C`, `G`, and `T`.
- `|`: Chain separator.
- `-`: Gap; a deleted position in an aligned sequence.
- `.`: Gap in an insertion column of an alignment.
- `*`: Stop (terminator).
- `?`: Missing data; the content of an aligned position is not known.
  It differs from `-`, which states that the position is deleted.

`U` is not a symbol of the DNA alphabets: `DnaTokenizer` reads `U` as `T` by default (`replace_U_with_T=True`).

In automatic word embedding conversion, an ambiguity symbol (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) or `X`
that is absent from the original vocabulary is initialized as the mean of the nucleotides it represents; `X` uses `A`, `C`, `G`, and `T`.

| Code | Represents             |
| ---- | ---------------------- |
| A    | Adenine                |
| C    | Cytosine               |
| G    | Guanine                |
| T    | Thymine                |
| N    | Any (A, C, G, or T)    |
| R    | A or G                 |
| Y    | C or T                 |
| S    | C or G                 |
| W    | A or T                 |
| K    | G or T                 |
| M    | A or C                 |
| B    | C, G, or T             |
| D    | A, G, or T             |
| H    | A, C, or T             |
| V    | A, C, or G             |
| I    | Deoxyinosine           |
| X    | Unknown or other       |
| \|   | Chain separator        |
| .    | Gap (insertion column) |
| \*   | Stop                   |
| -    | Gap                    |
| ?    | Missing data           |

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

In the IUPAC alphabet, `.` represents a gap.

## Streamline Alphabet

The streamline alphabet includes one additional symbol to the [nucleobase alphabet](#nucleobase-alphabet), `N` to represent any nucleobase.

| Code | Nucleotide |
| ---- | ---------- |
| A    | Adenine    |
| C    | Cytosine   |
| G    | Guanine    |
| T    | Thymine    |
| N    | Any        |

## Nucleobase Alphabet

The nucleobase alphabet is a minimal version of the DNA alphabet that includes only the four canonical nucleotides `A`, `C`, `G`, and `T`.

| Code | Nucleotide |
| ---- | ---------- |
| A    | Adenine    |
| C    | Cytosine   |
| G    | Guanine    |
| T    | Thymine    |
