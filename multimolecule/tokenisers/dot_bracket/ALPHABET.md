---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Alphabet

MultiMolecule provides a set of predefined alphabets for tokenization.

## Standard Alphabet

The standard alphabet is an extended version of the Extended Dot-Bracket Notation.
This extension includes most symbols from the WUSS notation for better compatibility with existing tools.

| Code | Represents                                                            |
| ---- | --------------------------------------------------------------------- |
| .    | unpaired                                                              |
| (    | internal helices of all terminal stems                                |
| )    | internal helices of all terminal stems                                |
| +    | nick between strand                                                   |
| ,    | unpaired in multibranch loops                                         |
| [    | internal helices that includes at least one annotated () stem         |
| ]    | internal helices that includes at least one annotated () stem         |
| {    | all internal helices of deeper multifurcations                        |
| }    | all internal helices of deeper multifurcations                        |
| \|   | mostly paired                                                         |
| <    | simple terminal stems                                                 |
| >    | simple terminal stems                                                 |
| -    | bulges and interior loops                                             |
| \_   | unpaired                                                              |
| :    | single stranded in the exterior loop                                  |
| ~    | local structural alignment left regions of target and query unaligned |
| $    | Not Used                                                              |
| @    | Not Used                                                              |
| ^    | Not Used                                                              |
| %    | Not Used                                                              |
| \*   | Not Used                                                              |

## Extended Alphabet

[Extended Dot-Bracket Notation](https://viennarna.readthedocs.io/en/latest/io/rna_structures.html#extended-dot-bracket-notation) is a more generalized version of the original Dot-Bracket notation may use additional pairs of brackets for annotating pseudo-knots, since different pairs of brackets are not required to be nested.

| Code | Represents                                                    |
| ---- | ------------------------------------------------------------- |
| .    | unpaired                                                      |
| (    | internal helices of all terminal stems                        |
| )    | internal helices of all terminal stems                        |
| +    | nick between strand                                           |
| ,    | unpaired in multibranch loops                                 |
| [    | internal helices that includes at least one annotated () stem |
| ]    | internal helices that includes at least one annotated () stem |
| {    | all internal helices of deeper multifurcations                |
| }    | all internal helices of deeper multifurcations                |
| \|   | mostly paired                                                 |
| <    | simple terminal stems                                         |
| >    | simple terminal stems                                         |

Note that we use `.` to represent a gap in the sequence.

## Streamline Alphabet

The streamline alphabet includes one additional symbol to the [nucleobase alphabet](#nucleobase-alphabet), `N` to represent unknown nucleobase.

| Code | Represents                             |
| ---- | -------------------------------------- |
| .    | unpaired                               |
| (    | internal helices of all terminal stems |
| )    | internal helices of all terminal stems |
| +    | nick between strand                    |
