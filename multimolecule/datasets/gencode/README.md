---
language: dna
tags:
  - Biology
  - DNA
license:
  - agpl-3.0
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: GENCODE
library_name: multimolecule
---

# GENCODE

![GENCODE](https://www.gencodegenes.org/images/gencodegenes-logo.png)

GENCODE is a comprehensive annotation project that aims to provide high-quality annotations of the human and mouse genomes.
The project is part of the ENCODE (ENCyclopedia Of DNA Elements) scale-up project, which seeks to identify all functional elements in the human genome.

## Disclaimer

This is an UNOFFICIAL release of the [GENCODE](https://www.gencodegenes.org) by Paul Flicek, Roderic Guigo, Manolis Kellis, Mark Gerstein, Benedict Paten, Michael Tress, Jyoti Choudhary, et al.

**The team releasing GENCODE did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/gencode
- **Point of Contact**: [GENCODE](mailto:gencode-help@ebi.ac.uk)
- **Original URL**: https://www.gencodegenes.org

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Datasets

The GENCODE dataset is available in Human and Mouse:

- [gencode-human](https://huggingface.co/datasets/multimolecule/gencode-human): The GENCODE dataset for the human genome.
- [gencode-mouse](https://huggingface.co/datasets/multimolecule/gencode-mouse): The GENCODE dataset for the mouse genome.

## Citation

```bibtex
@article{frankish2023gencode,
  author    = {Frankish, Adam and Carbonell-Sala, S{\'\i}lvia and Diekhans, Mark and Jungreis, Irwin and Loveland, Jane E and Mudge, Jonathan M and Sisu, Cristina and Wright, James C and Arnan, Carme and Barnes, If and Banerjee, Abhimanyu and Bennett, Ruth and Berry, Andrew and Bignell, Alexandra and Boix, Carles and Calvet, Ferriol and Cerd{\'a}n-V{\'e}lez, Daniel and Cunningham, Fiona and Davidson, Claire and Donaldson, Sarah and Dursun, Cagatay and Fatima, Reham and Giorgetti, Stefano and Giron, Carlos Garc{\i}a and Gonzalez, Jose Manuel and Hardy, Matthew and Harrison, Peter W and Hourlier, Thibaut and Hollis, Zoe and Hunt, Toby and James, Benjamin and Jiang, Yunzhe and Johnson, Rory and Kay, Mike and Lagarde, Julien and Martin, Fergal J and G{\'o}mez, Laura Mart{\'\i}nez and Nair, Surag and Ni, Pengyu and Pozo, Fernando and Ramalingam, Vivek and Ruffier, Magali and Schmitt, Bianca M and Schreiber, Jacob M and Steed, Emily and Suner, Marie-Marthe and Sumathipala, Dulika and Sycheva, Irina and Uszczynska-Ratajczak, Barbara and Wass, Elizabeth and Yang, Yucheng T and Yates, Andrew and Zafrulla, Zahoor and Choudhary, Jyoti S and Gerstein, Mark and Guigo, Roderic and Hubbard, Tim J P and Kellis, Manolis and Kundaje, Anshul and Paten, Benedict and Tress, Michael L and Flicek, Paul},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = {D1},
  pages     = {D942--D949},
  publisher = {Oxford University Press (OUP)},
  title     = {{GENCODE}: reference annotation for the human and mouse genomes in 2023},
  volume    = 51,
  year      = 2023
}

@article{frankish2021gencode,
  author    = {Frankish, Adam and Diekhans, Mark and Jungreis, Irwin and Lagarde, Julien and Loveland, Jane E and Mudge, Jonathan M and Sisu, Cristina and Wright, James C and Armstrong, Joel and Barnes, If and Berry, Andrew and Bignell, Alexandra and Boix, Carles and Carbonell Sala, Silvia and Cunningham, Fiona and Di Domenico, Tom{\'a}s and Donaldson, Sarah and Fiddes, Ian T and Garc{\'\i}a Gir{\'o}n, Carlos and Gonzalez, Jose Manuel and Grego, Tiago and Hardy, Matthew and Hourlier, Thibaut and Howe, Kevin L and Hunt, Toby and Izuogu, Osagie G and Johnson, Rory and Martin, Fergal J and Mart{\'\i}nez, Laura and Mohanan, Shamika and Muir, Paul and Navarro, Fabio C P and Parker, Anne and Pei, Baikang and Pozo, Fernando and Riera, Ferriol Calvet and Ruffier, Magali and Schmitt, Bianca M and Stapleton, Eloise and Suner, Marie-Marthe and Sycheva, Irina and Uszczynska-Ratajczak, Barbara and Wolf, Maxim Y and Xu, Jinuri and Yang, Yucheng T and Yates, Andrew and Zerbino, Daniel and Zhang, Yan and Choudhary, Jyoti S and Gerstein, Mark and Guig{\'o}, Roderic and Hubbard, Tim J P and Kellis, Manolis and Paten, Benedict and Tress, Michael L and Flicek, Paul},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = {D1},
  pages     = {D916--D923},
  publisher = {Oxford University Press (OUP)},
  title     = {{GENCODE} 2021},
  volume    = 49,
  year      = 2021
}

@article{frankish2019gencode,
  author    = {Frankish, Adam and Diekhans, Mark and Ferreira, Anne-Maud and Johnson, Rory and Jungreis, Irwin and Loveland, Jane and Mudge, Jonathan M and Sisu, Cristina and Wright, James and Armstrong, Joel and Barnes, If and Berry, Andrew and Bignell, Alexandra and Carbonell Sala, Silvia and Chrast, Jacqueline and Cunningham, Fiona and Di Domenico, Tom{\'a}s and Donaldson, Sarah and Fiddes, Ian T and Garc{\'\i}a Gir{\'o}n, Carlos and Gonzalez, Jose Manuel and Grego, Tiago and Hardy, Matthew and Hourlier, Thibaut and Hunt, Toby and Izuogu, Osagie G and Lagarde, Julien and Martin, Fergal J and Mart{\'\i}nez, Laura and Mohanan, Shamika and Muir, Paul and Navarro, Fabio C P and Parker, Anne and Pei, Baikang and Pozo, Fernando and Ruffier, Magali and Schmitt, Bianca M and Stapleton, Eloise and Suner, Marie-Marthe and Sycheva, Irina and Uszczynska-Ratajczak, Barbara and Xu, Jinuri and Yates, Andrew and Zerbino, Daniel and Zhang, Yan and Aken, Bronwen and Choudhary, Jyoti S and Gerstein, Mark and Guig{\'o}, Roderic and Hubbard, Tim J P and Kellis, Manolis and Paten, Benedict and Reymond, Alexandre and Tress, Michael L and Flicek, Paul},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = {D1},
  pages     = {D766--D773},
  publisher = {Oxford University Press (OUP)},
  title     = {{GENCODE} reference annotation for the human and mouse genomes},
  volume    = 47,
  year      = 2019
}

@article{mudge2015creating,
  author    = {Mudge, Jonathan M and Harrow, Jennifer},
  copyright = {https://creativecommons.org/licenses/by/4.0},
  journal   = {Mamm. Genome},
  language  = {en},
  month     = oct,
  number    = {9-10},
  pages     = {366--378},
  publisher = {Springer Science and Business Media LLC},
  title     = {Creating reference gene annotation for the mouse {C57BL6/J} genome assembly},
  volume    = 26,
  year      = 2015
}

@article{harrow2012gencode,
  author   = {Harrow, Jennifer and Frankish, Adam and Gonzalez, Jose M and Tapanari, Electra and Diekhans, Mark and Kokocinski, Felix and Aken, Bronwen L and Barrell, Daniel and Zadissa, Amonida and Searle, Stephen and Barnes, If and Bignell, Alexandra and Boychenko, Veronika and Hunt, Toby and Kay, Mike and Mukherjee, Gaurab and Rajan, Jeena and Despacio-Reyes, Gloria and Saunders, Gary and Steward, Charles and Harte, Rachel and Lin, Michael and Howald, C{\'e}dric and Tanzer, Andrea and Derrien, Thomas and Chrast, Jacqueline and Walters, Nathalie and Balasubramanian, Suganthi and Pei, Baikang and Tress, Michael and Rodriguez, Jose Manuel and Ezkurdia, Iakes and van Baren, Jeltje and Brent, Michael and Haussler, David and Kellis, Manolis and Valencia, Alfonso and Reymond, Alexandre and Gerstein, Mark and Guig{\'o}, Roderic and Hubbard, Tim J},
  journal  = {Genome Research},
  month    = sep,
  number   = 9,
  pages    = {1760--1774},
  title    = {{GENCODE}: the reference human genome annotation for The {ENCODE} Project},
  volume   = 22,
  year     = 2012
}

@article{harrow2006gencode,
  author    = {Harrow, Jennifer and Denoeud, France and Frankish, Adam and  Reymond, Alexandre and Chen, Chao-Kung and Chrast, Jacqueline  and Lagarde, Julien and Gilbert, James G R and Storey, Roy and  Swarbreck, David and Rossier, Colette and Ucla, Catherine and  Hubbard, Tim and Antonarakis, Stylianos E and Guigo, Roderic},
  journal   = {Genome Biology},
  month     = aug,
  number    = {Suppl 1},
  pages     = {S4.1--9},
  publisher = {Springer Nature},
  title     = {{GENCODE}: producing a reference annotation for {ENCODE}},
  volume    = {7 Suppl 1},
  year      = 2006
}
```
