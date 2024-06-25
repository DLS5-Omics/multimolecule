---
language: rna
tags:
  - Biology
  - RNA
license:
  - agpl-3.0
size_categories:
  - 10M<n<100M
task_categories:
  - text-generation
  - fill-mask
task_ids:
  - language-modeling
  - masked-language-modeling
pretty_name: Rfam
library_name: multimolecule
---

# Rfam

![Rfam](https://rfam.org/static/images/rfam_logo.svg)

Rfam is a database of structure-annotated multiple sequence alignments, covariance models and family annotation for a number of non-coding RNA, cis-regulatory and self-splicing intron families.

The seed alignments are hand curated and aligned using available sequence and structure data, and covariance models are built from these alignments using the [INFERNAL v1.1.4 software suite](http://infernal.janelia.org).

The full regions list is created by searching the RFAMSEQ database using the covariance model, and then listing all hits above a family specific threshold to the model.

Rfam is maintained by a consortium of researchers at the [European Bioinformatics Institute](http://www.ebi.ac.uk/), [Sean Eddy's laboratory](http://eddylab.org) and [Eric Nawrocki](https://github.com/nawrockie).

## Disclaimer

This is an UNOFFICIAL release of the [Rfam](https://rfam.org) by Ioanna Kalvari, Eric P. Nawrocki, Sarah W. Burge, Paul P Gardner, Sam Griffiths-Jones, et al.

**The team releasing Rfam did not write this dataset card for this dataset so this dataset card has been written by the MultiMolecule team.**

## Dataset Description

- **Homepage**: https://multimolecule.danling.org/datasets/rfam
- **Documentation**: https://docs.rfam.org/
- **datasets**: https://huggingface.co/datasets/multimolecule/rfam
- **Point of Contact**: [Blake Sweeney](https://www.ebi.ac.uk/people/person/blake-sweeney/)
- **Original URL**: https://rfam.org

## License

This dataset is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

> [!TIP]
> The original Rfam dataset is licensed under the [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license and is available at [Rfam](https://rfam.org).

## Citation

```bibtex
@article{kalvari2021rfam,
  author    = {Kalvari, Ioanna and Nawrocki, Eric P and Ontiveros-Palacios, Nancy and Argasinska, Joanna and Lamkiewicz, Kevin and Marz, Manja and Griffiths-Jones, Sam and Toffano-Nioche, Claire and Gautheret, Daniel and Weinberg, Zasha and Rivas, Elena and Eddy, Sean R and Finn, Robert D and Bateman, Alex and Petrov, Anton I},
  copyright = {http://creativecommons.org/licenses/by/4.0/},
  journal   = {Nucleic Acids Research},
  language  = {en},
  month     = jan,
  number    = {D1},
  pages     = {D192--D200},
  publisher = {Oxford University Press (OUP)},
  title     = {Rfam 14: expanded coverage of metagenomic, viral and {microRNA} families},
  volume    = 49,
  year      = 2021
}

@article{hufsky2021computational,
  author    = {Hufsky, Franziska and Lamkiewicz, Kevin and Almeida, Alexandre and Aouacheria, Abdel and Arighi, Cecilia and Bateman, Alex and Baumbach, Jan and Beerenwinkel, Niko and Brandt, Christian and Cacciabue, Marco and Chuguransky, Sara and Drechsel, Oliver and Finn, Robert D and Fritz, Adrian and Fuchs, Stephan and Hattab, Georges and Hauschild, Anne-Christin and Heider, Dominik and Hoffmann, Marie and H{\"o}lzer, Martin and Hoops, Stefan and Kaderali, Lars and Kalvari, Ioanna and von Kleist, Max and Kmiecinski, Ren{\'o} and K{\"u}hnert, Denise and Lasso, Gorka and Libin, Pieter and List, Markus and L{\"o}chel, Hannah F and Martin, Maria J and Martin, Roman and Matschinske, Julian and McHardy, Alice C and Mendes, Pedro and Mistry, Jaina and Navratil, Vincent and Nawrocki, Eric P and O'Toole, {\'A}ine Niamh and Ontiveros-Palacios, Nancy and Petrov, Anton I and Rangel-Pineros, Guillermo and Redaschi, Nicole and Reimering, Susanne and Reinert, Knut and Reyes, Alejandro and Richardson, Lorna and Robertson, David L and Sadegh, Sepideh and Singer, Joshua B and Theys, Kristof and Upton, Chris and Welzel, Marius and Williams, Lowri and Marz, Manja},
  copyright = {http://creativecommons.org/licenses/by/4.0/},
  journal   = {Briefings in Bioinformatics},
  month     = mar,
  number    = 2,
  pages     = {642--663},
  publisher = {Oxford University Press (OUP)},
  title     = {Computational strategies to combat {COVID-19}: useful tools to accelerate {SARS-CoV-2} and coronavirus research},
  volume    = 22,
  year      = 2021
}

@article{kalvari2018noncoding,
  author  = {Kalvari, Ioanna and Nawrocki, Eric P and Argasinska, Joanna and Quinones-Olvera, Natalia and Finn, Robert D and Bateman, Alex and Petrov, Anton I},
  journal = {Current Protocols in Bioinformatics},
  month   = jun,
  number  = 1,
  pages   = {e51},
  title   = {Non-coding {RNA} analysis using the rfam database},
  volume  = 62,
  year    = 2018
}

@article{kalvari2018rfam,
  author  = {Kalvari, Ioanna and Argasinska, Joanna and Quinones-Olvera,
             Natalia and Nawrocki, Eric P and Rivas, Elena and Eddy, Sean R
             and Bateman, Alex and Finn, Robert D and Petrov, Anton I},
  journal = {Nucleic Acids Research},
  month   = jan,
  number  = {D1},
  pages   = {D335--D342},
  title   = {Rfam 13.0: shifting to a genome-centric resource for non-coding {RNA} families},
  volume  = 46,
  year    = 2018
}

@article{nawrocki2015rfam,
  author    = {Nawrocki, Eric P and Burge, Sarah W and Bateman, Alex and Daub, Jennifer and Eberhardt, Ruth Y and Eddy, Sean R and Floden, Evan W and Gardner, Paul P and Jones, Thomas A and Tate, John and Finn, Robert D},
  copyright = {http://creativecommons.org/licenses/by/4.0/},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = {Database issue},
  pages     = {D130--7},
  publisher = {Oxford University Press (OUP)},
  title     = {Rfam 12.0: updates to the {RNA} families database},
  volume    = 43,
  year      = 2015
}

@article{burge2013rfam,
  author    = {Burge, Sarah W and Daub, Jennifer and Eberhardt, Ruth and Tate, John and Barquist, Lars and Nawrocki, Eric P and Eddy, Sean R and Gardner, Paul P and Bateman, Alex},
  copyright = {http://creativecommons.org/licenses/by-nc/3.0/},
  journal   = {Nucleic Acids Research},
  month     = jan,
  number    = {Database issue},
  pages     = {D226--32},
  publisher = {Oxford University Press (OUP)},
  title     = {Rfam 11.0: 10 years of {RNA} families},
  volume    = 41,
  year      = 2013
}

@article{gardner2011rfam,
  author  = {Gardner, Paul P and Daub, Jennifer and Tate, John and Moore, Benjamin L and Osuch, Isabelle H and Griffiths-Jones, Sam and Finn, Robert D and Nawrocki, Eric P and Kolbe, Diana L and Eddy, Sean R and Bateman, Alex},
  journal = {Nucleic Acids Research},
  month   = jan,
  number  = {Database issue},
  pages   = {D141--5},
  title   = {Rfam: Wikipedia, clans and the ``decimal'' release},
  volume  = 39,
  year    = 2011
}

@article{gardner2009rfam,
  author   = {Gardner, Paul P and Daub, Jennifer and Tate, John G and Nawrocki, Eric P and Kolbe, Diana L and Lindgreen, Stinus and Wilkinson, Adam C and Finn, Robert D and Griffiths-Jones, Sam and Eddy, Sean R and Bateman, Alex},
  journal  = {Nucleic Acids Research},
  month    = jan,
  number   = {Database issue},
  pages    = {D136--40},
  title    = {Rfam: updates to the {RNA} families database},
  volume   = 37,
  year     = 2009
}

@article{daub2008rna,
  author   = {Daub, Jennifer and Gardner, Paul P and Tate, John and Ramsk{\"o}ld, Daniel and Manske, Magnus and Scott, William G and Weinberg, Zasha and Griffiths-Jones, Sam and Bateman, Alex},
  journal  = {RNA},
  month    = dec,
  number   = 12,
  pages    = {2462--2464},
  title    = {The {RNA} {WikiProject}: community annotation of {RNA} families},
  volume   = 14,
  year     = 2008
}

@article{griffiths2005rfam,
  author   = {Griffiths-Jones, Sam and Moxon, Simon and Marshall, Mhairi and Khanna, Ajay and Eddy, Sean R. and Bateman, Alex},
  doi      = {10.1093/nar/gki081},
  eprint   = {https://academic.oup.com/nar/article-pdf/33/suppl\_1/D121/7622063/gki081.pdf},
  issn     = {0305-1048},
  journal  = {Nucleic Acids Research},
  month    = jan,
  number   = {suppl_1},
  pages    = {D121-D124},
  title    = {{Rfam: annotating non-coding RNAs in complete genomes}},
  url      = {https://doi.org/10.1093/nar/gki081},
  volume   = {33},
  year     = {2005}
}

@article{griffiths2003rfam,
  author   = {Griffiths-Jones, Sam and Bateman, Alex and Marshall, Mhairi and Khanna, Ajay and Eddy, Sean R.},
  doi      = {10.1093/nar/gkg006},
  eprint   = {https://academic.oup.com/nar/article-pdf/31/1/439/7125749/gkg006.pdf},
  issn     = {0305-1048},
  journal  = {Nucleic Acids Research},
  month    = jan,
  number   = {1},
  pages    = {439-441},
  title    = {{Rfam: an RNA family database}},
  url      = {https://doi.org/10.1093/nar/gkg006},
  volume   = {31},
  year     = {2003}
}
```
