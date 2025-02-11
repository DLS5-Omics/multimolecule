---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - multimolecule/rnacentral
  - multimolecule/rfam
  - multimolecule/ensembl-genome-browser
  - multimolecule/nucleotide
library_name: multimolecule
pipeline_tag: fill-mask
mask_token: "<mask>"
---

> [!IMPORTANT]
> This model is in a future release of MultiMolecule, and is under development.
> This model card is not final and will be updated in the future.

# RibonanzaNet

Pre-trained model on RNA chemical mapping for modeling RNA structure and other properties.

## Disclaimer

This is an UNOFFICIAL implementation of the [Ribonanza: deep learning of RNA structure through dual crowdsourcing](https://doi.org/10.1101/2024.02.24.581671) by Shujun He, Rui Huang, et al.

The OFFICIAL repository of RibonanzaNet is at [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet).

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet applied `dropout-residual-norm` path twice to the output of the Self-Attention layer.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_norm=True` in the model configuration to apply the `dropout-residual-norm` path once.
>
> See more at [issue #3](https://github.com/Shujun-He/RibonanzaNet/issues/3)

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet does not apply attention mask correctly.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_attention_mask=True` in the model configuration to apply the correct attention mask.
>
> See more at [issue #4](https://github.com/Shujun-He/RibonanzaNet/issues/4), [issue #5](https://github.com/Shujun-He/RibonanzaNet/issues/5), and [issue #7](https://github.com/Shujun-He/RibonanzaNet/issues/7)

> [!CAUTION]
> The MultiMolecule team is aware of a potential risk in reproducing the results of RibonanzaNet.
>
> The original implementation of RibonanzaNet applies dropout in an axis different from the one described in the paper.
>
> By default, the MultiMolecule follows the original implementation.
>
> You can set `fix_pairwise_dropout=True` in the model configuration to follow the description in the paper.
>
> See more at [issue #6](https://github.com/Shujun-He/RibonanzaNet/issues/6)

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing RibonanzaNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

RibonanzaNet is a [bert](https://huggingface.co/google-bert/bert-base-uncased)-style model.

### Links

- **Code**: [multimolecule.ribonanzanet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ribonanzanet)
- **Weights**: [`multimolecule/ribonanzanet`](https://huggingface.co/multimolecule/ribonanzanet)
- **Data**: [RNAcentral](https://rnacentral.org)
- **Paper**: [Ribonanza: deep learning of RNA structure through dual crowdsourcing](https://doi.org/10.1101/2024.02.24.581671)
- **Developed by**: Shujun He, Rui Huang, Jill Townley, Rachael C. Kretsch, Thomas G. Karagianes, David B.T. Cox, Hamish Blair, Dmitry Penzar, Valeriy Vyaltsev, Elizaveta Aristova, Arsenii Zinkevich, Artemy Bakulin, Hoyeol Sohn, Daniel Krstevski, Takaaki Fukui, Fumiya Tatematsu, Yusuke Uchida, Donghoon Jang, Jun Seong Lee, Roger Shieh, Tom Ma, Eduard Martynov, Maxim V. Shugaev, Habib S.T. Bukhari, Kazuki Fujikawa, Kazuki Onodera, Christof Henkel, Shlomo Ron, Jonathan Romano, John J. Nicol, Grace P. Nye, Yuan Wu, Christian Choe, Walter Reade, Eterna participants, Rhiju Das
- **Model type**: [BERT](https://huggingface.co/google-bert/bert-base-uncased)
- **Original Repository**: [Shujun-He/RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet)

## Citation

**BibTeX**:

```bibtex
@article{He2024.02.24.581671,
  author       = {He, Shujun and Huang, Rui and Townley, Jill and Kretsch, Rachael C. and Karagianes, Thomas G. and Cox, David B.T. and Blair, Hamish and Penzar, Dmitry and Vyaltsev, Valeriy and Aristova, Elizaveta and Zinkevich, Arsenii and Bakulin, Artemy and Sohn, Hoyeol and Krstevski, Daniel and Fukui, Takaaki and Tatematsu, Fumiya and Uchida, Yusuke and Jang, Donghoon and Lee, Jun Seong and Shieh, Roger and Ma, Tom and Martynov, Eduard and Shugaev, Maxim V. and Bukhari, Habib S.T. and Fujikawa, Kazuki and Onodera, Kazuki and Henkel, Christof and Ron, Shlomo and Romano, Jonathan and Nicol, John J. and Nye, Grace P. and Wu, Yuan and Choe, Christian and Reade, Walter and Eterna participants and Das, Rhiju},
  title        = {Ribonanza: deep learning of RNA structure through dual crowdsourcing},
  elocation-id = {2024.02.24.581671},
  year         = {2024},
  doi          = {10.1101/2024.02.24.581671},
  publisher    = {Cold Spring Harbor Laboratory},
  abstract     = {Prediction of RNA structure from sequence remains an unsolved problem, and progress has been slowed by a paucity of experimental data. Here, we present Ribonanza, a dataset of chemical mapping measurements on two million diverse RNA sequences collected through Eterna and other crowdsourced initiatives. Ribonanza measurements enabled solicitation, training, and prospective evaluation of diverse deep neural networks through a Kaggle challenge, followed by distillation into a single, self-contained model called RibonanzaNet. When fine tuned on auxiliary datasets, RibonanzaNet achieves state-of-the-art performance in modeling experimental sequence dropout, RNA hydrolytic degradation, and RNA secondary structure, with implications for modeling RNA tertiary structure.Competing Interest StatementStanford University is filing patent applications based on concepts described in this paper. R.D. is a cofounder of Inceptive.},
  url          = {https://www.biorxiv.org/content/early/2024/06/11/2024.02.24.581671},
  eprint       = {https://www.biorxiv.org/content/early/2024/06/11/2024.02.24.581671.full.pdf},
  journal      = {bioRxiv}
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [RibonanzaNet paper](https://doi.org/10.1101/2024.02.24.581671) for questions or comments on the paper/model.

## License

This model is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
