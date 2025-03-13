# [MultiMolecule](https://multimolecule.danling.org)

> [!TIP]
> Accelerate Molecular Biology Research with Machine Learning

[![Codacy - 代码质量](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - Coverage](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - Coverage](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - Version](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule)
[![Downloads Statistics](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Introduction

Welcome to MultiMolecule (浦原), a foundational library designed to accelerate scientific research in molecular biology through machine learning.
MultiMolecule provides a comprehensive yet flexible set of tools for researchers aiming to leverage AI with ease, focusing on biomolecular data (RNA, DNA, and protein).

## Overview

MultiMolecule is built with flexibility and ease of use in mind.
Its modular design allows you to utilize only the components you need, integrating seamlessly into your existing workflows without adding unnecessary complexity.

- [`data`](data): Smart [`Dataset`][multimolecule.data.Dataset] that automatically infer tasks—including their level (sequence, token, contact) and type (classification, regression).
- [`datasets`](datasets): A collection of widely-used biomolecular datasets.
- [`module`](module): Modular neural network building blocks, including [embeddings](module/embeddings), [heads](module/heads), and criterions for constructing custom models.
- [`models`](models): Implementation of state-of-the-art pre-trained models in molecular biology.
- [`tokenisers`](tokenisers): Tokenizers to convert DNA, RNA, protein and other sequences to one-hot encodings.
<!-- - [`runner`](runner): A powerful and extensible runner allows users to fine-tune models, evaluate and predict with ease. -->

## Installation

Install the most recent stable version on PyPI:

```shell
pip install multimolecule
```

Install the latest version from the source:

```shell
pip install git+https://github.com/DLS5-Omics/MultiMolecule
```

## Citation

If you use MultiMolecule in your research, please cite us as follows:

```bibtex
@software{chen_2024_12638419,
  author    = {Chen, Zhiyuan and Zhu, Sophia Y.},
  title     = {MultiMolecule},
  doi       = {10.5281/zenodo.12638419},
  publisher = {Zenodo},
  url       = {https://doi.org/10.5281/zenodo.12638419},
  year      = 2024,
  month     = may,
  day       = 4
}
```

## License

We believe openness is the Foundation of Research.

MultiMolecule is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
