---
authors:
  - zyc
date: 2024-05-04 00:00:00
---

# [MultiMolecule](https://multimolecule.danling.org)

## Introduction

Welcome to MultiMolecule (浦原), a foundational library designed to accelerate Scientific Research with Machine Learning. MultiMolecule aims to provide a comprehensive yet flexible set of tools for researchers who wish to leverage AI in their work.

We understand that AI4Science is a broad field, with researchers from different disciplines employing various practices. Therefore, MultiMolecule is designed with low coupling in mind, meaning that while it offers a full suite of functionalities, each module can be used independently. This allows you to integrate only the components you need into your existing workflows without adding unnecessary complexity. The key functionalities that MultiMolecule provides include:

- [`data`](data): Efficient data handling and preprocessing capabilities to streamline the ingestion and transformation of scientific datasets.
- [`datasets`](datasets): A collection of widely-used datasets across different scientific domains, providing a solid foundation for training and evaluation.
- [`module`](module): Modular components designed to provide flexibility and reusability across various machine learning tasks.
- [`models`](models): State-of-the-art model architectures optimized for scientific research applications, ensuring high performance and accuracy.
- [`tokenisers`](tokenisers): Advanced tokenization methods to effectively handle complex scientific text and data representations.
<!-- - [`utils`][multimolecule.utils]: A collection of utility functions and tools to simplify common tasks and enhance the overall user experience. -->

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

MultiMolecule is licensed under the GNU Affero General Public License.

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
