# [MultiMolecule](https://multimolecule.danling.org)

> [!TIP]
> Accelerate Molecular Biology Research with Machine Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119050.svg)](https://doi.org/10.5281/zenodo.15119050)

[![Codacy - Quality](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - Coverage](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - Coverage](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - Version](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule)
[![Downloads Statistics](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 🧬 Introduction

MultiMolecule is a framework that bridges molecular biology and machine learning. It offers machine learning tools specifically designed for biomolecular data (RNA, DNA, and protein).

MultiMolecule serves as a foundation for advancing research at the intersection of molecular biology and machine learning.

## 🚀 Features

### 📑 Resources

- **[Model Hub](https://multimolecule.danling.org/models)**: Models designed for biomolecular data.
- **[Dataset Hub](https://multimolecule.danling.org/datasets)**: Processed biomolecular datasets.

### 🛠️ Tools

- **[`pipelines`](pipelines)**: End-to-end workflows for applying models.
- **[`runner`](runner)**: Automatic Runner for training models.

### ⚙️ Infrastructure

- **[`data`](data)**: Smart [`Dataset`][multimolecule.data.Dataset] that automatically infer tasks—including their level (sequence, token, contact) and type (classification, regression).
- **[`tokenisers`](tokenisers)**: Tokenizers for biomolecular sequences.
- **[`module`](module)**: Neural network building blocks.

## 🔧 Installation

=== "Install the stable release from PyPI"

    ```bash
    pip install multimolecule
    ```

=== "Install the latest development version"

    ```bash
    pip install git+https://github.com/DLS5-Omics/multimolecule
    ```

## 📜 Citation

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

## 📄 License

We believe openness is the Foundation of Research.

MultiMolecule is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
