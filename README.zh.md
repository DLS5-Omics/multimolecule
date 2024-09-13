---
authors:
  - zyc
date: 2024-05-04 00:00:00
---

# [MultiMolecule](https://multimolecule.danling.org)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - Version](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule/)
[![Downloads](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 介绍

欢迎来到 MultiMolecule (浦原)，这是一个通过机器学习加速科学研究的基础库。MultiMolecule 旨在为希望在工作当中使用 AI 的研究人员提供一套全面而灵活的工具。

我们理解 AI4Science 是一个广泛的领域，来自不同学科的研究人员使用各种实践方法。因此，MultiMolecule 设计时考虑了低耦合性，这意味着虽然它提供了完整的功能套件，但每个模块都可以独立使用。这使您可以仅将所需组件集成到现有工作流程中，而不会增加不必要的复杂性。MultiMolecule 提供的主要功能包括：

- [`data`](data): 高效的数据处理和预处理功能，以简化科学数据集的摄取和转换。
- [`datasets`](datasets): 跨不同科学领域的广泛使用数据集集合，为训练和评估提供坚实基础。
- [`module`](module): 旨在提供灵活性和可重用性的模块化组件，适用于各种机器学习任务。
- [`models`](models): 为科学研究应用优化的最先进模型架构，确保高性能和高准确性。
- [`tokenisers`](tokenisers): 先进的分词方法，有效处理复杂的科学文本和数据表示。
<!-- - [`utils`][multimolecule.utils]: 一系列实用函数和工具，简化常见任务并增强整体用户体验。 -->

## 安装

从 PyPI 安装最新的稳定版本：

```shell
pip install multimolecule
```

从源代码安装最新版本：

```shell
pip install git+https://github.com/DLS5-Omics/MultiMolecule
```

## 引用

如果您在研究中使用 MultiMolecule，请按照以下方式引用我们：

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

## 许可证

我们相信开放是研究的基础。

MultiMolecule 在 GNU Affero 通用公共许可证下授权。

请加入我们，共同建立一个开放的研究社区。

`SPDX-License-Identifier: AGPL-3.0-or-later`
