# [MultiMolecule](https://multimolecule.danling.org)

> [!TIP]
> 机器学习加速分子生物学研究

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119050.svg)](https://doi.org/10.5281/zenodo.15119050)

[![Codacy - 代码质量](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - 测试覆盖](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - 测试覆盖](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - 版本](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule)
[![PyPI - Python版本](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule)
[![下载统计](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![授权：AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## 介绍

欢迎来到 MultiMolecule (浦原)，这是一款基础库，旨在通过机器学习加速分子生物学的科研进展。
MultiMolecule 提供了一套全面且灵活的工具，帮助研究人员轻松利用 AI，主要聚焦于生物分子数据（RNA、DNA 和蛋白质）。

## 概览

MultiMolecule 以灵活性和易用性为设计核心。
其模块化设计允许您根据需要仅使用所需的组件，并能无缝集成到现有的工作流程中，而不会增加不必要的复杂性。

- [`data`](data)：智能的 [`Dataset`][multimolecule.data.Dataset]，能够自动推断任务，包括任务的层级（序列、令牌、接触）和类型（分类、回归）。还提供多任务数据集和采样器，简化多任务学习，无需额外配置。
- [`datasets`](datasets)：广泛使用的生物分子数据集集合。
- [`modules`](modules)：模块化神经网络构建块，包括[嵌入层](modules/embeddings)、[预测头](modules/heads)和损失函数，用于构建自定义模型。
- [`models`](models)：分子生物学领域的最先进预训练模型实现。
- [`tokenisers`](tokenisers)：用于将 DNA、RNA、蛋白质及其他序列转换为独热编码的分词器。
<!-- - [`runner`](runner)：功能强大且可扩展的运行器，允许用户轻松进行模型微调、评估和预测。 -->

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

> [!NOTE]
> 本仓库提供的内容是 MultiMolecule 项目的一部分。
> 如果你在你的研究中使用 MultiMolecule，你必须以如下方式引用 MultiMolecule。

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

MultiMolecule 在 [GNU Affero 通用公共许可证](license.md) 下授权。

对于额外条款和澄清，请参阅我们的[许可协议常见问题解答](license-faq.md)。

请加入我们，共同建立一个开放的研究社区。

`SPDX-License-Identifier: AGPL-3.0-or-later`
