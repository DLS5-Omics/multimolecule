# [MultiMolecule](https://multimolecule.danling.org)

> [!TIP]
> 机器学习加速分子生物学研究。

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119050.svg)](https://doi.org/10.5281/zenodo.15119050)

[![Codacy - 代码质量](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - 测试覆盖](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - 测试覆盖](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - 版本](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule)
[![PyPI - Python 版本](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule)
[![下载统计](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![授权：AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

MultiMolecule 是面向分子机器学习的一站式生态。
它把数据集、模型实现、可复用的数据集与神经网络模块、基于 DanLing 的训练评估 runner，以及面向任务的推理 pipeline 串联起来，服务 RNA、DNA 和蛋白质相关工作流。

## 快速开始

从 PyPI 安装最新稳定版本：

```shell
pip install multimolecule
```

通过 Hugging Face `transformers` 接口运行已注册的 pipeline：

```python
import multimolecule  # 注册 MultiMolecule 模型和 pipeline
from transformers import pipeline

predictor = pipeline("rna-secondary-structure", model="multimolecule/ernierna-ss")
result = predictor("AUCAGCCUUCGUUCUGUAAACGG")
```

需要更底层控制时，可以直接加载模型：

```python
import multimolecule

model = multimolecule.AutoModelForSequencePrediction.from_pretrained("multimolecule/basset")
tokenizer = multimolecule.AutoTokenizer.from_pretrained("multimolecule/basset")
```

如果需要未发布的最新修改，可以从源代码安装：

```shell
pip install git+https://github.com/DLS5-Omics/MultiMolecule
```

## 浏览

| 入口 | 用途 |
| --- | --- |
| [`data`](data) | 感知任务类型的数据集、数据加载和多任务采样。 |
| [`datasets`](datasets) | 生物分子数据集与任务元数据。 |
| [`io`](io) | FASTA、DBN、BPSEQ 和 bpRNA ST 读写。 |
| [`models`](models) | 支持模型的模型卡与 API 参考。 |
| [`tokenisers`](tokenisers) | DNA、RNA、蛋白质和 dot-bracket tokeniser。 |
| [`pipelines`](pipelines) | 面向具体生物任务的推理流程。 |
| [`runner`](runner) | 训练、评估和推理配置。 |
| [`modules`](modules) | 可复用的神经网络构建模块。 |

## 社区

- [Discourse](https://multimolecule.discourse.group)：发布公告、使用问题、模型请求、RFC 和社区讨论。
- [GitHub Issues](https://github.com/DLS5-Omics/multimolecule/issues)：可复现的错误、API 问题和需要工程跟踪的功能请求。
- [Hugging Face](https://huggingface.co/multimolecule)：已发布的 checkpoint、数据集和演示 Space。

## 引用

> [!NOTE]
> 本仓库提供的内容是 MultiMolecule 项目的一部分。
> 如果 MultiMolecule 对你的研究有帮助，请按如下方式引用 MultiMolecule。

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
