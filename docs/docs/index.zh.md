---
template: home.html
title: MultiMolecule
description: 面向端到端生物分子机器学习的模块化生态。
summary: 不作意 任自然 即此语 是灵丹
authors:
  - Zhiyuan Chen
date: 2022-05-04
hide:
  - toc
hero:
  eyebrow: 机器学习加速分子生物学研究
  title: MultiMolecule
  description: "MultiMolecule 为 RNA、DNA 和蛋白质序列研究提供可直接使用的 pipeline、预训练模型 checkpoint、整理好的数据集和训练工具。"
---

<section class="mm-section mm-section--paths" id="paths" markdown>
<div class="mm-section__inner" markdown>

## 你想完成什么？

从你要完成的任务开始：基于序列预测、用自己的数据微调、加载预训练模型，或使用整理好的数据集。

<div class="mm-workflow-screens" aria-label="MultiMolecule entry points" markdown>

<section class="mm-workflow-panel" id="tasks" markdown>
<div class="mm-workflow-copy" markdown>
<span>Prediction</span>

### 基于序列预测

已注册的 pipeline 将生物任务名称和输入序列转换成结构化预测，不需要手动组装模型、tokeniser 和后处理。
</div>

```python
import multimolecule
from transformers import pipeline

predict = pipeline(
    "rna-secondary-structure",
    model="multimolecule/ernierna-ss",
)
structure = predict("AUCAGCCUUCGUUCUGUAAACGG")
```
</section>

<section class="mm-workflow-panel" id="training" markdown>
<div class="mm-workflow-copy" markdown>
<span>Training</span>

### 用自己的数据微调

runner 将预训练 checkpoint 与 Hugging Face 数据集或本地标注表格连接起来，并根据序列列和标签列启动监督训练。
</div>

```python
import multimolecule as mm

config = mm.Config(
    pretrained="multimolecule/ernierna",
    data={
        "root": "multimolecule/chanrg",
        "feature_cols": ["sequence"],
        "label_cols": ["secondary_structure"],
    },
)

runner = mm.Runner(config)
runner.train()
```
</section>

<section class="mm-workflow-panel" id="models" markdown>
<div class="mm-workflow-copy" markdown>
<span>Models</span>

### 加载预训练模型

模型卡提供 checkpoint ID、输入约定、引用和许可证；Python API 支持在任务 pipeline 之外直接控制模型。
</div>

```python
import multimolecule as mm

tokenizer = mm.RnaTokenizer.from_pretrained("multimolecule/ernierna-ss")
model = mm.AutoModelForRnaSecondaryStructurePrediction.from_pretrained(
    "multimolecule/ernierna-ss",
)

inputs = tokenizer("AUCAGCCUUCGUUCUGUAAACGG", return_tensors="pt")
outputs = model(**inputs)
```
</section>

<section class="mm-workflow-panel" id="datasets" markdown>
<div class="mm-workflow-copy" markdown>
<span>Datasets</span>

### 使用整理好的数据集

整理好的生物数据集包含序列与标签字段、任务元数据、来源信息、引用和许可证，可用于基准、示例和微调。
</div>

```python
from datasets import load_dataset

dataset = load_dataset("multimolecule/chanrg", split="train")
```
</section>

</div>

<nav class="mm-workflow-dots" aria-label="首页入口面板">
<a class="mm-workflow-dot is-active" href="#tasks" aria-current="true" aria-label="显示预测入口"></a>
<a class="mm-workflow-dot" href="#training" aria-label="显示训练入口"></a>
<a class="mm-workflow-dot" href="#models" aria-label="显示模型入口"></a>
<a class="mm-workflow-dot" href="#datasets" aria-label="显示数据集入口"></a>
</nav>

</div>
</section>

<section class="mm-section mm-section--stack" markdown>
<div class="mm-section__inner" markdown>

## 底层是一套栈

当你需要更细的控制时，同一套生态也提供带文档的资源、生物输入处理、可复用模型组件，以及用于预测、训练、评估和脚本化使用的执行工具。

<div class="mm-stack-tower" markdown>

<div class="mm-stack-row mm-stack-row--top" markdown>
<section class="mm-stack-step mm-stack-step--execution" markdown>
<span>Execution</span>

__Pipeline、runner 和 API__

pipeline 提供现成任务预测，runner 管理监督训练和评估，API 入口支持脚本和应用。

<p class="mm-stack-links" markdown>[`runner`](runner/index.md) · [`pipelines`](pipelines/index.md) · [`api`](api/index.md)</p>
</section>
</div>

<div class="mm-stack-row mm-stack-row--middle" markdown>
<section class="mm-stack-step mm-stack-step--resources" markdown>
<span>Resources</span>

__带来源信息的模型和数据集__

数据集卡和模型卡整理支持的输入、任务名称、checkpoint ID、引用、许可证和训练元数据。

<p class="mm-stack-links" markdown>[`datasets`](datasets/index.md) · [`models`](models/index.md)</p>
</section>
</div>

<div class="mm-stack-row mm-stack-row--base" markdown>
<section class="mm-stack-step mm-stack-step--data" markdown>
<span>Data layer</span>

__从生物数据到模型输入__

IO、tokeniser 和数据工具将生物序列、结构和注释转换成一致的输入，用于 pipeline、训练和评估。

<p class="mm-stack-links" markdown>[`io`](io/index.md) · [`tokenisers`](tokenisers/index.md) · [`data`](data/index.md)</p>
</section>

<section class="mm-stack-step mm-stack-step--model" markdown>
<span>Model layer</span>

__可复用的模型构件__

models 提供预训练 config、AutoModel 类、checkpoint 和输出约定；modules 提供 backbone、head、loss 和 embedding，用于自定义架构。

<p class="mm-stack-links" markdown>[`models`](models/index.md) · [`modules`](modules/index.md)</p>
</section>
</div>
</div>

</div>
</section>

<section class="mm-section mm-section--tail" markdown>
<div class="mm-section__inner" markdown>

## 社区

<div class="grid cards mm-trust-grid" markdown>

-   __Google Group__

    接收发布公告、迁移说明和设计 RFC，不需要跟进每一个 issue。

    [订阅公告](https://groups.google.com/g/multimolecule){ .md-button .md-button--primary }

-   __Discourse__

    讨论某个生物问题适合哪个 pipeline、模型或数据集，也可以分享配置、请求模型和交流模型组件设计。

    [加入讨论](https://multimolecule.discourse.group){ .md-button }

</div>

</div>
</section>
