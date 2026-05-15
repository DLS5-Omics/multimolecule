---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# runner

`runner` 提供了一个用于微调、评估和推理 MultiMolecule 模型的实验运行器。
它在 DanLing 的 runner 栈之上扩展了分子序列相关的默认行为：tokenizer 加载、数据集与任务推断、
预测头的自动构建、针对预训练 backbone 的优化器参数分组、训练阶段的流式 metrics 以及评估阶段的全局 metrics。

该 runner 适用于将预训练序列模型适配到本地或 Hugging Face 数据集中一个或多个标签的有监督场景。
它会根据推断出的网络结构构建 [`MonoModel`][multimolecule.MonoModel]
（单任务、仅序列输入的 backbone）或 [`PolyModel`][multimolecule.PolyModel]
（多任务 / 多输入的组合模型）。

## 快速开始

在工作目录创建 `config.yaml`：

```yaml
pretrained: multimolecule/rnafm

data:
  root: data/my-rna-task
  train: train.parquet
  validation: valid.parquet
  test: test.parquet
  sequence_cols: [sequence]
  label_cols: [activity]
  truncation: true
  max_seq_length: 512

dataloader:
  batch_size: 16
  num_workers: 4

network:
  backbone:
    sequence: {}

optim:
  type: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-2
  pretrained_ratio: 1.0e-2

sched:
  type: cosine
  final_lr: 0.0

epochs: 10
```

通过 `mmtrain` / `mmevaluate` / `mminfer` 控制台脚本启动；
关于 CLI 覆盖、可选标志以及工作目录契约，请参见 [api](../api)。

## 程序内调用

在不依赖 CLI 编排的情况下嵌入使用：

```python
from multimolecule.runner import Config, RUNNERS

config = Config(
    pretrained="multimolecule/rnafm",
    data={"root": "data/my-rna-task", "sequence_cols": ["sequence"], "label_cols": ["activity"]},
    optim={"lr": 1.0e-4, "weight_decay": 1.0e-2, "pretrained_ratio": 1.0e-2},
    epochs=10,
)
config.interpolate()

runner = RUNNERS.build(config)
runner.train()
```

`config.interpolate()` 会执行 [`Config.post`][multimolecule.runner.Config.post]，
填入 `epochs` 默认值，将 `pretrained` 传递到 `network.backbone.sequence.name`，并在未设置时推导 `name`。

## 数据与任务

`data.root` 既可以是本地目录，也可以是 Hugging Face 数据集 ID。
本地数据集可以通过 `train`、`validation`、`test`、`infer` 等键显式声明各个 split 文件；
否则 runner 会按 Hugging Face 标准的数据文件模式自动发现。

[`Dataset`][multimolecule.data.Dataset] 会为每个 label 列推断任务：

- 任务级别：sequence、token 或 contact
- 任务类型：binary、multiclass、multilabel 或 regression
- 标签或类别数

runner 使用这些任务元数据将兼容的预测头加入 `network.heads`。
已有的 head 设置会被保留，因此你仍然可以在 config 中覆盖 head 的 dropout、hidden size 等选项。

## 模型构建

除非 backbone 已经显式指定 `name`，否则 `pretrained` 字段会被复制到 `network.backbone.sequence.name`。
设置 `use_pretrained: false` 可以保留架构但从头初始化权重。

优化器支持 `pretrained_ratio`，对 backbone 的学习率和权重衰减按该比例缩放，
而 head 与 neck 仍使用基础优化器设置。这是用全新初始化的任务头微调预训练编码器的常规做法。

`network.type` 默认为 `auto`，会按以下规则分发：

- 当不存在 neck、恰好一个 head、backbone 仅有 sequence 子树并使用预训练权重，
  且 head 类型（`sequence`、`token`、`contact`）有对应的 `AutoModelFor*` 时，分发到
  [`MonoModel`][multimolecule.MonoModel]。
- 其他情况下分发到 [`PolyModel`][multimolecule.PolyModel]
  （多任务、含 neck、多输入或 head 类型不在支持列表内）。

显式设置 `network.type: mono` 或 `network.type: poly` 可以绕过自动分发。

## Metrics

Runner 的 metrics 由 `danling.METRICS` 构建，需要 DanLing `>=0.4.0`。

- 训练阶段使用 `mode: stream`，以低开销获取按 batch 平均的 metrics
- 评估阶段使用 `mode: global`，得到 AUROC、AUPRC 等需要完整数据集才能正确计算的指标
- 多任务运行使用 [`MultiTaskMetrics`][danling.MultiTaskMetrics] 并以 macro 方式聚合

这样既保持训练阶段开销低，也避免了在需要全量评估集才能正确计算的 metric 上使用近似值。

## 检查点与 EMA

Runner 继承 DanLing 的检查点流程。恢复源按以下顺序选择，使用第一个被设置的键：

1. `resume` —— 完整状态的检查点路径。
2. `checkpoint_path` —— `resume` 的别名。
3. `auto_resume: true` —— 自动发现后端最新的检查点。
4. `model_pretrained` / `load_pretrained` —— 仅初始化模型权重（不恢复 optimizer / scheduler / RNG）。

`checkpoint` 本身保留给 DanLing 的检查点*策略*（`backend`、`interval`、`keep_latest_k` 等）；
完整字段请参见 [DanLing runner 文档](https://danling.org/runners/)。

可选的指数滑动平均（EMA）通过下列配置启用：

```yaml
ema:
  enabled: true
  beta: 0.9999
  update_every: 8
```

启用 EMA 时，评估和推理使用 EMA 模型。
