---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# runner

`runner` provides an experiment runner for fine-tuning, evaluating, and running inference with MultiMolecule models.
It extends DanLing's runner stack with molecular-sequence defaults: tokenizer loading, dataset/task inference,
automatic prediction-head construction, optimizer parameter grouping for pre-trained backbones, stream metrics for
training, and global metrics for evaluation.

The runner is intended for supervised workflows where a pre-trained sequence model is adapted to one or more labels
in a local or Hugging Face dataset. It builds a [`MonoModel`][multimolecule.MonoModel] (single task, sequence-only
backbone) or [`PolyModel`][multimolecule.PolyModel] (multi-task / multi-input composition) depending on the inferred
network shape.

## Quick Start

Create a `config.yaml` in the working directory:

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

Launch via the `mmtrain` / `mmevaluate` / `mminfer` console scripts; see [api](../api) for CLI overrides, optional
flags, and the working-directory contract.

## Programmatic Usage

For embedded use without the CLI orchestration:

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

`config.interpolate()` runs [`Config.post`][multimolecule.runner.Config.post], which fills in `epochs` defaults,
propagates `pretrained` into `network.backbone.sequence.name`, and derives `name` when not set.

## Data And Tasks

`data.root` can be a local directory or a Hugging Face dataset ID. Local datasets may define split files explicitly
with keys such as `train`, `validation`, `test`, and `infer`; otherwise the runner discovers dataset files using the
standard Hugging Face data-file patterns.

[`Dataset`][multimolecule.data.Dataset] infers a task for each label column:

- task level: sequence, token, or contact
- task type: binary, multiclass, multilabel, or regression
- number of labels or classes

The runner uses that task metadata to add compatible heads to `network.heads`. Existing head settings are preserved,
so you can still override head dropout, hidden size, or other head-specific options in the config.

## Model Construction

The `pretrained` field is copied into `network.backbone.sequence.name` unless the backbone already specifies a name.
Set `use_pretrained: false` to keep the architecture but initialize weights from scratch.

The optimizer supports `pretrained_ratio`, which scales the backbone learning rate and weight decay while leaving
heads and necks at the base optimizer settings. This is the usual setup for fine-tuning a pre-trained encoder with
newly initialized task heads.

`network.type` defaults to `auto`, which dispatches to:

- [`MonoModel`][multimolecule.MonoModel] when there is no neck, exactly one head, a sequence-only backbone with
  pretrained weights, and a head type (`sequence`, `token`, `contact`) that has an `AutoModelFor*` counterpart.
- [`PolyModel`][multimolecule.PolyModel] otherwise (multi-task, neck, multi-input, or unsupported head type).

Set `network.type: mono` or `network.type: poly` explicitly to bypass auto dispatch.

## Metrics

Runner metrics are built from `danling.METRICS`, which requires DanLing `>=0.4.0`.

- training uses `mode: stream` for low-overhead batch-averaged metrics
- evaluation uses `mode: global` for dataset-level metrics such as AUROC and AUPRC
- multi-task runs use [`MultiTaskMetrics`][danling.MultiTaskMetrics] with macro aggregation

This keeps training cheap while avoiding approximate validation metrics for scores that need the full evaluation set.

## Checkpoints And EMA

The runner inherits DanLing's checkpoint flow. The resume source is selected by the first key that is set, in order:

1. `resume` — full-state checkpoint path.
2. `checkpoint_path` — alias for `resume`.
3. `auto_resume: true` — discover the latest backend checkpoint automatically.
4. `model_pretrained` / `load_pretrained` — pretrained-only initialization (no optimizer / scheduler / RNG restore).

`checkpoint` itself is reserved for DanLing's checkpoint _policy_ (`backend`, `interval`, `keep_latest_k`, ...); see
the [DanLing runner documentation](https://danling.org/runners/) for the full surface.

Optional exponential moving average can be enabled with:

```yaml
ema:
  enabled: true
  beta: 0.9999
  update_every: 8
```

When EMA is enabled, evaluation and inference use the EMA model.
