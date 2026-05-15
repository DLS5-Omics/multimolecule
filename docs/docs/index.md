---
template: home.html
title: MultiMolecule
description: Modular ecosystem for end-to-end biomolecular machine learning.
summary: 不作意 任自然 即此语 是灵丹
authors:
  - Zhiyuan Chen
date: 2022-05-04
hide:
  - toc
hero:
  eyebrow: Accelerate Molecular Biology Research with Machine Learning
  title: MultiMolecule
  description: "MultiMolecule provides ready-to-use pipelines, pretrained model checkpoints, curated datasets, and training tools for RNA, DNA, and protein sequence research."
---

<section class="mm-section mm-section--paths" id="paths" markdown>
<div class="mm-section__inner" markdown>

## What are you trying to do?

Start from the task you need: predict from a sequence, fine-tune on your data, load a pretrained model, or use a curated dataset.

<div class="mm-workflow-screens" aria-label="MultiMolecule entry points" markdown>

<section class="mm-workflow-panel" id="tasks" markdown>
<div class="mm-workflow-copy" markdown>
<span>Prediction</span>

### Predict from a sequence

Registered pipelines turn biological task names and input sequences into structured predictions without manual model assembly.
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

### Fine-tune on your data

The runner connects pretrained checkpoints with Hugging Face datasets or labelled local tables, using sequence and label columns to start supervised training.
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

### Load a pretrained model

Model cards give checkpoint IDs, expected inputs, citations, and licenses, while Python APIs support direct model control beyond task pipelines.
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

### Use a curated dataset

Curated biological datasets include sequence and label fields, task metadata, source information, citations, and licenses for benchmarks, examples, and fine-tuning.
</div>

```python
from datasets import load_dataset

dataset = load_dataset("multimolecule/chanrg", split="train")
```
</section>

</div>

<nav class="mm-workflow-dots" aria-label="Workflow panels">
<a class="mm-workflow-dot is-active" href="#tasks" aria-current="true" aria-label="Show prediction workflow"></a>
<a class="mm-workflow-dot" href="#training" aria-label="Show training workflow"></a>
<a class="mm-workflow-dot" href="#models" aria-label="Show model workflow"></a>
<a class="mm-workflow-dot" href="#datasets" aria-label="Show dataset workflow"></a>
</nav>

</div>
</section>

<section class="mm-section mm-section--stack" markdown>
<div class="mm-section__inner" markdown>

## One stack underneath

When you need more control, the same ecosystem exposes documented resources, biological input handling, reusable model components, and execution tools for prediction, training, evaluation, and scripted use.

<div class="mm-stack-tower" markdown>

<div class="mm-stack-row mm-stack-row--top" markdown>
<section class="mm-stack-step mm-stack-step--execution" markdown>
<span>Execution</span>

__Pipelines, runner, and API__

Pipelines provide ready task predictions, the runner manages supervised training and evaluation, and API entry points support scripts and applications.

<p class="mm-stack-links" markdown>[`runner`](runner/index.md) · [`pipelines`](pipelines/index.md) · [`api`](api/index.md)</p>
</section>
</div>

<div class="mm-stack-row mm-stack-row--middle" markdown>
<section class="mm-stack-step mm-stack-step--resources" markdown>
<span>Resources</span>

__Models and datasets with provenance__

Dataset cards and model cards collect supported inputs, task names, checkpoint IDs, citations, licenses, and training metadata.

<p class="mm-stack-links" markdown>[`datasets`](datasets/index.md) · [`models`](models/index.md)</p>
</section>
</div>

<div class="mm-stack-row mm-stack-row--base" markdown>
<section class="mm-stack-step mm-stack-step--data" markdown>
<span>Data layer</span>

__Biological data to model-ready inputs__

IO, tokenisers, and data utilities turn biological sequences, structures, and annotations into consistent inputs for pipelines, training, and evaluation.

<p class="mm-stack-links" markdown>[`io`](io/index.md) · [`tokenisers`](tokenisers/index.md) · [`data`](data/index.md)</p>
</section>

<section class="mm-stack-step mm-stack-step--model" markdown>
<span>Model layer</span>

__Reusable model building blocks__

Models provide pretrained configs, AutoModel classes, checkpoints, and output contracts; modules provide backbones, heads, losses, and embeddings for custom architectures.

<p class="mm-stack-links" markdown>[`models`](models/index.md) · [`modules`](modules/index.md)</p>
</section>
</div>
</div>

</div>
</section>

<section class="mm-section mm-section--tail" markdown>
<div class="mm-section__inner" markdown>

## Community

<div class="grid cards mm-trust-grid" markdown>

-   __Google Group__

    Receive release announcements, migration notes, and design RFCs without following every issue.

    [Subscribe to announcements](https://groups.google.com/g/multimolecule){ .md-button .md-button--primary }

-   __Discourse__

    Ask which pipeline, model, or dataset fits a biological problem; share configs, request models, and discuss model components.

    [Join discussion](https://multimolecule.discourse.group){ .md-button }

</div>

</div>
</section>
