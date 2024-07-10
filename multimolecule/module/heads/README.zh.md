---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# heads

`heads` 提供了一系列的模型预测头，用于处理不同的任务。

`heads` 接受 [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output)、[`dict`][] 或 [`tuple`][] 作为输入。
它会自动查找预测所需的模型输出并相应地处理。

一些预测头可能需要额外的信息，例如 `attention_mask` 或 `input_ids`，例如 [`ContactPredictionHead`][multimolecule.ContactPredictionHead]。
这些额外的参数可以作为参数/关键字参数传入。

请注意，`heads` 使用与 :hugs: Transformers 相同的 [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output) 约定。
如果模型输出是一个 [`tuple`][]，我们将第一个元素视为 `pooler_output`，第二个元素视为 `last_hidden_state`，最后一个元素视为 `attention_map`。
用户有责任确保模型输出格式正确。

如果模型输出是一个 [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output) 或一个 [`dict`][]，`heads` 将从模型输出中查找 [`HeadConfig.output_name`][multimolecule.module.HeadConfig]。
你可以在 [`HeadConfig`][multimolecule.module.HeadConfig] 中指定 `output_name`，以确保 `heads` 可以正确定位所需的张量。
