---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# heads

`heads` provide a collection of pre-defined prediction heads.

`heads` take in either a [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output), a [`dict`][], or a [`tuple`][] as input.
It automatically looks for the model output required for prediction and processes it accordingly.

Some prediction heads may require additional information, such as the `attention_mask` or the `input_ids`, like [`ContactPredictionHead`][multimolecule.ContactPredictionHead].
These additional arguments can be passed in as arguments/keyword arguments.

Note that `heads` use the same [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output) conventions as the :hugs: Transformers.
If the model output is a [`tuple`][], we consider the first element as the `pooler_output`, the second element as the `last_hidden_state`, and the last element as the `attention_map`.
It is the user's responsibility to ensure that the model output is correctly formatted.

If the model output is a [`ModelOutupt`](https://huggingface.co/docs/transformers/en/main_classes/output) or a [`dict`][], the `heads` will look for the [`HeadConfig.output_name`][multimolecule.module.HeadConfig] from the model output.
You can specify the `output_name` in the [`HeadConfig`][multimolecule.module.HeadConfig] to ensure that the `heads` can correctly locate the required tensor.
