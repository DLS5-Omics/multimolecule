---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# modules

`modules` provides a collection of pre-defined modules for users to implement their own architectures.

MultiMolecule is built upon the :hugs: ecosystem, embracing a similar design philosophy: [**~~Don't~~ Repeat Yourself**](https://huggingface.co/blog/transformers-design-philosophy).
We follow the `single model file policy` where each model under the [`models`](../models) package contains one and only one `modeling.py` file that describes the network design.

The `modules` package is intended for simple, reusable modules that are consistent across multiple models. This approach minimizes code duplication and promotes clean, maintainable code.

## Key Features

- Reusability: The `modules` package includes components that are commonly used across different models, such as the [`SequencePredictionHead`][multimolecule.SequencePredictionHead]. This reduces redundancy and simplifies the development process.
- Consistency: By centralizing common modules, we ensure that updates and improvements are consistently applied across all models, enhancing reliability and performance.
- Flexibility: While modules such as transformer encoders are widely used, they often vary in implementation details (e.g., pre-norm vs. post-norm, different residual connection strategies). The module package focuses on simpler components, leaving complex, model-specific variations to be defined within each model's `modeling.py`.

## Modules

- [heads](heads): Contains various prediction heads, such as [`SequencePredictionHead`][multimolecule.SequencePredictionHead], [`TokenPredictionHead`][multimolecule.TokenPredictionHead], and [`ContactPredictionHead`][multimolecule.ContactPredictionHead].
- [embeddings](embeddings): Contains various positional embeddings, such as [`SinusoidalEmbedding`][multimolecule.SinusoidalEmbedding] and [`RotaryEmbedding`][multimolecule.RotaryEmbedding].
- [model](model): The model layer that the [`Runner`][multimolecule.runner.Runner] consumes — abstract [`ModelBase`][multimolecule.ModelBase] plus two concrete subclasses, [`MonoModel`][multimolecule.MonoModel] (single-task wrapper around a Hugging Face `AutoModelFor*`) and [`PolyModel`][multimolecule.PolyModel] (composition of backbone, optional neck, and one head per task).

## Models

`modules` exposes a small model layer used by the [`runner`](../runner) package:

- [`ModelBase`][multimolecule.ModelBase]: Abstract base. Defines the `forward` and `trainable_parameters` contract every multimolecule model implements; the runner discriminates models with `isinstance(model, ModelBase)` rather than against any concrete subclass.
- [`MonoModel`][multimolecule.MonoModel]: Single-task wrapper around a multimolecule (or HuggingFace) `AutoModelFor*` prediction model. Hides the wrapper at the `state_dict` layer, so checkpoints round-trip with the bare HF model.
- [`PolyModel`][multimolecule.PolyModel]: Composes a backbone, optional neck, and one head per task into a single trainable module. Use when the task graph involves multiple labels, extra non-sequence features, or a neck transform.

Both classes are registered with [`MODELS`][multimolecule.MODELS] under the keys `"mono"` and `"poly"`. The default `network.type: auto` dispatches between them based on the resolved network shape; users may set `network.type: mono` or `network.type: poly` explicitly to bypass the dispatcher.
