---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# module

`module` provides a collection of pre-defined modules for users to implement their own architectures.

MultiMolecule is built upon the :hugs: ecosystem, embracing a similar design philosophy: [**~~Don't~~ Repeat Yourself**](https://huggingface.co/blog/transformers-design-philosophy).
We follow the `single model file policy` where each model under the [`models`](../models) package contains one and only one `modeling.py` file that describes the network design.

The `module` package is intended for simple, reusable modules that are consistent across multiple models. This approach minimizes code duplication and promotes clean, maintainable code.

## Key Features

- Reusability: The `module` package includes components that are commonly used across different models, such as the [`SequencePredictionHead`][multimolecule.SequencePredictionHead]. This reduces redundancy and simplifies the development process.
- Consistency: By centralizing common modules, we ensure that updates and improvements are consistently applied across all models, enhancing reliability and performance.
- Flexibility: While modules such as transformer encoders are widely used, they often vary in implementation details (e.g., pre-norm vs. post-norm, different residual connection strategies). The module package focuses on simpler components, leaving complex, model-specific variations to be defined within each model's `modeling.py`.

## Modules

- [heads](heads): Contains various prediction heads, such as [`SequencePredictionHead`][multimolecule.SequencePredictionHead], [`NucleotidePredictionHead`][multimolecule.NucleotidePredictionHead], and [`ContactPredictionHead`][multimolecule.ContactPredictionHead].
- [embeddings](embeddings): Contains various positional embeddings, such as [`SinusoidalEmbedding`][multimolecule.SinusoidalEmbedding] and [`RotaryEmbedding`][multimolecule.RotaryEmbedding].
