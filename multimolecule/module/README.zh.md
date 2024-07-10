---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# module

`module` 提供了一系列预定义模块，供用户实现自己的架构。

MultiMolecule 建立在 :hugs: 生态系统之上，拥抱类似的设计理念：[**~~不要~~ 重复自己**](https://huggingface.co/blog/transformers-design-philosophy)。
我们遵循 `单一模型文件策略`，其中 [`models`](../models) 包中的每个模型都包含一个且仅有一个描述网络设计的 `modeling.py` 文件。

`module` 包旨在提供简单、可重用的模块，这些模块在多个模型中保持一致。这种方法最大程度地减少了代码重复，并促进了干净、易于维护的代码。

## 核心特性

- 可重用性：`module` 包括一些在不同模型中常用的组件，例如 [`SequencePredictionHead`][multimolecule.SequencePredictionHead]。这减少了冗余，并简化了开发过程。
- 一致性：通过集中常见模块，我们确保更新和改进在所有模型中一致应用，提高了可靠性和性能。
- 灵活性：虽然变换网络编码器等模块被广泛使用，但它们在实现细节上经常有所不同（例如，前-归一化 vs. 后-归一化，不同的残差连接策略）。`module` 包专注于更简单的组件，将复杂的、特定于模型的变化留给每个模型的 `modeling.py` 中定义。

## Modules

- [heads](heads): 包括多种预测头，比如[`SequencePredictionHead`][multimolecule.SequencePredictionHead]、[`NucleotidePredictionHead`][multimolecule.NucleotidePredictionHead]和[`ContactPredictionHead`][multimolecule.ContactPredictionHead]。
- [embeddings](embeddings)：包括多种位置编码，比如[`SinusoidalEmbedding`][multimolecule.SinusoidalEmbedding]和 [`RotaryEmbedding`][multimolecule.RotaryEmbedding]。
