# Model Implementation Guide

This guide defines the conventions for adding or refactoring models in MultiMolecule.
The goal is to expose a consistent MultiMolecule API across models.
Most models are converted from external repositories, but upstream implementation details should not determine the downstream API.

## Core Principles

- Ship model implementations and downstream-facing docs under MultiMolecule's AGPL-3.0-or-later license.
- Preserve the original checkpoint behavior, but expose a clean MultiMolecule API.
- Rewrite model code in MultiMolecule style instead of copying upstream source structure.
- Use a different implementation path only when it is semantically equivalent and improves correctness, performance, maintainability, or compatibility.
- Keep upstream checkpoint quirks in conversion code when possible; do not leak them into downstream APIs.
- Treat ensemble members, checkpoint shards, and original training script details as implementation details unless users must configure them.
- Do not require upstream repository files, scripts, or binaries at runtime.
- Prefer small, explicit PyTorch modules over opaque wrappers around upstream concepts.

## Provenance And Licensing

- Record the upstream paper, repository, checkpoint source, and any non-code asset provenance needed for conversion or reproducibility.
- Fully rewritten implementations should expose the MultiMolecule AGPL-3.0-or-later license; do not mirror upstream code-license text in runtime APIs or downstream-facing README license sections.
- Do not copy upstream source code into the model implementation.
- Keep source-specific provenance in the converter or README, not in runtime model APIs.
- If the upstream paper, code, and checkpoint disagree, implement the checkpoint behavior and document the discrepancy.

## File Layout

Each model should use the standard package layout:

- `configuration_<model>.py`: configuration classes only.
- `modeling_<model>.py`: model, modules, output dataclasses, and model-local helpers.
- `convert_checkpoint.py`: checkpoint conversion, tokenizer configuration, and state-dict key/value conversion when the model is converted from an external checkpoint.
- `README.md`: model card content following existing MultiMolecule model READMEs.
- `__init__.py`: public exports and auto registration.

New Python files should keep the standard MultiMolecule license header and `from __future__ import annotations`.

## Configuration

- Define a `<ModelName>Config` that inherits from `PreTrainedConfig` and sets `model_type`.
- Keep configuration fields user-facing when possible; checkpoint-only details should stay private unless they are required to reconstruct the model.
- Use `HeadConfig`, `MaskedLMHeadConfig`, or `BaseHeadConfig` for standard prediction heads.
- Use small `FlatDict` nested configs only when they make repeated architecture pieces clearer, such as stages or internal modules.
- Validate incompatible configuration values in `__init__` instead of silently accepting invalid architectures.

## Naming

- Use `<ModelName>PreTrainedModel` for the abstract base class.
- Use `<ModelName>Model` for the public backbone model registered with `AutoModel`.
- Use `<ModelName>For<Task>` for task-specific public heads, such as `ForMaskedLM`, `ForPreTraining`, `ForSequencePrediction`, `ForTokenPrediction`, `ForContactPrediction`, and `ForSecondaryStructurePrediction`.
- Use `<ModelName>Module` for a full repeated internal checkpoint member; avoid upstream names such as `Network`.
- Use `<ModelName>Embeddings`, `<ModelName>Encoder`, and `<ModelName>Layer` for embedding layers, top-level stacks, and repeated blocks.
- Use established component names such as `Attention`, `SelfAttention`, `Intermediate`, `Output`, `Pooler`, `PredictionHead`, `conv1`, `batch_norm1`, and `prediction`.
- Avoid upstream abbreviations and implementation names in the public module tree unless they are established model terminology.

State-dict names should be stable, readable, and consistent across MultiMolecule models.
Checkpoint key translation is the default convention, even when upstream names are acceptable, because it keeps module names aligned across models.
Implement key translation in `convert_checkpoint.py` rather than preserving upstream naming.

## PreTrainedModel Base

- Set `config_class`, `base_model_prefix = "model"`, gradient-checkpointing support, attention backend support, and `_no_split_modules` on `<ModelName>PreTrainedModel` when applicable.
- Implement `_init_weights` on the base class when the architecture needs non-default initialization.
- Call `self.post_init()` after all child modules are defined in public model classes.
- Implement `get_input_embeddings` and `set_input_embeddings` for learned-embedding backbones.
- Implement `get_output_embeddings` and `set_output_embeddings` for language-model heads with output embeddings.

## Gradient Checkpointing

- Only set `supports_gradient_checkpointing = True` when the model has a real checkpointed execution path.
- Do not explicitly set `supports_gradient_checkpointing = False` without a concrete reason explaining why checkpointing is unsafe or not implementable for that architecture.
- Add `self.gradient_checkpointing = False` to the module that owns the checkpointable stack, and use `self._gradient_checkpointing_func(module.__call__, ...)` in the training path.
- Checkpoint the major repeated blocks that dominate activation memory when they are semantically safe to recompute.
- Include recurrent blocks such as `nn.LSTM` or BLSTM when forward/backward equivalence has been verified; do not exclude them only because they are less common than transformer or convolution blocks.
- Do not treat removing `supports_gradient_checkpointing = True` as a fix for a model that should support gradient checkpointing. Implement support, or explicitly document the concrete blocker that makes checkpointing unsafe or not implementable.
- Verify that `gradient_checkpointing_enable()` succeeds, training forward/backward works, and checkpointed outputs match the non-checkpointed path on a representative deterministic configuration.
- Keep checkpointing inactive in evaluation mode so inference behavior remains unchanged.

## Code Organization

For newly implemented or heavily rewritten models, order classes by execution:

1. `PreTrainedModel` base class.
2. Public backbone and task-specific model classes.
3. Top-level internal module.
4. Embeddings or input projection.
5. Encoder.
6. Layer.
7. Lower-level attention, convolution, MLP, normalization, and helper modules.
8. Output heads.
9. Local helper functions.
10. Output dataclasses.

Within each class, follow the same rule: define modules in the order they are used in `forward`.
For model families that intentionally follow an existing shared layout, keep the family order consistent across siblings.

## Modules

- Prefer explicit `nn.Module` classes with named child modules and a readable `forward`.
- Do not wrap a single anonymous `nn.Sequential` inside a module just to rename it.
- If a block is truly sequential, either use direct named modules with an explicit `forward`, or make the block itself a clear sequential abstraction with named children.
- Do not copy upstream class names when they do not meet the MultiMolecule naming bar.
- Isolate unavoidable optional runtime dependencies behind clear imports and errors.
- Preserve dtype and device behavior; create new tensors on the relevant input, parameter, or buffer device.
- Raise explicit errors for unsupported argument combinations instead of silently ignoring them.
- Keep comments rare and factual; explain non-obvious compatibility choices, not obvious tensor assignments.

## Public API And Registration

- Export the config, `<ModelName>PreTrainedModel`, `<ModelName>Model`, task heads, and output classes through `__all__`.
- Register `AutoConfig`, `AutoModel`, and task-specific AutoModel classes in `__init__.py`.
- Register `AutoBackbone` for backbone models that support the backbone API.
- Prefer MultiMolecule in-house tokenizers and register `AutoTokenizer` when the config maps directly to one.
- Use or convert an upstream tokenizer only when tokenization is part of the model semantics and cannot be represented by a MultiMolecule tokenizer.
- Ensure the converter saves a complete tokenizer config with the checkpoint.
- Keep Hugging Face aliases aligned with MultiMolecule task names, such as mapping sequence prediction to sequence classification and token prediction to token classification when applicable.

## Outputs And Heads

- Return `ModelOutput` dataclasses or standard Transformers output dataclasses.
- Preserve Transformers-compatible tuple behavior with `@can_return_tuple` and config-default merging with `@merge_with_config_defaults` where established by the model family.
- Use `@capture_outputs` for models that support output recording.
- Reuse common heads such as `SequencePredictionHead`, `TokenPredictionHead`, `ContactPredictionHead`, `MaskedLMHead`, and `BasePredictionHead` before adding model-local heads.
- Use shared loss/output helpers such as `Criterion` and `HeadOutput` when implementing supervised heads.
- Add `postprocess` only when the downstream pipeline needs a model-specific final representation, such as a contact map.

## Inputs And Vocabulary

Models should consume the vocabulary order defined by the MultiMolecule tokenizer and configuration.

- Do not reinterpret token ids in `forward` to match an upstream repository's vocabulary order.
- For learned-embedding models, convert `word_embeddings` and preserve tokenizer/config semantics for special tokens, `padding_idx`, and `attention_mask`.
- For one-hot or feature-channel models, derive features from MultiMolecule token order and preserve the architecture's masking behavior.
- Convert word embeddings, first-layer input channels, pairwise channels, classifier channels, or equivalent feature projections in `convert_checkpoint.py`.
- Fixed vocabulary-dependent constants should use MultiMolecule order.
- Preserve the model family's `input_ids` and `inputs_embeds` interface; if both are supported, validate that callers provide exactly one.
- Preserve `danling.NestedTensor` compatibility where the model family supports variable-length inputs.
- Follow the original model semantics; keep `NestedTensor` inputs native when the implementation path supports them.

For models without learned word embeddings, apply the same principle to the first learned layer that sees vocabulary-dependent channels.
Use tokenizer helper functions such as `get_alphabet`, `get_tokenizer_config`, and `convert_word_embeddings` instead of hand-writing vocabulary conversion logic.

## Buffers And Checkpoint State

- Learned parameters and original checkpoint tensors should be persistent.
- Deterministic constants should usually be non-persistent buffers.
- Examples of non-persistent constants include Gaussian windows, fixed pairing-score matrices, index maps, and masks that can be rebuilt from config.
- Do not add deterministic constants to converted state dicts just to satisfy loading. Register them on the model instead.

## Conversion

Conversion scripts should own checkpoint compatibility work:

- Map upstream key names to MultiMolecule names.
- Convert tensor layouts and vocabulary/channel order, including transposes, QKV splits or fusions, convolution kernel layouts, tied embeddings, decoder weights, and decoder biases.
- Add only checkpoint-required state such as criterion buffers.
- Use `load_checkpoint` and `save_checkpoint` from `multimolecule.models.conversion_utils`.
- Use MultiMolecule tokenizer utilities such as `get_alphabet`, `get_tokenizer_config`, and `convert_word_embeddings`, or convert the original tokenizer when the upstream model requires it.
- Save tokenizer configuration with the converted checkpoint.
- Keep conversion helpers small and testable, such as `_convert_checkpoint`, `_convert_name`, `convert_original_state_dict_key`, and `convert_original_state_dict_value`.

The converted model should not need to know the upstream vocabulary order or upstream state-dict names.

## README And Docs

READMEs should follow existing MultiMolecule conventions:

- Include a `Model Specification` table using `count_parameters`, `calculate_flops`, and `calculate_macs` from `multimolecule.utils`.
- Include architecture columns such as `Num Layers`, `Hidden Size`, `Num Heads`, `Intermediate Size`, and `Max Num Tokens` when they are meaningful; otherwise use only parameters, FLOPs, and MACs.
- Use bullet lists as `key: value`, not prose paragraphs disguised as lists.
- Include training data, training details, citation, license, and known limitations when the source provides them.
- The license section should state the MultiMolecule AGPL-3.0-or-later license for fully rewritten model implementations.
- Describe training hardware and training details in the same style as existing READMEs.
- Keep internal ensemble/checkpoint-member details out of downstream-facing documentation unless they affect user behavior.
- Add a model page under `docs/docs/models/` and register it in `docs/mkdocs.yml` when the model is part of the public docs.

## Review Checklist

- The config is public-facing, validated, and contains only necessary architecture fields.
- The model follows class naming and execution-order organization.
- The downstream API is a single clean model unless there is a real user-facing reason for variants.
- Public exports and Auto\* registrations match the supported tasks.
- Outputs use standard dataclasses and preserve expected `return_dict` behavior.
- If `supports_gradient_checkpointing = True` is set, `gradient_checkpointing_enable()` and training backward pass are verified.
- Vocabulary/order differences are handled in conversion, not runtime remapping.
- `NestedTensor` compatibility follows the original model semantics and stays native where the implementation path supports it.
- Deterministic constants are non-persistent buffers.
- New tensors created in `forward` follow the active dtype and device.
- Conversion loads the original checkpoint without missing or unexpected keys.
- `py_compile` and `ruff check` pass for the touched model, converter, README-adjacent code, and related docs.
