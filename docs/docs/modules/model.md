---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# Model

The `model` sub-module of [`modules`][multimolecule.modules] defines the model layer the
[`Runner`][multimolecule.runner.Runner] consumes: an abstract [`ModelBase`][multimolecule.ModelBase] plus two
concrete subclasses ([`MonoModel`][multimolecule.MonoModel] and [`PolyModel`][multimolecule.PolyModel]) registered
with [`MODELS`][multimolecule.MODELS].

::: multimolecule.modules.ModelBase

::: multimolecule.modules.MonoModel

::: multimolecule.modules.PolyModel

::: multimolecule.modules.MODELS
