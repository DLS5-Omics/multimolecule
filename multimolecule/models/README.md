---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# models

`models` provide a collection of pre-trained models.

## Usage

### Direct Access

All models can be directly loaded with the `from_pretrained` method.

```python
--8<-- "demo/direct-access.py:17:"
```

### Build with [`transformers.AutoModel`][]s

All models have been registered with [`transformers.AutoModel`][]s, and can be directly loaded using corresponding [`transformers.AutoModel`][]s classes.

```python
--8<-- "demo/transformers-automodel.py:17:"
```

!!! danger "`import multimolecule` before use"

    Note that you must `import multimolecule` before building the model using `transformers.AutoModel`.
    The registration of models is done in the `multimolecule` package, and the models are not available in the `transformers` package.

    The following error will be raised if you do not `import multimolecule` before using `transformers.AutoModel`:

    ```python
    ValueError: The checkpoint you are trying to load has model type `rnafm` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
    ```

### Build with `multimolecule.AutoModel`s

Similar to [`transformers.AutoModel`][]s, MultiMolecule provides a set of `multimolecule.AutoModel`s for direct access to models.

These include:

### `multimolecule.AutoModelForNucleotideClassification`

Similar to [`transformers.AutoModelForTokenClassification`][], but removes the `<bos>` token and the `<eos>` token if they are defined in the model config.

!!! note "`<bos>` and `<eos>` tokens"

    In tokenizers provided by MultiMolecule, `<bos>` token is pointed to `<cls>` token, and `<sep>` token is pointed to `<eos>` token.

```python
--8<-- "demo/multimolecule-automodel.py:17:"
```

### Initialize a vanilla model

You can also initialize a vanilla model using the model class.

```python
--8<-- "demo/vanilla.py:17:"
```

## Available Models

### DeoxyriboNucleic Acid (DNA)

- [CaLM](models/calm.md)

### RiboNucleic acid (RNA)

- [ERNIE-RNA](models/ernierna.md)
- [RiNALMo](models/rinalmo.md)
- [RNABERT](models/rnabert.md)
- [RNA-FM](models/rnafm.md)
- [RNA-MSM](models/rnamsm.md)
- [SpliceBERT](models/splicebert.md)
- [3UTRBERT](models/utrbert.md)
- [UTR-LM](models/utrlm.md)
