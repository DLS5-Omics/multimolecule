---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Models

`models` 提供了一系列预训练模型。

## 使用

### 直接访问

所有模型可以通过 from_pretrained 方法直接加载。

```python
--8<-- "demo/direct-access.py:17:"
```

### 使用 [`transformers.AutoModel`][] 构建

所有模型都已注册到 [transformers.AutoModel][], 并可以使用相应的 [transformers.AutoModel][] 类直接加载。

```python
--8<-- "demo/transformers-automodel.py:17:"
```

!!! danger "使用前先 `import multimolecule`"

    请注意，在使用 `transformers.AutoModel` 构建模型之前，必须先 `import multimolecule`。
    模型的注册在 `multimolecule` 包中完成，模型在 `transformers` 包中不可用。

    如果在使用 `transformers.AutoModel` 之前未 `import multimolecule`，将会引发以下错误：

    ```python
    ValueError: The checkpoint you are trying to load has model type `rnafm` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
    ```

### 使用 multimolecule.AutoModel 构建

类似于 [transformers.AutoModel][], MultiMolecule 提供了一系列 multimolecule.AutoModel 以直接访问模型。

这包括：

### `multimolecule.AutoModelForNucleotideClassification`

类似于 [transformers.AutoModelForTokenClassification][], 但如果模型配置中定义了 `<bos>` 或者 `<eos>` 令牌，将其移除。

!!! note "`<bos>` 和 `<eos>` 令牌"

    在 MultiMolecule 提供的分词器中，`<bos>` 令牌指向 `<cls>` 令牌，`<sep>` 令牌指向 `<eos>` 令牌。

```python
--8<-- "demo/multimolecule-automodel.py:17:"
```

### 初始化一个香草模型

你也可以使用模型类初始化一个基础模型。

```python
--8<-- "demo/vanilla.py:17:"
```

## 可用模型

### 脱氧核糖核酸（DNA）

- [CaLM](models/calm.md)

### 核糖核酸（RNA）

- [ERNIE-RNA](models/ernierna.md)
- [RiNALMo](models/rinalmo.md)
- [RNABERT](models/rnabert.md)
- [RNA-FM](models/rnafm.md)
- [RNA-MSM](models/rnamsm.md)
- [SpliceBERT](models/splicebert.md)
- [3UTRBERT](models/utrbert.md)
- [UTR-LM](models/utrlm.md)
