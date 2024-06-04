---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# 模型

`models` 提供了一系列预训练模型。

## 模型类

在 [`transformers`](https://huggingface.co/docs/transformers/en/tasks/sequence_classification) 库当中，模型类的名字有时可以引起误解。
尽管这些类支持回归和分类任务，但它们的名字通常包含 `xxxForSequenceClassification`，这可能暗示它们只能用于分类。

为了避免这种歧义，MultiMolecule 提供了一系列模型类，这些类的名称清晰、直观，反映了它们的预期用途：

- `multimolecule.AutoModelForContactPrediction`: 接触预测
- `multimolecule.AutoModelForNucleotidePrediction`: 核苷酸预测
- `multimolecule.AutoModelForSequencePrediction`: 序列预测
- `multimolecule.AutoModelForTokenPrediction`: 令牌预测

每个模型都支持回归和分类任务，为广泛的应用提供了灵活性和精度。

### 接触预测

接触预测为序列中的每一对令牌分配一个标签。
最常见的接触预测任务之一是蛋白质距离图预测。
蛋白质距离图预测试图找到三维蛋白质结构中所有可能的氨基酸残基对之间的距离

### 核苷酸预测

与 [Token Classification](https://huggingface.co/docs/transformers/en/tasks/token_classification) 类似，但如果模型配置中定义了 `<bos>` 或 `<eos>` 令牌，则将其移除。

!!! note "`<bos>` 和 `<eos>` 令牌"

    在 MultiMolecule 提供的分词器中，`<bos>` 令牌指向 `<cls>` 令牌，`<sep>` 令牌指向 `<eos>` 令牌。

## 使用

### 使用 `multimolecule.AutoModel` 构建

```python
--8<-- "demo/multimolecule-automodel.py:17:"
```

### 直接访问

所有模型可以通过 `from_pretrained` 方法直接加载。

```python
--8<-- "demo/direct-access.py:17:"
```

### 使用 [`transformers.AutoModel`][] 构建

虽然我们为模型类使用了不同的命名约定，但模型仍然注册到相应的 [`transformers.AutoModel`][] 中。

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
