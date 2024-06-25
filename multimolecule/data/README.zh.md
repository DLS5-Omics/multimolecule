---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# data

`data` 提供了一系列用于处理数据的实用工具。

尽管 :hugs: [`datasets`](https://huggingface.co/docs/datasets) 是一个强大的管理数据集的库，但它是一个通用工具，可能无法涵盖科学应用程序的所有特定功能。

`data` 包旨在通过提供在科学任务中常用的数据处理实用程序来补充 [`datasets`](https://huggingface.co/docs/datasets)。

## 使用

### 从本地数据文件加载

```python
--8<-- "demo/data/local-file.py:17:"
```

### 从:hugs: [`datasets`](https://huggingface.co/docs/datasets)加载

```python
--8<-- "demo/data/huggingface-datasets.py:17:"
```
