---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# data

`data` provides a collection of data processing utilities for handling data.

While :hugs: [`datasets`](https://huggingface.co/docs/datasets) is a powerful library for managing datasets, it is a general-purpose tool that may not cover all the specific functionalities of scientific applications.

The `data` package is designed to complement [`datasets`](https://huggingface.co/docs/datasets) by offering additional data processing utilities that are commonly used in scientific tasks.

## Usage

### Load from local data file

```python
--8<-- "demo/data/local-file.py:17:"
```

### Load from :hugs: [`datasets`](https://huggingface.co/docs/datasets)

```python
--8<-- "demo/data/huggingface-datasets.py:17:"
```
