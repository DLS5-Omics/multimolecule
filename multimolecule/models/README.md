---
authors:
  - Zhiyuan Chen
date: 2024-05-04
---

# Models

`models` provide a collection of pre-trained models.

## Model Class

In the [`transformers`](https://huggingface.co/docs/transformers/en/tasks/sequence_classification) library, the names of model classes can sometimes be misleading.
While these classes support both regression and classification tasks, their names often include `xxxForSequenceClassification`, which may imply they are only for classification.

To avoid this ambiguity, MultiMolecule provides a set of model classes with clear, intuitive names that reflect their intended use:

- `multimolecule.AutoModelForContactPrediction`: Contact Prediction
- `multimolecule.AutoModelForNucleotidePrediction`: Nucleotide Prediction
- `multimolecule.AutoModelForSequencePrediction`: Sequence Prediction
- `multimolecule.AutoModelForTokenPrediction`: Token Prediction

Each of these models supports both regression and classification tasks, offering flexibility and precision for a wide range of applications.

### Contact Prediction

Contact prediction assign a label to each pair of token in a sentence.
One of the most common contact prediction tasks is protein distance map prediction.
Protein distance map prediction attempts to find the distance between all possible amino acid residue pairs of a three-dimensional protein structure

### Nucleotide Prediction

Similar to [Token Classification](https://huggingface.co/docs/transformers/en/tasks/token_classification), but removes the `<bos>` token and the `<eos>` token if they are defined in the model config.

!!! note "`<bos>` and `<eos>` tokens"

    In tokenizers provided by MultiMolecule, `<bos>` token is pointed to `<cls>` token, and `<sep>` token is pointed to `<eos>` token.

## Usage

### Build with `multimolecule.AutoModel`s

```python
--8<-- "demo/multimolecule-automodel.py:17:"
```

### Direct Access

All models can be directly loaded with the `from_pretrained` method.

```python
--8<-- "demo/direct-access.py:17:"
```

### Build with [`transformers.AutoModel`][]s

While we use a different naming convention for model classes, the models are still registered to corresponding [`transformers.AutoModel`][]s.

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
