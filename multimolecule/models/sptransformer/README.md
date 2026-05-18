---
language: dna
tags:
  - Biology
  - DNA
  - Splicing
license: agpl-3.0
datasets:
  - multimolecule/gencode
library_name: multimolecule
---

# SpTransformer

Transformer network for predicting tissue-specific splicing from pre-mRNA sequences.

## Disclaimer

This is an UNOFFICIAL implementation of [SpliceTransformer predicts tissue-specific splicing linked to human diseases](https://doi.org/10.1038/s41467-024-53088-6) by Ningyuan You et al.

The OFFICIAL repository of SpliceTransformer (SpTransformer) is at [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing SpTransformer did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

SpTransformer (SpliceTransformer) is a deep neural network that predicts tissue-specific splicing from primary pre-mRNA sequence.
It combines two pretrained SpliceAI-style dilated-residual convolutional feature extractors with a trainable input-projection path; the concatenated features are processed by a Sinkhorn transformer attention block with axial positional embeddings.
For each position the network predicts a 3-channel splice-site score (no-splice / acceptor / donor) and a per-position splice-site usage score across 15 human tissues.
The model uses a fixed flanking context of 4,000 nucleotides on each side of every predicted position.
SpTransformer is typically used to estimate the effect of genetic variants on tissue-specific splicing by scoring reference and alternate sequences and taking the difference.
Please refer to the [Training Details](#training-details) section for more information on the training process.

### Model Specification

| Num Layers | Hidden Size | Num Heads | Intermediate Size | Max Seq Len | Num Parameters (M) | FLOPs (G) | MACs (G) | Context |
| ---------- | ----------- | --------- | ----------------- | ----------- | ------------------ | --------- | -------- | ------- |
| 8          | 256         | 8         | 1024              | 8192        | 17.07              | 290.72    | 144.65   | 4000    |

### Links

- **Code**: [multimolecule.sptransformer](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/sptransformer)
- **Weights**: [multimolecule/sptransformer](https://huggingface.co/multimolecule/sptransformer)
- **Data**: GTEx human RNA-seq across 15 tissues with gene annotations from GENCODE and multi-species sequence data
- **Paper**: [SpliceTransformer predicts tissue-specific splicing linked to human diseases](https://doi.org/10.1038/s41467-024-53088-6)
- **Developed by**: Ningyuan You, Chang Liu, Yuxin Gu, Rong Wang, Hanying Jia, Tianyun Zhang, Song Jiang, Jinsong Shi, Ming Chen, Min-Xin Guan, Siqi Sun, Shanshan Pei, Zhihong Liu, Ning Shen
- **Model type**: Transformer encoder with windowed-local and Sinkhorn sorted-bucket attention for tissue-specific splicing prediction
- **Original Repository**: [ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### RNA Splicing Site Prediction

You can use this model directly to predict per-nucleotide tissue-specific splicing of a pre-mRNA sequence:

```python
>>> from multimolecule import DnaTokenizer, SpTransformerModel

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/sptransformer")
>>> model = SpTransformerModel.from_pretrained("multimolecule/sptransformer")
>>> output = model(tokenizer("AGCAGTCATTATGGCGAA", return_tensors="pt")["input_ids"])

>>> output.keys()
odict_keys(['last_hidden_state', 'logits'])
```

The `logits` tensor reproduces the original SpTransformer output: a 3-channel splice-site score (no-splice / acceptor / donor) and a per-tissue (15 tissues) splice-site usage score for each position.

### Downstream Use

#### Token Prediction

You can fine-tune SpTransformer for per-nucleotide tissue-specific splicing regression with [`SpTransformerForTokenPrediction`][multimolecule.models.SpTransformerForTokenPrediction], which adds a shared token prediction head on top of the backbone.

### Interface

- **Input length**: variable pre-mRNA sequence
- **Flanking context**: fixed 4,000 nt on each side of every predicted position
- **Padding**: ends padded with `N`
- **Output**: per-position 3-channel splice-site score (`no-splice` / `acceptor` / `donor`) + per-tissue (15 tissues) splice-site usage score
- **Attention recording**: opt-in via `output_attentions=True`; returns faithful sparse-attention maps — see [Faithful Sparse-Attention Exposure](#faithful-sparse-attention-exposure)

### Faithful Sparse-Attention Exposure

SpTransformer's attention block does **not** compute dense self-attention. Each layer
([`SpTransformerSelfAttention`][multimolecule.models.sptransformer.modeling_sptransformer.SpTransformerSelfAttention])
splits its heads into two groups with **fundamentally different sparse-attention structures**:

- **Windowed-local heads** — each window of `bucket_size` tokens attends only to itself plus the immediately
  preceding and following window (a `look_backward=1`, `look_forward=1` look-around). Boundary positions are
  masked.
- **Sinkhorn sorted-bucket heads** — each query bucket attends to the concatenation of (a) one _sorted /
  reordered_ key bucket selected by a parameter-free attention-sort net (`differentiable_topk(R, k=1)`) and
  (b) its own local bucket.

Because these two patterns operate on different key axes, there is **no single dense `(batch, heads,
sequence, sequence)` tensor that faithfully represents the computation**. Materialising a zero-filled
`sequence x sequence` grid would be a _misleading_ interpretability artifact, so this model does **not**
expose one.

Instead, attention recording is **opt-in** and faithful. Passing `output_attentions=True` (or setting
`config.output_attentions=True`) returns, for every attention layer, a
[`SpTransformerAttentionMap`][multimolecule.models.SpTransformerAttentionMap] holding the _actual_ `softmax`
weights used in the forward pass plus the indexing/permutation needed to map them back to absolute sequence
positions:

- `local_attentions` `(B, num_local_heads, num_windows, W, (look_backward + 1 + look_forward) * W)` — the
  real per-window softmax weights; padded look-around columns carry weight `0`.
- `local_key_positions` `(num_windows, (look_backward + 1 + look_forward) * W)` — absolute source position
  of every local key-axis column (`-1` marks padded columns).
- `sinkhorn_attentions` `(B, num_sinkhorn_heads, num_buckets, W, 2 * W)` — the real per-bucket softmax
  weights over the `[reordered-bucket | own-bucket]` key axis.
- `sinkhorn_reorder` `(B, num_sinkhorn_heads, num_buckets, num_buckets)` — the exact bucket-permutation
  matrix; for query bucket `u`, the nonzero column `v` of row `u` says the reordered key bucket (columns
  `0:W` of `sinkhorn_attentions`) is source bucket `v` (absolute positions `v*W : v*W + W`).
- scalar metadata: `bucket_size`, `look_backward`, `look_forward`, `num_local_heads`,
  `num_sinkhorn_heads`, `sequence_length`.

`W` is `bucket_size`; local heads come first along the head axis, Sinkhorn heads second. These are
**structured block weights, not dense attention matrices** — re-deriving the per-type attention output by
contracting these exact weights with the (block-gathered) values reproduces the layer output exactly.
Recording is opt-in, so the default forward path and its numerics are byte-for-byte unchanged.

```python
>>> import torch
>>> from multimolecule import SpTransformerConfig, SpTransformerModel
>>> config = SpTransformerConfig(bucket_size=4, max_seq_len=16, context=2, num_hidden_layers=2)
>>> model = SpTransformerModel(config)
>>> output = model(torch.randint(config.vocab_size, (1, 16)), output_attentions=True)
>>> layer0 = output.attentions[0]
>>> layer0.local_attentions.shape
torch.Size([1, 2, 4, 4, 12])
>>> layer0.sinkhorn_attentions.shape
torch.Size([1, 6, 4, 4, 8])
>>> layer0.sinkhorn_reorder.shape
torch.Size([1, 6, 4, 4])
```

## Training Details

SpTransformer was trained to predict tissue-specific splicing from primary pre-mRNA sequence.

### Training Data

SpTransformer was trained on splicing measurements derived from RNA-seq data across 15 human tissues, using gene annotations from [GENCODE](https://multimolecule.danling.org/datasets/gencode), together with multi-species sequence data.
The two convolutional feature extractors were pre-trained as SpliceAI-style splice-site predictors and remain trainable submodules for downstream fine-tuning.
For each predicted nucleotide, a sequence window centered on that nucleotide was used, with the flanking context padded with `N` (unknown nucleotide) when near transcript ends.

### Training Procedure

#### Pre-training

The model was trained to minimize a combination of cross-entropy loss over splice-site classification and a regression loss over per-tissue splice-site usage, comparing predictions against measurements derived from RNA-seq.

## Citation

```bibtex
@article{You2024,
  author    = {You, Ningyuan and Liu, Chang and Gu, Yuxin and Wang, Rong and Jia, Hanying and Zhang, Tianyun and Jiang, Song and Shi, Jinsong and Chen, Ming and Guan, Min-Xin and Sun, Siqi and Pei, Shanshan and Liu, Zhihong and Shen, Ning},
  title     = {{SpliceTransformer predicts tissue-specific splicing linked to human diseases}},
  journal   = {Nature Communications},
  year      = {2024},
  volume    = {15},
  number    = {1},
  pages     = {9129},
  month     = {oct},
  doi       = {10.1038/s41467-024-53088-6},
  issn      = {2041-1723},
  url       = {https://doi.org/10.1038/s41467-024-53088-6}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If you use MultiMolecule in your research, you must cite the MultiMolecule project as follows:

```bibtex
@software{chen_2024_12638419,
  author    = {Chen, Zhiyuan and Zhu, Sophia Y.},
  title     = {MultiMolecule},
  doi       = {10.5281/zenodo.12638419},
  publisher = {Zenodo},
  url       = {https://doi.org/10.5281/zenodo.12638419},
  year      = 2024,
  month     = may,
  day       = 4
}
```

## Contact

Please use GitHub issues of [MultiMolecule](https://github.com/DLS5-Omics/multimolecule/issues) for any questions or comments on the model card.

Please contact the authors of the [SpliceTransformer paper](https://doi.org/10.1038/s41467-024-53088-6) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
