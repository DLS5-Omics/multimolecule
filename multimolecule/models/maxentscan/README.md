---
language: rna
tags:
  - Biology
  - RNA
  - Splicing
license: agpl-3.0
library_name: multimolecule
pipeline_tag: other
pipeline: splice-site
---

# MaxEntScan

Maximum-entropy model for scoring short sequence motifs at RNA splice sites.

## Disclaimer

This is an UNOFFICIAL implementation of [Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals](https://doi.org/10.1089/1066527041410418) by Gene Yeo, et al.

The OFFICIAL distribution of MaxEntScan is at [the Burge Lab MaxEntScan page](http://hollywood.mit.edu/burgelab/maxent/Xmaxentscan_scoreseq.html).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing MaxEntScan did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

MaxEntScan is a maximum-entropy model for the splice donor (5') and splice acceptor (3') sequence motifs. It is **not a neural network** and has **no trainable weights**. The model parameters are fixed maximum-entropy probability tables estimated by Yeo & Burge (2004) from human splice-site sequences.

### Model Specification

MaxEntScan is a parameter-free maximum-entropy model. It performs fixed table lookups and contains no learnable weights or floating-point arithmetic that the profiler can attribute to a module. The bundled score tables that serve as the model's fixed parameters are:

- **`score5`**: a single 16,384-entry `me2x5` probability table (4⁷ floats) indexed by the base-4 hash of the 7 non-consensus positions of the 9-mer.
- **`score3`**: nine overlapping maximum-entropy decomposition tables (`me2x3acc1..9`) with sizes 4⁷, 4⁷, 4⁷, 4⁷, 4⁷, 4³, 4⁴, 4³, 4⁴ (5 × 16384 + 2 × 64 + 2 × 256 = 82560 floats total).

<table>
<thead>
  <tr>
    <th>Mode</th>
    <th>Window</th>
    <th>Num Parameters (M)</th>
    <th>FLOPs (G)</th>
    <th>MACs (G)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>score5</td>
    <td>9</td>
    <td rowspan="2">0.00</td>
    <td rowspan="2">0.00</td>
    <td rowspan="2">0.00</td>
  </tr>
  <tr>
    <td>score3</td>
    <td>23</td>
  </tr>
</tbody>
</table>

### Links

- **Code**: [multimolecule.maxentscan](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/maxentscan)
- **Data**: Human RefSeq splice-site sequences curated by Yeo and Burge
- **Paper**: [Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals](https://doi.org/10.1089/1066527041410418)
- **Developed by**: Gene Yeo, Christopher B. Burge
- **Model type**: Maximum-entropy splice-site scoring with fixed probability tables for 5' and 3' splice sites
- **Original Repository**: [Burge Lab MaxEntScan](http://hollywood.mit.edu/burgelab/maxent/Xmaxentscan_scoreseq.html)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### 5' Splice-Site Scoring

```python
>>> import torch
>>> from multimolecule import RnaTokenizer, MaxEntScanModel, MaxEntScanConfig

>>> config = MaxEntScanConfig()
>>> model = MaxEntScanModel(config)
>>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/maxentscan-score5")
>>> # MaxEntScan scores a raw fixed-length window; do not add special tokens.
>>> input = tokenizer("CAGGUAAGU", add_special_tokens=False, return_tensors="pt")["input_ids"]
>>> output = model(input)
>>> output.logits.shape
torch.Size([1, 1])
```

#### 3' Splice-Site Scoring

```python
>>> config = MaxEntScanConfig(mode="score3")
>>> model = MaxEntScanModel(config)
>>> output = model(torch.randint(4, (1, config.window)))
>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: 9 nt fixed window for `score5`; 23 nt fixed window for `score3`
- **Alphabet**: `ACGU` only; unknown / `N` tokens are clamped onto `A` before table lookup
- **Special tokens**: do not add (`add_special_tokens=False`)
- **`inputs_embeds`**: not supported; the model scores discrete token windows only
- **Output**: single scalar splice-site log-odds score per window

## Training Details

MaxEntScan is not trained. Its maximum-entropy probability tables were estimated once by Yeo & Burge (2004) from a set of human constitutive splice-site sequences using an iterative maximum-entropy procedure. The published tables are reused verbatim.

### Scoring Modes

- `score5`: scores 5' (donor) splice sites over a 9-nucleotide window (3 exonic + 6 intronic nucleotides). The score is read from the published `me2x5` maximum-entropy probability table combined with the consensus background ratios.
- `score3`: scores 3' (acceptor) splice sites over a 23-nucleotide window. The 23-mer is decomposed into nine overlapping maximum-entropy submodels following the published maximum-entropy decomposition; the score is the log-ratio of the numerator and denominator submodel products.

### Training Data

- Source: human RefSeq splice-site sequences as described in Yeo & Burge (2004).
- Maximum-entropy constraints: pairwise and higher-order positional dependencies within the splice-site window.

The model parameters are the fixed maximum-entropy probability tables distributed as plain-text files with the original Yeo & Burge (2004) MaxEntScan tool: `me2x5` for the 5' scorer and the nine maximum-entropy decomposition matrices `me2x3acc1..9` for the 3' scorer. The consensus and background ratios are fixed constants from the original `score5.pl` and `score3.pl` programs.

### Training Procedure

#### Pre-training

MaxEntScan does not use neural-network pre-training. Its maximum-entropy probability tables are reused from the original MaxEntScan distribution.

## Citation

```bibtex
@article{yeo2004maximum,
  author    = {Yeo, Gene and Burge, Christopher B.},
  title     = {Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals},
  journal   = {Journal of Computational Biology},
  volume    = {11},
  number    = {2-3},
  pages     = {377--394},
  year      = {2004},
  publisher = {Mary Ann Liebert, Inc.},
  doi       = {10.1089/1066527041410418}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If MultiMolecule supports your research, please cite the MultiMolecule project as follows:

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

Please contact the authors of the [MaxEntScan paper](https://doi.org/10.1089/1066527041410418) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
