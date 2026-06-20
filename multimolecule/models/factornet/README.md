---
language: dna
tags:
  - Biology
  - DNA
license: agpl-3.0
library_name: multimolecule
pipeline_tag: regulatory-activity
---

# FactorNet

Convolutional + bidirectional LSTM model for predicting cell-type-specific transcription-factor binding from one-hot DNA sequence augmented with per-position DNase-seq and per-window RNA-seq features.

## Disclaimer

This is an UNOFFICIAL implementation of [FactorNet: A deep learning framework for predicting cell type specific transcription factor binding from nucleotide-resolution sequential data](https://doi.org/10.1016/j.ymeth.2019.03.020) by **Daniel Quang and Xiaohui Xie**.

The OFFICIAL repository of FactorNet is at [uci-cbcl/FactorNet](https://github.com/uci-cbcl/FactorNet).

> [!TIP]
> The MultiMolecule team has confirmed that the provided model and checkpoints are producing the same intermediate representations as the original implementation.

**The team releasing FactorNet did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

FactorNet is a Siamese convolutional + bidirectional LSTM model that predicts per-cell-type transcription-factor (TF) binding from a fixed-length DNA window (1002 bp by default) jointly with cell-type-specific signal channels. The same convolution / BLSTM / dense stack is applied to the forward strand and to its reverse complement; the two pooled representations are concatenated with per-window metadata features (RNA-seq principal components, etc.), projected through a fully-connected layer, and averaged to produce the final per-TF binding probability. Per-position auxiliary signals (DNase-seq cleavage `DGF`, mappability `Unique35`) are concatenated with the one-hot DNA tensor on the channel axis before the first convolution; per-window metadata features are fused after the flatten layer. This combination of sequence + cell-type-specific tracks is what distinguishes FactorNet from purely sequence-based TF-binding models such as DeepBind. Please refer to the [Training Details](#training-details) section for more information on the training process.

### Variants

A separate FactorNet model is released for every TF / cell-type pairing in the ENCODE-DREAM challenge. The MultiMolecule port preserves the upstream architecture as a single `FactorNetModel` class and exposes each trained TF / cell-type model as its own Hugging Face Hub checkpoint:

<table>
<thead><tr><th>Transcription factor</th><th>Variant</th><th>Hub repository</th></tr></thead>
<tbody>
<tr><td>CTCF</td><td>meta_RNAseq_Unique35_DGF</td><td>multimolecule/factornet-ctcf</td></tr>
<tr><td>E2F1</td><td>onePeak_Unique35_DGF</td><td>multimolecule/factornet-e2f1</td></tr>
<tr><td>FOXA1</td><td>multiTask_DGF</td><td>multimolecule/factornet-foxa1-multitask</td></tr>
<tr><td>JUND</td><td>meta_Unique35_DGF</td><td>multimolecule/factornet-jund</td></tr>
<tr><td>NANOG</td><td>onePeak_Unique35_DGF</td><td>multimolecule/factornet-nanog</td></tr>
<tr><td>REST</td><td>GENCODE_Unique35_DGF</td><td>multimolecule/factornet-rest</td></tr>
</tbody>
</table>

### Model Specification

| Conv Kernels | Conv Width | LSTM Hidden | FC Hidden | Num Parameters (M) | FLOPs (M) | MACs (M) | Max Num Tokens |
| ------------ | ---------- | ----------- | --------- | ------------------ | --------- | -------- | -------------- |
| 128          | 34         | 64          | 128       | 1.09               | 192.66    | 95.47    | 1002           |

### Links

- **Code**: [multimolecule.factornet](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/factornet)
- **Weights**: [multimolecule/factornet](https://huggingface.co/multimolecule/factornet)
- **Data**: [ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge](https://www.synapse.org/Synapse:syn6131484)
- **Paper**: [FactorNet: A deep learning framework for predicting cell type specific transcription factor binding from nucleotide-resolution sequential data](https://doi.org/10.1016/j.ymeth.2019.03.020)
- **Developed by**: Daniel Quang, Xiaohui Xie
- **Model type**: Siamese 1D CNN + BLSTM with auxiliary per-position DNase-seq and per-window RNA-seq features for cell-type-specific TF binding prediction
- **Original Repository**: [uci-cbcl/FactorNet](https://github.com/uci-cbcl/FactorNet)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### Direct Use

#### Transcription Factor Binding Prediction

You can use this model directly to predict the binding probability of its target transcription factor at a 1002 bp DNA window:

```python
>>> import torch
>>> from multimolecule import DnaTokenizer, FactorNetForSequencePrediction

>>> tokenizer = DnaTokenizer.from_pretrained("multimolecule/factornet")
>>> model = FactorNetForSequencePrediction.from_pretrained("multimolecule/factornet")
>>> sequence = "ACGT" * 250 + "AC"  # 1002 bp window
>>> input = tokenizer(sequence, return_tensors="pt")
>>> auxiliary_signal = torch.randn(1, 1002, model.config.num_auxiliary_signals)
>>> metadata_features = torch.randn(1, model.config.num_metadata_features)
>>> output = model(**input, auxiliary_signal=auxiliary_signal, metadata_features=metadata_features)

>>> output.logits.shape
torch.Size([1, 1])
```

### Interface

- **Input length**: fixed 1002 bp DNA window
- **Channels**: 4 one-hot DNA channels plus `num_auxiliary_signals` per-position auxiliary signal channels (DNase-seq cleavage, mappability) supplied as the `auxiliary_signal` kwarg; the reverse-complement strand is computed inside the model
- **Auxiliary inputs**: `metadata_features` (per-window RNA-seq principal components and similar features, shape `(batch_size, num_metadata_features)`) fused after the convolutional flatten layer
- **Output**: per-TF binding probability logits of shape `(batch_size, num_labels)`; apply `model.postprocess(output)` (or `torch.sigmoid`) to recover the upstream `[0, 1]` probabilities

## Training Details

FactorNet was trained as part of the [ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge](https://www.synapse.org/Synapse:syn6131484), with a separate model trained for each (TF, cell type) pair.

### Training Data

Per-TF training labels come from ENCODE ChIP-seq peaks aggregated across the cell types released with the ENCODE-DREAM challenge. Each 1002 bp genomic window is labeled with a binary vector indicating whether the target TF is bound. The DNase-seq cleavage track (`DGF`) and the 35 bp mappability track (`Unique35`) are aligned to the same genomic windows and supplied as per-position auxiliary channels. The optional per-window metadata features are RNA-seq principal components (the first eight `GEPC1`...`GEPC8`) derived from the cell type's expression profile.

### Training Procedure

#### Pre-training

The model was trained to minimize a per-TF binary cross-entropy loss, comparing the predicted binding probability against the observed ChIP-seq label.

- Optimizer: Adam
- Learning rate: 1e-3
- Loss: Binary cross-entropy
- Negative-to-positive ratio: 1
- Regularization: Dropout (0.1 after the first convolution; 0.5 between the BLSTM and the dense layers)

## Citation

```bibtex
@article{quang2019factornet,
  author    = {Quang, Daniel and Xie, Xiaohui},
  title     = {FactorNet: A deep learning framework for predicting cell type specific transcription factor binding from nucleotide-resolution sequential data},
  journal   = {Methods},
  volume    = 166,
  pages     = {40--47},
  year      = 2019,
  publisher = {Elsevier},
  doi       = {10.1016/j.ymeth.2019.03.020}
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

Please contact the authors of the [FactorNet paper](https://doi.org/10.1016/j.ymeth.2019.03.020) for questions or comments on the paper/model.

## License

This model implementation is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
