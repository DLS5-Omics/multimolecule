---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0-or-later
library_name: multimolecule
pipeline_tag: other
pipeline: rna-secondary-structure
---

# UFold

Pre-trained model for RNA secondary structure prediction using an image-like sequence representation and a U-Net.

## Disclaimer

This is an UNOFFICIAL implementation of [UFold: fast and accurate RNA secondary structure prediction with deep learning](https://doi.org/10.1093/nar/gkab1074) by Laiyi Fu, Yingxin Cao, Jie Wu, Qinke Peng, Qing Nie, and Xiaohui Xie.

The OFFICIAL repository of UFold is at [uci-cbcl/UFold](https://github.com/uci-cbcl/UFold).

> [!TIP]
> The MultiMolecule implementation is a direct PyTorch port of the original U-Net architecture and feature construction.

**The team releasing UFold did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

UFold predicts RNA base-pair contact maps from single RNA sequences. It represents a sequence as a 17-channel image: 16 channels are outer products of one-hot nucleotide indicators and one channel is a hand-crafted canonical/wobble pairing score. A U-Net predicts a symmetric contact score matrix, and the original constrained post-processing routine can be enabled to enforce base-pairing constraints.

### Model Specification

| Num Parameters (M) | FLOPs (G) | MACs (G) |
| ------------------ | --------- | -------- |
| 8.64               | 188.29    | 93.81    |

FLOPs and MACs are computed with `multimolecule.utils` for one 600 nt sequence.

### Links

- **Code**: [multimolecule.ufold](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/ufold)
- **Weights**: [multimolecule/ufold](https://huggingface.co/multimolecule/ufold)
- **Paper**: [UFold: fast and accurate RNA secondary structure prediction with deep learning](https://doi.org/10.1093/nar/gkab1074)
- **Developed by**: Laiyi Fu, Yingxin Cao, Jie Wu, Qinke Peng, Qing Nie, Xiaohui Xie
- **Original Repository**: [uci-cbcl/UFold](https://github.com/uci-cbcl/UFold)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### RNA Secondary Structure Pipeline

```python
import multimolecule
from transformers import pipeline

predictor = pipeline("rna-secondary-structure", model="multimolecule/ufold")
output = predictor("GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCUCA")
```

### PyTorch Inference

```python
from multimolecule import RnaTokenizer, UfoldModel

tokenizer = RnaTokenizer.from_pretrained("multimolecule/ufold")
model = UfoldModel.from_pretrained("multimolecule/ufold")

sequence = "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCUCA"
inputs = tokenizer(sequence, return_tensors="pt")
output = model(**inputs)

contact_map = output.contact_map
```

To run the original constrained post-processing loop:

```python
output = model(**inputs, use_postprocessing=True)
contact_map = output.postprocessed_contact_map
```

## Training Details

UFold was trained for RNA secondary structure prediction from annotated contact maps and base-pairing rules.

### Training Data

- RNAStrAlign: 30,451 unique RNAs from eight RNA families; the paper reports a random split with 24,895 training RNAs and 2,854 test RNAs after redundancy filtering.
- bpRNA-1m: 102,318 RNAs from 2,588 families; CD-HIT was used to remove redundant sequences before splitting the data into TR0 and TS0.
- augmented data: synthetic training examples were generated from bpRNA-new sequences by random mutation and structure prediction.
- PDB training data: high-resolution RNA structures from bpRNA and the PDB were used for fine-tuning/evaluation experiments; test sets TS1, TS2, and TS3 were filtered at 80% sequence identity.
- evaluation data: ArchiveII, TS0, bpRNA-new, and PDB test data were used for benchmark evaluation.

### Training Procedure

- input representation: 16 outer-product channels following the MultiMolecule tokenizer order plus one hand-crafted pairing-score channel.
- objective: weighted binary cross entropy over base-pair contact maps.
- optimizer: Adam.
- training epochs: 100.
- batch size: 1.
- positive-class weight: 300.
- post-processing: constrained optimization with canonical/wobble pairing rules, sparsity shrinkage, and a 0.5 threshold.

## Citation

```bibtex
@article{fu2022ufold,
  author = {Fu, Laiyi and Cao, Yingxin and Wu, Jie and Peng, Qinke and Nie, Qing and Xie, Xiaohui},
  title = {UFold: fast and accurate RNA secondary structure prediction with deep learning},
  journal = {Nucleic Acids Research},
  volume = {50},
  number = {3},
  pages = {e14},
  year = {2022},
  doi = {10.1093/nar/gkab1074}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the MultiMolecule project.
> If you use MultiMolecule in your research, you must cite the MultiMolecule project.

## License

This model is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
