---
language: rna
tags:
  - Biology
  - RNA
license: agpl-3.0
datasets:
  - RNAStrAlign
  - bpRNA
library_name: multimolecule
pipeline_tag: other
pipeline: rna-secondary-structure
---

# BPfold

Pre-trained model for RNA secondary structure prediction using base pair motif energy.

## Disclaimer

This is an UNOFFICIAL implementation of [Deep generalizable prediction of RNA secondary structure via base pair motif energy](https://doi.org/10.1038/s41467-025-60048-1) by Heqin Zhu, Fenghe Tang, Quan Quan, Ke Chen, Peng Xiong, and S. Kevin Zhou.

The OFFICIAL repository of BPfold is at [heqin-zhu/BPfold](https://github.com/heqin-zhu/BPfold).

> [!TIP]
> The MultiMolecule implementation preserves the released BPfold architecture, base-pair motif energy feature construction, and canonical/non-canonical post-processing semantics.

**The team releasing BPfold did not write this model card for this model so this model card has been written by the MultiMolecule team.**

## Model Details

BPfold predicts RNA base-pair contact maps from a single RNA sequence. It augments a transformer encoder with two `L x L` base-pair motif energy maps computed from three-neighbor base-pair motifs. MultiMolecule exposes BPfold as a single checkpoint and stores the motif-energy lookup tables inside it.

The model uses:

- **token order**: follows the MultiMolecule tokenizer.
- **unknown bases**: tokenized as `N` and treated as `U` during BPfold feature construction, matching the upstream fallback; padding follows `attention_mask`.
- **self-attention**: dynamic position bias with adjacency bias from motif-energy maps.
- **pairwise convolutions**: three residual 2D convolution layers over the adjacency maps before the transformer blocks.
- **post-processing**: constrained refinement for canonical pairs, plus the optional BPfold non-canonical pass and mixed canonical/non-canonical outputs.

### Model Specification

| Num Layers | Hidden Size | Num Parameters (M) | FLOPs (G) | MACs (G) |
| ---------- | ----------- | ------------------ | --------- | -------- |
| 12         | 256         | 47.77              | 87.78     | 42.74    |

FLOPs and MACs are computed with `multimolecule.utils` for one 600 nt sequence.

### Links

- **Code**: [multimolecule.bpfold](https://github.com/DLS5-Omics/multimolecule/tree/master/multimolecule/models/bpfold)
- **Paper**: [Deep generalizable prediction of RNA secondary structure via base pair motif energy](https://doi.org/10.1038/s41467-025-60048-1)
- **Developed by**: Heqin Zhu, Fenghe Tang, Quan Quan, Ke Chen, Peng Xiong, S. Kevin Zhou
- **Original Repository**: [heqin-zhu/BPfold](https://github.com/heqin-zhu/BPfold)

## Usage

The model file depends on the [`multimolecule`](https://multimolecule.danling.org) library. You can install it using pip:

```bash
pip install multimolecule
```

### RNA Secondary Structure Pipeline

```python
import multimolecule
from transformers import pipeline

predictor = pipeline("rna-secondary-structure", model="multimolecule/bpfold")
output = predictor("GGUAAAACAGCCUGU")
```

### PyTorch Inference

```python
from multimolecule import BpfoldModel, RnaTokenizer

tokenizer = RnaTokenizer.from_pretrained("multimolecule/bpfold")
model = BpfoldModel.from_pretrained("multimolecule/bpfold")
input = tokenizer("GGUAAAACAGCCUGU", return_tensors="pt")

output = model(**input)
contact_map = output.contact_map  # (1, L, L) base-pair probability matrix
```

## Training Details

BPfold was trained for RNA secondary structure prediction with base-pair motif energy priors.

### Training Data

- RNAStrAlign: 37,149 RNAs from eight RNA families were filtered to remove redundant sequences and invalid secondary structures, yielding 29,647 unique RNAs. Sequences longer than 600 nt were removed for training, leaving 19,313 training RNAs.
- bpRNA-1m: 102,318 RNAs from 2,588 families were deduplicated with CD-HIT at 80% sequence identity and split into TR0/TS0 with 12,114/1,305 RNAs.
- evaluation data: ArchiveII contains 3,966 RNAs; Rfam12.3-14.10 contains 10,791 RNAs from 1,992 families; bpRNA-new contains 5,401 RNAs; PDB contains 116 high-resolution RNAs split into TS1/TS2/TS3.

### Training Procedure

- objective: binary cross entropy over base-pair contact maps.
- optimizer: Adam.
- learning rate: 5e-4.
- training epochs: 150.
- batch size: 48.
- positive-class weight: 300.
- batching: length-matching mini-batches to reduce padding.
- sequence features: token embeddings converted to the MultiMolecule tokenizer order.
- structural priors: two `L x L` energy maps from three-neighbor base-pair motifs.
- post-processing: constrained refinement for canonical pairs, minimum loop length, non-overlapping pairs, and isolated-pair removal.

## Citation

```bibtex
@article{zhu2025bpfold,
  title   = {Deep generalizable prediction of {RNA} secondary structure via base pair motif energy},
  author  = {Zhu, Heqin and Tang, Fenghe and Quan, Quan and Chen, Ke and Xiong, Peng and Zhou, S. Kevin},
  journal = {Nature Communications},
  volume  = {16},
  number  = {1},
  pages   = {5856},
  year    = {2025},
  doi     = {10.1038/s41467-025-60048-1},
  url     = {https://doi.org/10.1038/s41467-025-60048-1}
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
