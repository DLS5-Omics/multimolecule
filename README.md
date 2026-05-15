# [MultiMolecule](https://multimolecule.danling.org)

> [!TIP]
> Accelerate Molecular Biology Research with Machine Learning.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119050.svg)](https://doi.org/10.5281/zenodo.15119050)

[![Codacy - Quality](https://app.codacy.com/project/badge/Grade/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy - Coverage](https://app.codacy.com/project/badge/Coverage/ad5fd8904c2e426bb0a865a9160d6c69)](https://app.codacy.com/gh/DLS5-Omics/multimolecule/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![CodeCov - Coverage](https://codecov.io/gh/DLS5-Omics/multimolecule/graph/badge.svg?token=G9WGWCOFQE)](https://codecov.io/gh/DLS5-Omics/multimolecule)

[![PyPI - Version](https://img.shields.io/pypi/v/multimolecule)](https://pypi.org/project/multimolecule)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multimolecule)](https://pypi.org/project/multimolecule)
[![Downloads Statistics](https://static.pepy.tech/badge/multimolecule/month)](https://multimolecule.danling.org)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

MultiMolecule is a one-stop ecosystem for molecular machine learning.
It connects datasets, model implementations, reusable dataset and neural-network modules, the DanLing-based runner for training and evaluation, and task-oriented inference pipelines for RNA, DNA, and protein workflows.

## Get Started

Install the latest stable release from PyPI:

```shell
pip install multimolecule
```

Run a registered pipeline through the Hugging Face `transformers` interface:

```python
import multimolecule  # registers MultiMolecule models and pipelines
from transformers import pipeline

predictor = pipeline("rna-secondary-structure", model="multimolecule/ernierna-ss")
result = predictor("AUCAGCCUUCGUUCUGUAAACGG")
```

Load models directly when you need lower-level control:

```python
import multimolecule

model = multimolecule.AutoModelForSequencePrediction.from_pretrained("multimolecule/basset")
tokenizer = multimolecule.AutoTokenizer.from_pretrained("multimolecule/basset")
```

Install the latest source version when you need unreleased changes:

```shell
pip install git+https://github.com/DLS5-Omics/MultiMolecule
```

## Explore

| Entry point | Use it for |
| --- | --- |
| [`data`](data) | Task-aware datasets, data loading, and multi-task sampling. |
| [`datasets`](datasets) | Curated biomolecular datasets and task metadata. |
| [`io`](io) | FASTA, DBN, BPSEQ, and bpRNA ST readers and writers. |
| [`models`](models) | Model cards and API references for supported architectures. |
| [`tokenisers`](tokenisers) | DNA, RNA, protein, and dot-bracket tokenisers. |
| [`pipelines`](pipelines) | Task-focused inference workflows for supported biological tasks. |
| [`runner`](runner) | Training, evaluation, and inference configuration. |
| [`modules`](modules) | Reusable neural-network building blocks. |

## Community

- [Discourse](https://multimolecule.discourse.group): release announcements, usage questions, model requests, RFCs, and community discussion.
- [GitHub Issues](https://github.com/DLS5-Omics/multimolecule/issues): reproducible bugs, API issues, and implementation-tracked feature requests.
- [Hugging Face](https://huggingface.co/multimolecule): released checkpoints, datasets, and demo Spaces.

## Citation

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

## License

We believe openness is the Foundation of Research.

MultiMolecule is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
