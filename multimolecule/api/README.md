---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# api

`api` provides the user-facing entry points that wrap the [`Runner`][multimolecule.runner.Runner] in a CLI-friendly
way and expose `train`, `evaluate`, and `infer` as both Python functions and console scripts.

## Console Scripts

| Command      | Function                                 | Notes                                       |
| ------------ | ---------------------------------------- | ------------------------------------------- |
| `mmtrain`    | [`train`][multimolecule.api.train]       | Forces `config.training = True`.            |
| `mmevaluate` | [`evaluate`][multimolecule.api.evaluate] | Forces `config.training = False`.           |
| `mminfer`    | [`infer`][multimolecule.api.infer]       | Writes predictions to `config.result_path`. |

Each command parses arguments through [Chanfig](https://chanfig.danling.org). The default config file is `config.yaml`
in the current working directory; CLI arguments override individual fields:

```shell
mmtrain optim.lr=2e-4 epochs=5 data.root=data/my-task
```

`mminfer` defaults `result_path` to `./result.json` and emits a `RuntimeWarning` if the field is not set explicitly.

## Working Directory Contract

Before building the runner, [`dynamic_import`][multimolecule.api.run.dynamic_import] appends the parent of `cwd` to
`sys.path` and imports `os.path.basename(cwd)` as a Python module. If `config.pretrained` matches a sub-directory of
`cwd`, that sub-directory is also appended and imported.

This is how custom datasets, models, and runners register themselves before the runner is built. The working
directory must therefore be importable as a Python package (i.e. contain an `__init__.py`).

## Programmatic Usage

The same entry points are usable from Python:

```python
from multimolecule.api import train
from multimolecule.runner import Config

config = Config(
    pretrained="multimolecule/rnafm",
    data={"root": "data/my-rna-task", "sequence_cols": ["sequence"], "label_cols": ["activity"]},
    optim={"lr": 1.0e-4, "weight_decay": 1.0e-2, "pretrained_ratio": 1.0e-2},
    epochs=10,
)
train(config)
```

`train` / `evaluate` / `infer` always run `Config.parse` on the provided config, so `config.yaml` (when present) and
CLI overrides are still applied. Pass `config=None` to start from an empty `Config` and rely entirely on the YAML
file plus CLI arguments.

To bypass the CLI orchestration entirely (no banner, no [`dl.debug`][danling.debug] context, no NNI integration),
build the runner directly through [`RUNNERS`][multimolecule.runner.registry.RUNNERS] — see the [runner](../runner)
page.

## Optional Flags

- `debug=true` — wrap the run in [`dl.debug`][danling.debug] for verbose tracing.
- `nni=true` — read hyperparameters from `nni.get_next_parameter()` and merge them into the config; requires the
  `nni` package.

## Result Aggregation

`multimolecule.api.stat` walks an experiment root, parses each run's `best.json` and `trainer.yaml`, and produces a
CSV summary. Run as a script:

```shell
python -m multimolecule.api.stat experiment_root=experiments out_path=result.csv
```
