---
authors:
  - Zhiyuan Chen
date: 2025-06-12
---

# api

`api` 提供面向用户的入口，将 [`Runner`][multimolecule.runner.Runner] 包装成 CLI 友好的形式，
并将 `train`、`evaluate`、`infer` 同时暴露为 Python 函数和控制台脚本。

## 控制台脚本

| 命令         | 函数                                     | 说明                                  |
| ------------ | ---------------------------------------- | ------------------------------------- |
| `mmtrain`    | [`train`][multimolecule.api.train]       | 强制 `config.training = True`。       |
| `mmevaluate` | [`evaluate`][multimolecule.api.evaluate] | 强制 `config.training = False`。      |
| `mminfer`    | [`infer`][multimolecule.api.infer]       | 将预测结果写入 `config.result_path`。 |

每个命令都通过 [Chanfig](https://chanfig.danling.org) 解析参数。
默认配置文件是当前工作目录下的 `config.yaml`；
命令行参数会覆盖单个字段：

```shell
mmtrain optim.lr=2e-4 epochs=5 data.root=data/my-task
```

`mminfer` 在 `result_path` 未显式设置时默认写入 `./result.json`，并发出 `RuntimeWarning`。

## 工作目录契约

在构建 runner 之前，[`dynamic_import`][multimolecule.api.run.dynamic_import] 会把 `cwd` 的父目录加入
`sys.path`，并将 `os.path.basename(cwd)` 作为 Python 模块导入。
若 `config.pretrained` 与 `cwd` 下的某个子目录同名，该子目录也会被加入 `sys.path` 并导入。

这是自定义数据集、模型与 runner 在 runner 构建之前注册自身的方式。
因此，工作目录必须是一个可导入的 Python 包（即包含 `__init__.py`）。

## 程序内调用

同一组入口同样可在 Python 中调用：

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

`train` / `evaluate` / `infer` 始终对传入的 config 执行 `Config.parse`，
因此 `config.yaml`（如果存在）和命令行覆盖仍会被应用。
将 `config=None` 表示从空 `Config` 开始，完全依赖 YAML 文件加上命令行参数。

如果想完全绕过 CLI 编排（不打印 banner、不进入 [`dl.debug`][danling.debug] 上下文、不集成 NNI），
可以直接通过 [`RUNNERS`][multimolecule.runner.registry.RUNNERS] 构建 runner —— 详见
[runner](../runner) 页面。

## 可选标志

- `debug=true` —— 在 [`dl.debug`][danling.debug] 上下文中运行，打印更详细的追踪信息。
- `nni=true` —— 通过 `nni.get_next_parameter()` 读取超参并合并到 config；需要安装 `nni` 包。

## 结果汇总

`multimolecule.api.stat` 会遍历实验根目录，解析每次运行的 `best.json` 与 `trainer.yaml`，
生成 CSV 汇总。作为脚本运行：

```shell
python -m multimolecule.api.stat experiment_root=experiments out_path=result.csv
```
