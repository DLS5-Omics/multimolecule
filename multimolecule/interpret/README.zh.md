# 模型解释

本快速入门用两种方法将 RNA-FM 的预测分数归因到输入 token。示例使用随机初始化的小型模型和两条合成序列，可在 CPU 上运行。

## 运行示例

在仓库根目录安装模型解释的可选依赖并运行脚本。

```bash
python -m pip install -e ".[interpret]"
python examples/interpret/quickstart.py
```

`output="logits"` 选择模型返回的张量，`target=1` 选择每个样本的第二个 logit。脚本将 `interpret-quickstart.png` 保存到当前目录，用统一的零中心颜色刻度展示第一条序列的两种归因结果。

```python
--8<-- "examples/interpret/quickstart.py:23:"
```

## 解读分数

- 用 `layer_integrated_gradients` 沿参考嵌入到输入嵌入的可微路径做归因。示例将返回的 `[B, L, D]` 沿嵌入维度求和，得到 `[B, L]` 的有符号 token 分数。收敛残差衡量分数总和与「输入 logit 减去参考 logit」有多接近。
- 用 `feature_ablation` 从原序列的 logit 减去替换一个 token 后的 logit。它可以直接处理整数 token ID，本例返回 `[B, L]`。token 之间的相互作用可能同时影响多次消融，因此两种方法的分数可能不同。
- 用 `model.eval()` 切换到评估模式，例如关闭 dropout。在 `torch.inference_mode()` 之外运行 Layer Integrated Gradients，以便计算梯度。
- 参考输入用 `<unk>` 替换序列 token，保留特殊 token、padding 和 attention mask。用 `<pad>` 替换序列 token 还可能改变 RNA-FM 自动推导的位置编号。脚本用 `special_tokens_mask` 选择构造参考和绘图的位置，只把 `input_ids` 和 `attention_mask` 传给模型。
