# PyTorch to Transformer

通过小型可运行示例，学习模型的数据流、训练循环与 Transformer 模块实现。

## 目录

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| [01_mlp](01_mlp/README.md) | 二维三分类 MLP：数据、训练、验证、保存与加载 | 已有可运行代码 |
| 02_attention | Attention 与 Multi-Head Attention | 待实现 |
| 03_transformer | FFN 与 Transformer Block | 待实现 |
| 04_tiny_gpt | 小 GPT 训练与生成 | 待实现 |

## 环境与运行

在本项目根目录执行。当前示例已在 Python 3.13.16、PyTorch 2.9.1、macOS arm64 CPU 上验证。

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 01_mlp/train.py --device cpu
```

已有兼容的 Python / PyTorch 环境时，也可用该环境的解释器运行 `01_mlp/train.py`。
默认 CPU、float32；数据由脚本生成，无需下载。

运行结果写入 `01_mlp/outputs/`。虚拟环境、缓存、输出目录和模型权重已加入 `.gitignore`。
