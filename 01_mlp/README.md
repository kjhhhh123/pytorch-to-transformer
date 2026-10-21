# MLP 训练代码练习

五部分 PyTorch / Transformer 代码学习的第二部分。目标是看清数据、模型、Loss、梯度、优化器和验证之间的数据流。

## 文件

- `train.py`：数据生成、DataLoader、MLP、训练、验证、保存和重载。
- `../requirements.txt`：项目统一依赖版本。
- `outputs/`：运行时生成权重、逐轮指标和结果摘要；已忽略，不提交 Git。

## 运行

在 `pytorch-to-transformer` 项目根目录，使用已安装依赖的环境：

```zsh
.venv/bin/python 01_mlp/train.py --device cpu
```

首次配置环境见 [项目说明](../README.md#环境与运行)，依赖文件位于项目根目录。

默认使用 CPU、float32、seed=42，训练 30 个 epoch；已有环境时不必重新安装。
Mac GPU 需要当前进程中 MPS 可用，再显式选择 `--device mps`；CUDA 环境在 NVIDIA 机器上单独配置。

## 任务与数据流

数据由脚本生成：三个二维高斯簇，每类 200 个样本；固定划分 480 个训练样本和 120 个验证样本，无需下载数据。

```text
输入 [B, 2]，标签 [B]
-> Linear(2, 16) -> ReLU -> Linear(16, 3)
-> logits [B, 3] -> CrossEntropyLoss
-> backward 计算梯度 -> Adam.step 更新参数
```

默认网络包含 99 个可训练参数。标签为 long，输入和模型参数为 float32；交叉熵直接接收 logits。
验证不反向传播、不更新参数。这个纯 Linear/ReLU 模型的 train/eval 不改变前向行为，但保留两种模式的规范调用。

该数据只用于学习训练流程；验证结果不代表真实业务能力，也没有单独的最终测试集。
保存的是最后一轮权重。重载后检查同一批输入的输出是否近似一致。

## 参考

本例使用自建数据，参考 PyTorch 官方代码接口与训练流程，不复现原教程的 FashionMNIST 实验：

- [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save and Load the Model](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
