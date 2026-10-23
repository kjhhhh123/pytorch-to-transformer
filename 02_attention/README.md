# Attention / Multi-Head Attention

本阶段沿真实开源 MHA 实现观察数据流，并验证因果遮挡、梯度和一次参数更新。
这不是完整语言模型训练；Transformer Block、下一 Token 预测训练与生成在后续目录继续。

## 先打开什么

1. 打开 `mha.py`：先读 `__init__`，再沿 `forward` 看原实现。
2. 打开 `demo_attention.py`：先看 `make_example` 和 `set_example_parameters`，明确输入和全部参数从哪里来。
3. 运行演示，对照终端的 1–14 步看 Q/K/V、轴交换、遮挡、汇总、合并和输出。
4. 看 `check_parameter_update`，区分 `backward()` 计算梯度与 `optimizer.step()` 更新参数。

`observe_forward` 是辅助打印工具：临时观察原 `forward` 的局部变量，不替代原计算、不改变原计算图；本阶段不需要掌握 Python 行追踪机制。请先通过终端运行演示。

## 源码来源与边界

- 作者 / 仓库：[Sebastian Raschka / LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- 固定 commit：`cdbd33e6d57a71e20ea31434841095fac50cecd8`
- [原文件](https://github.com/rasbt/LLMs-from-scratch/blob/cdbd33e6d57a71e20ea31434841095fac50cecd8/ch03/01_main-chapter-code/multihead-attention.ipynb)：`ch03/01_main-chapter-code/multihead-attention.ipynb`
- 标题：`Multi-head Attention from Chapter 3` → `Variant B: Alternative implementation`
- 搜索：`class MultiHeadAttention`，重点看 `__init__` 和 `forward`。
- 单元：从 0 开始计数的 cell 13，id 为 `2773c09d-c136-4372-a2be-04b58d292842`；单元内代码第 1–54 行。这个序号包含 Markdown，不是运行序号 `In [n]`。
- 配套说明：[同版本 ch03.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/cdbd33e6d57a71e20ea31434841095fac50cecd8/ch03/01_main-chapter-code/ch03.ipynb) 的 `3.5.1 Applying a causal attention mask` 和 `3.6.2 Implementing multi-head attention with weight splits`。

`upstream/multihead-attention.ipynb` 是完整未改动的原文件快照，Git blob SHA 为 `b58ee71132ecf04a85512c2093322f7d0eabf0d4`；许可见 `upstream/LICENSE.txt`（Apache-2.0）。快照中 Data Loader 等内容不在本次运行范围，无需安装 tiktoken 或下载数据。

| 文件 / 内容 | 性质与改动 |
| --- | --- |
| `mha.py` 的 `MultiHeadAttention` | 从上述单元提取；仅增加模块级导入、来源和中文注释，类的执行语句未改动；已用 Python AST 核对一致 |
| `demo_attention.py` | 新增的输入、参数初值、观察、打印、断言和更新检查；不是原书例子 |
| `outputs/run_results.json` | 本机运行产生的输入、初值、中间数值、Shape、布局与检查结果；由已有 `.gitignore` 忽略 |

## 运行

从项目根目录运行，使用已安装 PyTorch 的解释器。依赖与根目录一致：Python 3.13.16 / PyTorch 2.9.1 已验证。

```zsh
.venv/bin/python 02_attention/demo_attention.py
```

如果使用其他已有虚拟环境，将 `.venv/bin/python` 换成那个环境的 Python 路径。无需另建环境或下载模型。

默认固定为 CPU、float32、单线程、seed=123、Dropout=0；数据和参数都是人工案例，便于复算，不经过随机训练生成。

```zsh
# 只改变诊断更新的学习率
.venv/bin/python 02_attention/demo_attention.py --lr 0.0001

# 也可显式指定随机种子和结果目录
.venv/bin/python 02_attention/demo_attention.py --seed 123 --output-dir 02_attention/outputs
```

设定随机种子便于复现，但本案例会覆盖全部参数为固定初值，因此单独改变 seed 不会改变初始输出。

## 同一个数值案例

输入是已经变成向量的表示，不是 Token ID。B=2、T=3、d_in=d_out=4、H=2、head_dim=2。

第一条序列：

```text
[[1,0,1,0],
 [0,1,1,0],
 [1,1,0,1]]
```

完整 Q/K/V 参数在 `set_example_parameters` 中明确指定。对于一个位置的 `[a,b,c,d]`：

```text
Q = [a, b,   c,   d]
K = [a, 2*b, c,   2*d]
V = [a, 2*b, 3*c, 4*d]
```

这是为方便手算特意选的对角矩阵；一般的可训练投影矩阵会组合所有输入特征。各头是对投影结果分组，不是限制每个头只能读取一半输入。

Q/K/V 不含 bias；输出投影含 bias。全部共 68 个可训练数值；Mask 是 buffer，不是可训练参数。

| 操作 | Shape | 轴 / 含义 |
| --- | --- | --- |
| 输入、Q/K/V 投影 | `[2,3,4]` | `[batch,位置,特征]` |
| `view` 拆头 | `[2,3,2,2]` | `[batch,位置,head,头内特征]` |
| `transpose(1,2)` | `[2,2,3,2]` | `[batch,head,位置,头内特征]` |
| `Q @ K.transpose(2,3)` | `[2,2,3,3]` | `[batch,head,查询位置,被读取位置]` |
| Mask、缩放、Softmax | `[2,2,3,3]` | 每行在最后一个轴上归一化；本例 Dropout 为 0 |
| 权重汇总 V | `[2,2,3,2]` | `[batch,head,位置,头内特征]` |
| 换轴回位置优先 | `[2,3,2,2]` | `[batch,位置,head,头内特征]` |
| `contiguous().view(...)` | `[2,3,4]` | 拼接同一位置的两个头 |
| 输出投影 | `[2,3,4]` | 组合同一位置的各头特征 |

例如 `queries[0,0]`（换轴后）固定了第一条序列、第一个头，得到普通的 `[3,2]` 矩阵：`[[1,0],[0,1],[1,1]]`。

合并前第一条序列的位置 2：`[[0.716005,1.719941],[0.981475,2.691367]]`；合并后是 `[0.716005,1.719941,0.981475,2.691367]`。`contiguous` 整理布局，`view` 合并轴，均不进行加权求和。

初始化时 `context_length=3` 是 Mask 的最大长度；输入需满足 `1 <= T <= 3`。演示的预期数值和检查针对这个固定案例；若改动案例、头数或参数，也应更新观察预期。

## 教师实际运行结果

2026-10-23，Mac arm64 CPU、Python 3.13.16、PyTorch 2.9.1、float32、seed=123、threads=1，脚本退出码 0。下面数值来自实际运行。

输出 Shape 为 `[2,3,4]`，第一条序列输出约为：

```text
[[ 2.600000, -0.100000, 3.700000, -0.200000],
 [ 1.795570,  1.508859, 3.297785,  0.604430],
 [ 1.306742,  2.965625, 1.539477,  3.351338]]
```

- 未来位置的注意力权重最大值为 0；行和最大误差约 `5.96e-8`。
- 改变第一条序列最后一个位置后，前两个位置输出最大变化为 0，另一条序列变化为 0；该位置自身输出变化约 16.551287。
- 只对第一条序列前两个输出位置求诊断 Loss，未来输入位置梯度为 0；前面输入位置梯度范数约 5.842726。
- 本例中直接 `view` 不能完成合并；`reshape` 与原 `contiguous().view` 结果一致。省去轴交换后错误合并的最大差约 3.339523。
- Q/K/V 与输出投影参数的梯度均有限且范数非零；`backward()` 后参数未变，`optimizer.step()` 后各参数均有变化。
- AdamW、lr=0.001、weight_decay=0，输出平方均值从 6.029803 变为 5.985525（一次更新）。

这里的 `output.square().mean()` 只是用于检查梯度和更新，它推动输出变小，不是在学习语言，也不能证明泛化能力。后续会将 MHA 放入完整模型，用下一 Token 预测的 Loss 联合训练。

## 自己动一次，再解释结果

1. 对着输出指出四维 Tensor 每个轴的含义，并找到 `queries[0,0]` 的三行来自哪里。
2. 看脚本改变未来输入的对照，解释为什么前面位置不变。
3. 比较正确合并与省略轴交换的错误合并，解释 Shape 相同为什么数值错位。
4. 将 `--lr` 改为 `0.0001`，比较同一初值下的一次参数变化量；解释为什么只执行 backward 不会更新参数。

这些文件和教师运行只说明代码可用；自己的运行、解释和理解检查仍需分别完成。
