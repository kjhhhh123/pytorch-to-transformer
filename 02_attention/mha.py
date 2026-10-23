"""原仓库 MultiHeadAttention：仅增加导入、来源与中文注释，类的执行语句未改动。

作者：Sebastian Raschka / rasbt/LLMs-from-scratch
许可：Apache-2.0，见 upstream/LICENSE.txt
commit: cdbd33e6d57a71e20ea31434841095fac50cecd8
文件：ch03/01_main-chapter-code/multihead-attention.ipynb
章节：Variant B: Alternative implementation
cell: 2773c09d-c136-4372-a2be-04b58d292842（0-based index 13）

输入必须为 [B,T,d_in]，且 1 <= T <= context_length。
本次采用 qkv_bias=False、dropout=0，固定长度输入，无 Padding Mask。
完整原 Notebook 保存在 upstream/；不需要执行其中的数据加载代码。
"""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 总输出维度均分给各头

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 同一位置的各头特征通过输出投影组合；默认有 bias
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # [B,T,D]：每个位置使用同一套投影参数
        queries = self.W_query(x)
        values = self.W_value(x)

        # 投影结果按头分组；每个头的投影仍可读取完整输入
        # [B,T,D] -> [B,T,H,d]，D=H*d；此步只改变形状
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # [B,T,H,d] -> [B,H,T,d]，让每个头独立处理各位置
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 各 batch、各 head 分别在最后两个轴上做矩阵乘法
        attn_scores = queries @ keys.transpose(2, 3)  # [B,H,T,d] @ [B,H,d,T] -> [B,H,T,T]
        
        # 截取当前长度；True 表示禁止读取的未来位置，广播到所有 batch/head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 先遮挡再缩放；保留对角线，当前位置可以读取自己
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 权重汇总 V 后换轴：[B,H,T,d] -> [B,T,H,d]
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # 先整理内存布局，再将同一位置的 H 个 d 维向量合并成 D 维
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # 保留原实现的输出投影；Shape 仍为 [B,T,D]

        return context_vec
