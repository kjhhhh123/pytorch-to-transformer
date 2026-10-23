"""沿仓库原版 MHA 观察数据流，再验证遮挡、梯度和一次参数更新。"""

import argparse
import inspect
import json
import math
import platform
import sys
from pathlib import Path

import torch

from mha import MultiHeadAttention


COMMIT = "cdbd33e6d57a71e20ea31434841095fac50cecd8"


def make_example():
    # 人工构造的已嵌入向量，不是原书样本，也不是 Token ID。
    return torch.tensor([
        [[1., 0., 1., 0.], [0., 1., 1., 0.], [1., 1., 0., 1.]],
        [[0., 1., 0., 1.], [1., 0., 0., 1.], [1., 1., 1., 0.]],
    ], dtype=torch.float32, device="cpu")


def set_example_parameters(model):
    # 指定便于手算的初值；no_grad 仅跳过复制操作，不冻结参数。
    with torch.no_grad():
        model.W_query.weight.copy_(torch.eye(4))
        model.W_key.weight.copy_(torch.diag(torch.tensor([1., 2., 1., 2.])))
        model.W_value.weight.copy_(torch.diag(torch.tensor([1., 2., 3., 4.])))
        model.out_proj.weight.copy_(torch.tensor([
            [1., 0., .5, 0.], [0., 1., 0., .5],
            [.5, 0., 1., 0.], [0., .5, 0., 1.],
        ]))
        model.out_proj.bias.copy_(torch.tensor([.1, -.1, .2, -.2]))


def observe_forward(model, x):
    """辅助观察工具：原 forward 正常执行，仅复制指定位置的局部变量。"""
    # 行事件在语句执行前触发，因此读取的是前一步已经算出的结果。
    markers = {
        "keys = keys.view(": ("projected", ("queries", "keys", "values")),
        "keys = keys.transpose(": ("split", ("queries", "keys", "values")),
        "attn_scores = queries @": ("heads", ("queries", "keys", "values")),
        "mask_bool =": ("scores", ("attn_scores",)),
        "attn_weights = torch.softmax(": ("masked", ("attn_scores", "mask_bool")),
        "attn_weights = self.dropout(": ("weights", ("attn_weights",)),
        "context_vec = context_vec.contiguous(": ("context_transposed", ("context_vec",)),
        "context_vec = self.out_proj(": ("merged", ("context_vec",)),
    }
    lines, first_line = inspect.getsourcelines(model.forward)
    locations = {}
    for offset, line in enumerate(lines):
        for marker, stage in markers.items():
            if line.strip().startswith(marker):
                locations[first_line + offset] = stage
    if len(locations) != len(markers):
        raise RuntimeError("未找到全部观察位置；请检查 mha.py 是否改动，更新观察标记。")

    snapshots, metadata = {}, {}
    forward_code = model.forward.__func__.__code__

    def observer(frame, event, arg):
        if frame.f_code is not forward_code or frame.f_locals.get("self") is not model:
            return None
        if event == "line" and frame.f_lineno in locations:
            stage, names = locations[frame.f_lineno]
            snapshots[stage], metadata[stage] = {}, {}
            for name in names:
                tensor = frame.f_locals[name]
                snapshots[stage][name] = tensor.detach().clone()
                metadata[stage][name] = {
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "contiguous": tensor.is_contiguous(),
                }
        return observer

    previous_trace = sys.gettrace()
    try:
        sys.settrace(observer)
        output = model(x)
    finally:
        sys.settrace(previous_trace)
    if len(snapshots) != len(markers):
        raise RuntimeError("观察未完成，未生成完整数据流。")
    return output, snapshots, metadata


def show_data_flow(model, x, output, snapshots, metadata):
    print("\n1. 人工输入：轴为 [batch, 位置, 特征]，Shape =", list(x.shape))
    print(x)
    print("\n2. 全部投影参数：Linear 使用 x @ weight.T + bias")
    for name, parameter in model.named_parameters():
        print(f"{name}，Shape={list(parameter.shape)}，requires_grad={parameter.requires_grad}")
        print(parameter.detach())

    print("\n3. Q/K/V 投影：[B,T,D]；以下展示第一条序列")
    for name, tensor in snapshots["projected"].items():
        print(f"{name}，完整 Shape={list(tensor.shape)}")
        print(tensor[0])

    print("\n4. view 拆头：[B,T,H,d]；第一条序列按位置分组")
    print("Shape =", metadata["split"]["queries"]["shape"])
    print(snapshots["split"]["queries"][0])
    print("\n5. transpose 换轴：[B,H,T,d]；选定 batch/head 后是 [T,d]")
    heads = snapshots["heads"]
    print("Shape =", metadata["heads"]["queries"]["shape"])
    for head_index in range(model.num_heads):
        print(f"queries[0,{head_index}] =")
        print(heads["queries"][0, head_index])

    print("\n6. Q @ K.transpose(2,3)：[B,H,T,T]，行是读取方，列是被读取位置")
    for head_index in range(model.num_heads):
        print(f"头 {head_index} 的原始分数：")
        print(snapshots["scores"]["attn_scores"][0, head_index])
    print("\n7. 因果 Mask：[T,T] 广播到 [B,H,T,T]；True 位置填 -inf")
    print(snapshots["masked"]["mask_bool"])
    print("第一条序列各头的遮挡后分数：")
    print(snapshots["masked"]["attn_scores"][0])
    print(f"\n8. 除以 sqrt(head_dim)={math.sqrt(model.head_dim):.6f}，再 Softmax(dim=-1)")
    print("Dropout p=0，以下是第一条序列各头的权重：")
    print(snapshots["weights"]["attn_weights"][0])

    transposed = snapshots["context_transposed"]["context_vec"]
    print("\n9. 权重 @ V：[B,H,T,d]；第一条序列各头汇总结果：")
    print(transposed.transpose(1, 2)[0])
    print("\n10. 换轴回 [B,T,H,d]，把同一位置的头放到一起：")
    print("Shape =", list(transposed.shape))
    print("位置 2 的两个头：")
    print(transposed[0, 2])
    print("布局信息：", metadata["context_transposed"]["context_vec"])
    print("\n11. contiguous().view(B,T,D)：合并两个头，第一条序列：")
    print(snapshots["merged"]["context_vec"][0])
    print("\n12. 输出投影后，Shape =", list(output.shape))
    print(output.detach()[0])


def check_forward(model, x, output, snapshots):
    weights = snapshots["weights"]["attn_weights"]
    mask = snapshots["masked"]["mask_bool"]
    merged = snapshots["merged"]["context_vec"]
    transposed = snapshots["context_transposed"]["context_vec"]
    assert list(output.shape) == [2, 3, 4]
    assert torch.isfinite(output).all()
    future_weight = weights.masked_select(mask[None, None]).abs().max().item()
    assert future_weight == 0.
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 3))
    torch.testing.assert_close(
        output.detach()[0, 2],
        torch.tensor([1.306742, 2.965625, 1.539477, 3.351338]),
        atol=1e-6, rtol=1e-6,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 68
    assert "mask" in dict(model.named_buffers())
    assert "mask" not in dict(model.named_parameters())

    with torch.no_grad():
        changed = x.clone()
        changed[0, 2] += torch.tensor([3., -2., 5., -4.])
        changed_output = model(changed)
        prefix_difference = (output[0, :2] - changed_output[0, :2]).abs().max().item()
        last_difference = (output[0, 2] - changed_output[0, 2]).abs().max().item()
        other_batch_difference = (output[1] - changed_output[1]).abs().max().item()
        assert prefix_difference == 0. and other_batch_difference == 0.
        assert last_difference > 0.
        separate = torch.cat([model(sequence[None]) for sequence in x])
        torch.testing.assert_close(separate, output.detach())

    # 只让第一条序列前两个位置贡献 Loss，未来位置不应收到输入梯度。
    prefix_input = x.clone().requires_grad_()
    prefix_loss = model(prefix_input)[0, :2].square().mean()
    prefix_gradient, = torch.autograd.grad(prefix_loss, prefix_input)
    assert prefix_gradient[0, 2].abs().max().item() == 0.
    assert prefix_gradient[0, :2].norm().item() > 0.
    assert prefix_gradient[1].abs().max().item() == 0.

    try:
        transposed.view(2, 3, 4)
    except RuntimeError:
        direct_view_failed = True
    else:
        direct_view_failed = False
    assert direct_view_failed
    torch.testing.assert_close(transposed.reshape(2, 3, 4), merged)
    wrong_merge = transposed.transpose(1, 2).reshape(2, 3, 4)
    wrong_merge_difference = (wrong_merge - merged).abs().max().item()
    assert wrong_merge_difference > 0.

    checks = {
        "future_weight_max": future_weight,
        "weight_row_sum_max_error": (weights.sum(-1) - 1).abs().max().item(),
        "earlier_output_max_difference": prefix_difference,
        "changed_position_output_max_difference": last_difference,
        "other_batch_output_max_difference": other_batch_difference,
        "future_input_gradient_max": prefix_gradient[0, 2].abs().max().item(),
        "past_input_gradient_norm": prefix_gradient[0, :2].norm().item(),
        "direct_view_failed_for_this_layout": direct_view_failed,
        "wrong_merge_max_difference": wrong_merge_difference,
        "parameter_count": 68,
    }
    print("\n13. 遮挡、batch 独立性和布局检查通过：")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("正确合并的第一个位置：", merged[0, 0])
    print("省去 transpose 的错误合并：", wrong_merge[0, 0])
    return checks


def check_parameter_update(model, x, learning_rate):
    # 这是诊断目标：检查更新链路；它不是下一 Token 预测训练。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()

    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradients[name] = parameter.grad.norm().item()
        assert gradients[name] > 0.
        # backward 只算梯度，这时参数还没有更新。
        torch.testing.assert_close(parameter.detach(), before[name], rtol=0., atol=0.)

    optimizer.step()
    changes = {
        name: (parameter.detach() - before[name]).abs().max().item()
        for name, parameter in model.named_parameters()
    }
    assert all(change > 0. for change in changes.values())
    with torch.no_grad():
        loss_after = model(x).square().mean().item()
    assert math.isfinite(loss_after)

    result = {
        "objective": "输出平方均值；仅验证求导和更新，不是语言模型训练",
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": 0.,
        "loss_before": loss.item(),
        "loss_after_one_step": loss_after,
        "parameter_gradient_norms": gradients,
        "parameter_max_changes": changes,
        "parameters_unchanged_after_backward": True,
    }
    print("\n14. 一次参数更新检查通过：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0 or args.threads < 1:
        parser.error("lr 必须是有限正数，threads 必须至少为 1。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_printoptions(precision=6, sci_mode=False)
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "seed": args.seed, "threads": args.threads,
    }
    print("运行环境：", json.dumps(environment, ensure_ascii=False))
    x = make_example()
    model = MultiHeadAttention(4, 4, 3, 0., num_heads=2, qkv_bias=False).to(
        device="cpu", dtype=torch.float32
    )
    set_example_parameters(model)
    initial_parameters = {name: parameter.detach().tolist() for name, parameter in model.named_parameters()}
    output, snapshots, metadata = observe_forward(model, x)
    show_data_flow(model, x, output, snapshots, metadata)
    checks = check_forward(model, x, output, snapshots)
    update = check_parameter_update(model, x, args.lr)

    report = {
        "environment": environment,
        "source_commit": COMMIT,
        "case": "新增人工向量与人工参数初值，不是原书样本",
        "config": {"d_in": 4, "d_out": 4, "context_length": 3, "num_heads": 2, "dropout": 0., "qkv_bias": False},
        "input": x.tolist(), "initial_parameters": initial_parameters,
        "forward_before_update": {
            stage: {name: tensor.tolist() for name, tensor in values.items()}
            for stage, values in snapshots.items()
        },
        "layout_metadata": metadata,
        "output_before_update": output.detach().tolist(),
        "checks": checks, "diagnostic_update": update,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "run_results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n全部检查通过。详细结果：", result_path.resolve())


if __name__ == "__main__":
    main()
