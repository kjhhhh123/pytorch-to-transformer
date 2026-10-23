"""MLP 三分类训练：数据 -> batch -> logits -> Loss -> 梯度 -> 参数更新。"""

import argparse
import csv
import json
import platform
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_datasets(samples_per_class, seed):
    # generator 是独立随机数发生器；seed 固定本次造数据和划分的随机序列。
    generator = torch.Generator().manual_seed(seed)
    # centers：[3, 2]，第 0、1、2 类各自围绕的二维中心坐标。
    centers = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    # labels：[N]，每类编号连续重复 samples_per_class 次；N = 3 × 每类样本数。
    labels = torch.arange(3, dtype=torch.long).repeat_interleave(samples_per_class)
    # features：[N, 2]，按标签选中心，再加标准差为 0.6 的随机扰动作为输入坐标。
    features = centers[labels] + 0.6 * torch.randn(len(labels), 2, generator=generator)

    # 先固定样本划分，再由训练 DataLoader 每轮打乱；验证样本不参与更新。
    indices = torch.randperm(len(labels), generator=generator)  # 打乱后的样本下标。
    split = int(0.8 * len(labels))  # 前 80% 为训练集，剩余为验证集。
    train_indices, val_indices = indices[:split], indices[split:]
    # TensorDataset 用同一组下标取坐标与标签，保证每条输入和答案配对。
    train_data = TensorDataset(features[train_indices], labels[train_indices])
    val_data = TensorDataset(features[val_indices], labels[val_indices])
    return train_data, val_data


class MLP(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()  # 初始化 Module，随后赋给 self 的子层会被注册。
        # hidden_dim 是中间特征数；两个 Linear 层的权重和偏置参与训练。
        self.hidden = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()  # 负数置零，本层没有可训练参数。
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, features):
        # [B, 2] -> [B, hidden_dim] -> [B, 3]；返回 logits，不提前 Softmax。
        hidden = self.activation(self.hidden(features))
        return self.output(hidden)


def train_one_epoch(model, loader, loss_fn, optimizer, device, trace=False):
    model.train()  # 切换层的训练模式；调用本身不执行训练。
    # 分别累计样本损失总和、预测正确数、已处理样本数。
    loss_sum, correct, total = 0.0, 0, 0
    for batch_index, (features, labels) in enumerate(loader):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)  # 清旧梯度，保留模型参数。
        logits = model(features)  # [B, 3]：每条样本的三个类别分数。
        loss = loss_fn(logits, labels)  # 标量：本批样本交叉熵的平均值。

        # trace 只在首轮启用，before 备份首批更新前的权重，供观察参数变化。
        before = None
        if trace and batch_index == 0:
            before = model.hidden.weight.detach().clone()
            print(f"batch: x={tuple(features.shape)} {features.dtype}, "
                  f"y={tuple(labels.shape)} {labels.dtype}")
            print(f"logits={tuple(logits.shape)}, loss={tuple(loss.shape)}")

        loss.backward()  # 计算并累积到参数的 .grad，不更新权重。
        if before is not None:
            print(f"first_weight_grad_norm={model.hidden.weight.grad.norm().item():.6f}")
        optimizer.step()  # Adam 根据梯度和历史状态更新权重、偏置。
        if before is not None:
            change = (model.hidden.weight.detach() - before).abs().mean().item()
            print(f"first_step_weight_change={change:.6f}")

        # Loss 是本 batch 的均值，乘实际 batch 大小后再累计。
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        # argmax 沿类别维取最大分数的位置，再统计本批预测正确的条数。
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += batch_size
    return {"loss": loss_sum / total, "accuracy": correct / total}


def evaluate(model, loader, loss_fn, device):
    model.eval()  # 切换验证模式；本函数不反向传播、不更新参数。
    loss_sum, correct, total = 0.0, 0, 0
    # eval 切换层的模式；no_grad 关闭本段计算的求导记录。
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += batch_size
    return {"loss": loss_sum / total, "accuracy": correct / total}


def main():
    # parser 读取终端参数；args 中保存本次运行的超参数、设备和输出路径。可以在终端配置参数
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_dim, args.threads) < 1:
        parser.error("epochs, batch-size, hidden-dim and threads must be positive")
    if args.samples_per_class < 2 or args.learning_rate <= 0:
        parser.error("samples-per-class must be >= 2 and learning-rate must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this process; use --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    torch.manual_seed(args.seed)  # 固定全局随机序列，用于模型权重初始化等。
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train_data, val_data = make_datasets(args.samples_per_class, args.seed)
    # loader 每次取一批配对样本；单独的 generator 控制训练集每轮的取样顺序。
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed + 1))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    model = MLP(args.hidden_dim).to(device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss()  # 接收原始 logits 和 long 类型的类别标签。
    # optimizer 管理 model.parameters() 中的权重和偏置，lr 是学习率。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"python={platform.python_version()}, torch={torch.__version__}, "
          f"platform={platform.system()} {platform.machine()}, device={device}")
    print(f"train_samples={len(train_data)}, val_samples={len(val_data)}, "
          f"parameters={sum(p.numel() for p in model.parameters())}")
    initial = evaluate(model, val_loader, loss_fn, device)  # 未训练时的验证指标。
    print(f"initial_val_loss={initial['loss']:.6f}, initial_val_acc={initial['accuracy']:.4f}")

    history = []  # 每个 epoch 记录一行训练/验证指标；模型和优化器跨轮复用。
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                        device, trace=(epoch == 1))
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"]})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:02d} train_loss={train_metrics['loss']:.6f} "
                  f"train_acc={train_metrics['accuracy']:.4f} "
                  f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    # 保存最后一轮的参数；不称为通过验证集挑选的最佳模型。
    torch.save(model.state_dict(), checkpoint_path)
    # restored 是相同结构的新模型；state 是从文件读取的参数字典。
    restored = MLP(args.hidden_dim).to(device=device, dtype=torch.float32)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored.load_state_dict(state)
    restored.eval()
    probe_features, _ = next(iter(val_loader))  # 取一批验证输入，忽略标签。
    with torch.no_grad():
        expected = model(probe_features.to(device))
        actual = restored(probe_features.to(device))
    # expected/actual 是原模型与重载模型对同一批输入的输出。
    reload_matches = torch.allclose(expected, actual)
    print(f"reload_outputs_match={reload_matches}")
    if not reload_matches:
        raise RuntimeError("Reloaded model outputs differ from the saved model")

    # history.csv 保存逐轮指标；下面的 summary 汇总配置及初始、最终结果。
    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"seed": args.seed, "device": args.device, "dtype": "float32",
               "torch_version": str(torch.__version__), "epochs": args.epochs,
               "batch_size": args.batch_size, "hidden_dim": args.hidden_dim,
               "learning_rate": args.learning_rate, "samples_per_class": args.samples_per_class,
               "threads": args.threads, "train_samples": len(train_data),
               "val_samples": len(val_data), "initial_validation": initial,
               "final_epoch": history[-1], "reload_outputs_match": reload_matches}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
