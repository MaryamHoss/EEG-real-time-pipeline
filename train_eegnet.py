"""Train EEGNet on cached MI tensors (short run)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eegnet import EEGNet


def main() -> None:
    p = argparse.ArgumentParser(description="Train EEGNet on data_cache MI tensors.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data_cache/mi_subject1_leftright_runs4812.pt"),
    )
    p.add_argument("--out", type=Path, default=Path("data_cache/eegnet_mi.pt"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    bundle = torch.load(args.data, map_location="cpu", weights_only=False) #load the .pt file we preprocessed
    X = bundle["X"].float()
    y = bundle["y"].long()
    n_total, n_ch, n_times = X.shape
    if n_total < 8: #if there are less than 8 trials, set the validation fraction to 0.2
        args.val_frac = min(args.val_frac, 0.2)

    n_val = max(1, int(round(n_total * args.val_frac))) #calculate the number of validation trials
    perm = torch.randperm(n_total)# Randomly shuffle the indices of the trials
    val_idx = perm[:n_val] #select the first n_val indices for validation
    train_idx = perm[n_val:] #select the remaining indices for training
    if train_idx.numel() == 0:
        train_idx = val_idx #if there are no training trials, use the validation trials for training

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EEGNet(n_ch, n_times, n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_correct = 0
        val_n = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_n += yb.numel()
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_channels": n_ch,
            "n_times": n_times,
            "n_classes": 2,
            "class_names": bundle.get("class_names", {0: "left_hand", 1: "right_hand"}),
        },
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
