"""Train EEGNet on cached MI tensors (short run)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eegnet import EEGNet


def stratified_train_val_split(
    y: torch.Tensor,
    val_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split indices so each class is represented in val (when enough trials exist)."""
    g = torch.Generator()
    g.manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    for c in (0, 1):
        idx = (y == c).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        perm = idx[torch.randperm(n, generator=g)]
        n_val = max(1, int(round(float(n) * val_frac)))
        if n > 1 and n_val >= n:
            n_val = n - 1
        if n == 1:
            n_val = 0
        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])
    if not train_parts or not val_parts:
        perm = torch.randperm(len(y), generator=g)
        n_val = max(1, int(round(len(y) * val_frac)))
        return perm[n_val:], perm[:n_val]
    return torch.cat(train_parts), torch.cat(val_parts)


def class_balanced_weights(y: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
    w = len(y) / (n_classes * counts)
    return w / w.mean()


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, n_classes: int = 2) -> list[float]:
    pred = logits.argmax(dim=-1)
    out: list[float] = []
    for c in range(n_classes):
        m = labels == c
        if not m.any():
            out.append(float("nan"))
        else:
            out.append((pred[m] == labels[m]).float().mean().item())
    return out


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
    p.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use a single random split instead of stratified train/val (not recommended for small N).",
    )
    p.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Class-weighted cross-entropy (helps if one class is harder / underrepresented in a fold).",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    bundle = torch.load(args.data, map_location="cpu", weights_only=False) #load the .pt file we preprocessed
    X = bundle["X"].float()
    y = bundle["y"].long()
    n_total, n_ch, n_times = X.shape
    if n_total < 8: #if there are less than 8 trials, set the validation fraction to 0.2
        args.val_frac = min(args.val_frac, 0.2)

    if args.no_stratify:
        n_val = max(1, int(round(n_total * args.val_frac)))
        perm = torch.randperm(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if train_idx.numel() == 0:
            train_idx = val_idx
    else:
        train_idx, val_idx = stratified_train_val_split(y, args.val_frac, args.seed)
        if train_idx.numel() == 0:
            train_idx, val_idx = val_idx, train_idx

    n_train_l = int((y[train_idx] == 0).sum())
    n_train_r = int((y[train_idx] == 1).sum())
    n_val_l = int((y[val_idx] == 0).sum())
    n_val_r = int((y[val_idx] == 1).sum())
    print(
        f"split: train L/R = {n_train_l}/{n_train_r}, val L/R = {n_val_l}/{n_val_r} "
        f"({'random' if args.no_stratify else 'stratified'})"
    )

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EEGNet(n_ch, n_times, n_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.balanced_loss:
        cw = class_balanced_weights(y[train_idx], n_classes=2).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        print(f"balanced loss weights (train): left={cw[0]:.3f} right={cw[1]:.3f}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

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
        val_logits_chunks: list[torch.Tensor] = []
        val_y_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_n += yb.numel()
                val_logits_chunks.append(logits.cpu())
                val_y_chunks.append(yb.cpu())
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        v_logits = torch.cat(val_logits_chunks, dim=0)
        v_y = torch.cat(val_y_chunks, dim=0)
        acc_l, acc_r = per_class_accuracy(v_logits, v_y, n_classes=2)
        acc_l_s = f"{acc_l:.2f}" if acc_l == acc_l else "n/a"
        acc_r_s = f"{acc_r:.2f}" if acc_r == acc_r else "n/a"
        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = "  (best val_acc)"
        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"val_acc_left={acc_l_s}  val_acc_right={acc_r_s}{tag}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs
        best_val_acc = float("nan")
    torch.save(
        {
            "model_state": best_state,
            "n_channels": n_ch,
            "n_times": n_times,
            "n_classes": 2,
            "class_names": bundle.get("class_names", {0: "left_hand", 1: "right_hand"}),
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        },
        args.out,
    )
    print(f"saved {args.out}  (checkpoint from epoch {best_epoch} with val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
