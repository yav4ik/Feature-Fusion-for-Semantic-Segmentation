from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
import pytorch_warmup as warmup

from src.data import build_dataloaders
from src.model import MyModel
from src.train import fit
from src.utils import ensure_dir, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stepwise Feature Fusion model.")
    parser.add_argument("--config", default="config/training.yaml", help="Path to the training config YAML.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader = build_dataloaders(data_cfg, device=device)

    model = MyModel(
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg.get("embed_dim", 96)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("max_lr", 6e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    total_iters = max(len(train_loader) * int(train_cfg.get("epochs", 1)), 1)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=total_iters,
    )

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps) if warmup_steps > 0 else None

    start_epoch = 0
    if args.resume:
        state = load_checkpoint(Path(args.resume), model, optimizer, scheduler, warmup_scheduler)
        start_epoch = int(state.get("epoch", 0))

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_scheduler=warmup_scheduler,
        device=device,
        epochs=int(train_cfg.get("epochs", 1)),
        start_epoch=start_epoch,
        num_classes=int(model_cfg["num_classes"]),
        ignore_index=int(train_cfg.get("ignore_index", 0)),
        checkpoint_dir=Path(train_cfg.get("checkpoint_dir", "checkpoints")),
        checkpoint_every=int(train_cfg.get("checkpoint_every", 5)),
        warmup_steps=warmup_steps,
    )

    output_dir = ensure_dir(Path(train_cfg.get("output_dir", "runs")))
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

