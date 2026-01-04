from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import get_lr, miou, pixel_accuracy
from .utils import ensure_dir, save_checkpoint


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    warmup_scheduler: Optional[Any],
    device: torch.device,
    *,
    epochs: int,
    start_epoch: int = 0,
    num_classes: int,
    ignore_index: int = 0,
    checkpoint_dir: Path = Path("checkpoints"),
    checkpoint_every: int = 5,
    warmup_steps: int = 0,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_acc": [],
        "val_acc": [],
        "lrs": [],
    }

    model.to(device)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        train_miou_total = 0.0
        train_acc_total = 0.0

        for images, masks in tqdm(train_loader, desc=f"Train {epoch + 1}/{start_epoch + epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_steps and scheduler is not None:
                        scheduler.step()
            elif scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            train_miou_total += miou(outputs, masks, num_classes, ignore_index)
            train_acc_total += pixel_accuracy(outputs, masks, ignore_index)
            history["lrs"].append(get_lr(optimizer))

        model.eval()
        val_loss = 0.0
        val_miou_total = 0.0
        val_acc_total = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validate"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_miou_total += miou(outputs, masks, num_classes, ignore_index)
                val_acc_total += pixel_accuracy(outputs, masks, ignore_index)

        train_loss_avg = running_loss / max(len(train_loader), 1)
        val_loss_avg = val_loss / max(len(val_loader), 1)

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["train_miou"].append(train_miou_total / max(len(train_loader), 1))
        history["val_miou"].append(val_miou_total / max(len(val_loader), 1))
        history["train_acc"].append(train_acc_total / max(len(train_loader), 1))
        history["val_acc"].append(val_acc_total / max(len(val_loader), 1))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_checkpoint(
                checkpoint_dir / f"best_epoch_{epoch + 1}.pt",
                model,
                optimizer,
                scheduler,
                warmup_scheduler,
                epoch + 1,
            )

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1}.pt",
                model,
                optimizer,
                scheduler,
                warmup_scheduler,
                epoch + 1,
            )

    return history

