from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

COLOR_MAP = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 255, 36],
        [148, 148, 148],
        [255, 255, 255],
        [34, 97, 98],
        [0, 69, 255],
        [75, 181, 73],
        [226, 31, 7],
    ],
    dtype=np.uint8,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a predicted mask with integer classes to a color image."""
    mask = np.asarray(mask, dtype=np.uint8)
    mask = np.clip(mask, 0, len(COLOR_MAP) - 1)
    colored = COLOR_MAP[mask]
    return Image.fromarray(colored)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    warmup_scheduler: Optional[Any],
    epoch: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "warmup": warmup_scheduler.state_dict() if hasattr(warmup_scheduler, "state_dict") else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    warmup_scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    if optimizer is not None and "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if warmup_scheduler is not None and "warmup" in state and state["warmup"] is not None:
        warmup_scheduler.load_state_dict(state["warmup"])
    return state

