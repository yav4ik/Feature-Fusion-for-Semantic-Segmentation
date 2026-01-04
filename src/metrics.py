from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pixel_accuracy(output: torch.Tensor, mask: torch.Tensor, ignore_index: int = 0) -> float:
    with torch.no_grad():
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        valid = mask != ignore_index
        correct = (pred == mask) & valid
        total = int(valid.sum().item())
        if total == 0:
            return 0.0
        return float(correct.sum().item()) / float(total)


def miou(output: torch.Tensor, mask: torch.Tensor, num_classes: int, ignore_index: int = 0, smooth: float = 1e-10) -> float:
    with torch.no_grad():
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        pred = pred.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        ious = []
        for cls_idx in range(num_classes):
            if cls_idx == ignore_index:
                continue
            pred_i = pred == cls_idx
            mask_i = mask == cls_idx
            if mask_i.long().sum().item() == 0:
                ious.append(np.nan)
                continue
            intersection = torch.logical_and(pred_i, mask_i).sum().float().item()
            union = torch.logical_or(pred_i, mask_i).sum().float().item()
            ious.append((intersection + smooth) / (union + smooth))
        return float(np.nanmean(ious))


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group.get("lr", 0.0))
    return 0.0

