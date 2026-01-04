from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def create_folder_names(image_names: np.ndarray) -> np.ndarray:
    """Derive folder names by trimming everything after the last underscore."""
    folder_names: list[str] = []
    for name in np.atleast_1d(image_names):
        file_name = Path(str(name)).name
        underscore_idx = file_name.rfind("_")
        folder_names.append(file_name[:underscore_idx] if underscore_idx != -1 else file_name)
    return np.asarray(folder_names)


def build_joint_transform(image_size: int, augment: bool = False) -> v2.Compose:
    ops: list[object] = [v2.Resize(size=(image_size, image_size))]
    if augment:
        ops.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )
    return v2.Compose(ops)


def build_image_only_augmentations() -> v2.Compose:
    return v2.Compose(
        [
            v2.ColorJitter(brightness=[0.625, 1.6], contrast=[0.8, 1.25]),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            v2.RandomPosterize(bits=4, p=0.2),
            v2.RandomAutocontrast(0.2),
        ]
    )


def _resolve_split_path(root: Path, split_path: Path) -> Path:
    return split_path if split_path.is_absolute() else root / split_path


class SatelliteDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        split_file: Path | str,
        image_size: int = 512,
        augment: bool = False,
        has_labels: bool = True,
        image_subdir: str = "images",
        label_subdir: str = "labels",
    ) -> None:
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir
        image_names = np.loadtxt(self.split_file, dtype=str)
        image_names = np.atleast_1d(image_names)

        self.image_names: list[str] = [str(name) for name in image_names.tolist()]
        self.folder_names: list[str] = create_folder_names(image_names).tolist()

        self.has_labels = has_labels
        self.joint_transform = build_joint_transform(image_size=image_size, augment=augment)
        self.image_only_transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )
        self.image_augment = build_image_only_augmentations() if augment else None

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_image(self, idx: int) -> Tuple[Image.Image, Optional[Image.Image]]:
        image_name = self.image_names[idx]
        folder = self.folder_names[idx]
        image_path = self.root / folder / self.image_subdir / image_name
        image = Image.open(image_path).convert("RGB")
        mask: Optional[Image.Image] = None

        if self.has_labels:
            mask_path = self.root / folder / self.label_subdir / image_name
            mask = Image.open(mask_path)
        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._load_image(idx)

        if self.joint_transform is not None:
            if mask is not None:
                image, mask = self.joint_transform(image, mask)
            else:
                image = self.joint_transform(image)

        if self.image_augment is not None:
            image = self.image_augment(image)

        image = self.image_only_transform(image)
        if mask is None:
            return image

        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask_tensor


def build_dataloaders(
    data_cfg: Mapping[str, object],
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    root = Path(str(data_cfg.get("root", ".")))
    splits = data_cfg.get("splits", {})
    train_split = Path(str(splits.get("train", "")))
    val_split = Path(str(splits.get("val", "")))

    image_size = int(data_cfg.get("image_size", 512))
    train_batch_size = int(data_cfg.get("train_batch_size", 2))
    val_batch_size = int(data_cfg.get("val_batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    image_subdir = str(data_cfg.get("image_subdir", "images"))
    label_subdir = str(data_cfg.get("label_subdir", "labels"))

    pin_memory = bool(device is not None and device.type == "cuda")

    train_dataset = SatelliteDataset(
        root=root,
        split_file=_resolve_split_path(root, train_split),
        image_size=image_size,
        augment=True,
        has_labels=True,
        image_subdir=image_subdir,
        label_subdir=label_subdir,
    )
    val_dataset = SatelliteDataset(
        root=root,
        split_file=_resolve_split_path(root, val_split),
        image_size=image_size,
        augment=False,
        has_labels=True,
        image_subdir=image_subdir,
        label_subdir=label_subdir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_test_loader(
    data_cfg: Mapping[str, object],
    device: Optional[torch.device] = None,
) -> DataLoader:
    root = Path(str(data_cfg.get("root", ".")))
    splits = data_cfg.get("splits", {})
    test_split = Path(str(splits.get("test", splits.get("val", ""))))
    image_size = int(data_cfg.get("image_size", 512))
    num_workers = int(data_cfg.get("num_workers", 0))
    image_subdir = str(data_cfg.get("image_subdir", "images"))

    pin_memory = bool(device is not None and device.type == "cuda")

    test_dataset = SatelliteDataset(
        root=root,
        split_file=_resolve_split_path(root, test_split),
        image_size=image_size,
        augment=False,
        has_labels=False,
        image_subdir=image_subdir,
        label_subdir=str(data_cfg.get("label_subdir", "labels")),
    )
    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
