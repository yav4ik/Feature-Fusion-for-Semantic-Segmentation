from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from src.data import build_test_loader
from src.model import MyModel
from src.utils import ensure_dir, load_checkpoint, mask_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the trained model.")
    parser.add_argument("--config", default="config/inference.yaml", help="Path to the inference config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load. Overrides config value.")
    parser.add_argument("--colorize", action="store_true", help="Save colorized masks instead of raw label PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})

    device = torch.device(infer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_value = args.checkpoint or infer_cfg.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("A checkpoint path is required (--checkpoint or inference.checkpoint in config).")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_loader = build_test_loader(data_cfg, device=device)
    model = MyModel(
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg.get("embed_dim", 96)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    output_dir = ensure_dir(Path(infer_cfg.get("output_dir", "predictions")))
    dataset = test_loader.dataset

    with torch.no_grad():
        for idx, images in enumerate(tqdm(test_loader, desc="Predict")):
            images = images.to(device)
            logits = model(images)
            mask = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            mask_np = mask.cpu().numpy()[0].astype(np.uint8)
            name = Path(dataset.image_names[idx]).stem

            if args.colorize:
                image = mask_to_image(mask_np)
            else:
                image = Image.fromarray(mask_np)

            image.save(output_dir / f"{name}.png")


if __name__ == "__main__":
    main()
