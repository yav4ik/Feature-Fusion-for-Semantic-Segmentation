# Feature Fusion for Semantic Segmentation

Semantic segmentation on land cover map dataset OpenEarthMap with Stepwise Feature Fusion method and Swin Transformer version 2 tiny model.

Weights can be load from [here](https://drive.google.com/file/d/1Sfrsa5wiJr72UF6wGX2JMEpunievuihm/view?usp=sharing)

| Method                  | Backbone | IoU Bareland (%) | IoU Rangeland (%) | IoU Developed space (%) | IoU Road (%) | IoU Tree (%) | IoU Water (%) | IoU Agriculture land (%) | IoU Building (%) | mIoU (%) | Param (M) |
|-------------------------|:--------:|-----------------:|-------------------|-------------------------|--------------|--------------|---------------|--------------------------|------------------|----------|-----------|
| Stepwise Feature Fusion | SwinV2_T |            49.80 | 56.39             | 50.97                   | 59.25        | 69.81        | 82.08         | 74.89                    | 74.54            | 64.72    | 29.33     |

# Visual Comparison

![enter image description here](/images/visual_comparison.png)

# References

OpenEarthMap: [dataset](https://open-earth-map.org/), [paper](https://arxiv.org/abs/2210.10732)

Stepwise Feature Fusion: Local Guides Global: [paper](https://arxiv.org/abs/2203.03635)

Swin Transformer V2: Scaling Up Capacity and Resolution: [paper](https://arxiv.org/abs/2111.09883)

## Project layout
- `src/`: datasets, model, metrics, training loop, and helpers.
- `tools/`: entrypoints for training (`tools/train.py`) and inference (`tools/predict.py`).
- `config/`: YAML configs for training and inference defaults.
- `requirements.txt`: runtime dependencies.
- `sff_swinv2T.ipynb`: original research notebook.

## Quickstart
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Train: `python tools/train.py --config config/training.yaml`
3. Predict (needs a checkpoint): `python tools/predict.py --config config/inference.yaml --checkpoint path/to/checkpoint.pt`


