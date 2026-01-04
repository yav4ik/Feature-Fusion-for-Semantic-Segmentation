from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision.models import swin_transformer
from torchvision.ops.misc import Permute


class Mlp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None, drop: float = 0.05):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class EncoderSwinTransformer(swin_transformer.SwinTransformer):
    """Wrap torchvision SwinV2-Tiny to expose intermediate feature maps."""

    def __init__(self, pretrained: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if pretrained:
            weights = swin_transformer.Swin_V2_T_Weights.IMAGENET1K_V1
            state_dict = weights.get_state_dict(progress=True, check_hash=True)
            self.load_state_dict(state_dict, strict=False)

        # replace classification components with identities
        self.head = nn.Identity()
        self.avgpool = nn.Identity()
        self.norm = nn.Identity()
        self.permute = nn.Identity()
        self.flatten = nn.Identity()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx % 2 == 1:
                outputs.append(x)
        return outputs[::-1]


class Decoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.permute = Permute([0, 3, 1, 2])
        self.permute_back = Permute([0, 2, 3, 1])

        self.conv_relu1 = ConvReLU(8 * embed_dim, embed_dim)
        self.up_sample1 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_relu2 = ConvReLU(4 * embed_dim, embed_dim)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.linear_fuse1 = Mlp(2 * embed_dim, embed_dim)

        self.conv_relu3 = ConvReLU(2 * embed_dim, embed_dim)
        self.up_sample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.linear_fuse2 = Mlp(2 * embed_dim, embed_dim)

        self.conv_relu4 = ConvReLU(embed_dim, embed_dim)
        self.linear_fuse3 = Mlp(2 * embed_dim, embed_dim)

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        x4, x3, x2, x1 = encoder_features

        x1 = self.up_sample1(self.conv_relu1(self.permute(x1)))
        x2 = self.up_sample2(self.conv_relu2(self.permute(x2)))

        x = torch.cat([x1, x2], dim=1)
        x = self.linear_fuse1(self.permute_back(x))

        x3 = self.up_sample3(self.conv_relu3(self.permute(x3)))
        x = self.permute(x)
        x = torch.cat([x, x3], dim=1)
        x = self.linear_fuse2(self.permute_back(x))

        x4 = self.conv_relu4(self.permute(x4))
        x = self.permute(x)
        x = torch.cat([x, x4], dim=1)
        x = self.linear_fuse3(self.permute_back(x))

        return x


class MyModel(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, dropout: float = 0.1, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = EncoderSwinTransformer(
            pretrained=pretrained,
            patch_size=[4, 4],
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 8],
            stochastic_depth_prob=0.1,
            block=swin_transformer.SwinTransformerBlockV2,
            downsample_layer=swin_transformer.PatchMergingV2,
        )
        self.decoder = Decoder(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = Mlp(embed_dim, num_classes)
        self.permute_back = Permute([0, 3, 1, 2])
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.dropout(x)
        logits = self.mlp(x)
        logits = self.permute_back(logits)
        logits = self.upsample(logits)
        return logits

