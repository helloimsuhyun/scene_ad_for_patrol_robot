# backbone_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
)

from cnn_emb import (
    load_model as load_cnn_model,
    extract_grid_layers,
    make_aligned_local_transform,
)


# ============================================================
# 공통 유틸
# ============================================================

def bgr_to_pil_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb).convert("RGB")


class LocalBackbone(Protocol):
    name: str
    img_size: int

    def extract_grid(self, img_bgr, device: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Returns
        -------
        feat : torch.Tensor
            shape [C, H, W], patch/location-wise normalized
        feat_hw : (H, W)
        """
        ...


# ============================================================
# ResNet wrapper
# ============================================================

@dataclass
class ResNetBackbone:
    name: str
    out_layer: str = "layer3"
    img_size: int = 560
    model_name: str = "resnet18"

    _model: Optional[nn.Module] = None
    _transform: Optional[object] = None

    def __post_init__(self):
        self._model, _ = load_cnn_model(
            model_name=self.model_name,
            out_layer=self.out_layer,
            device="cpu",
        )
        self._transform = make_aligned_local_transform(self.img_size)

    @torch.no_grad()
    def extract_grid(self, img_bgr, device: str):
        if self._model is None:
            raise RuntimeError("ResNet model is not initialized.")

        self._model = self._model.to(device).eval()

        img_pil = bgr_to_pil_rgb(img_bgr)
        x = self._transform(img_pil)  # [3,H,W]

        feat, feat_hw = extract_grid_layers(self._model, device, x)
        # extract_grid_layers 내부에서 normalize 하지만 한번 더 안전하게 보장
        feat = F.normalize(feat, dim=0)
        return feat, feat_hw


# ============================================================
# ConvNeXt wrapper
# ============================================================

class ConvNeXtGridExtractor(nn.Module):
    """
    torchvision ConvNeXt의 feature map extractor.
    model.features(x) 출력은 [B, C, H, W]
    """

    def __init__(self, variant: str = "tiny"):
        super().__init__()

        variant = variant.lower()
        if variant == "tiny":
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        elif variant == "small":
            base = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        elif variant == "base":
            base = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        self.features = base.features
        self.variant = variant

    def forward(self, x):
        return self.features(x)  # [B,C,H,W]


def make_imagenet_square_transform(img_size=560):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@dataclass
class ConvNeXtBackbone:
    name: str
    variant: str = "tiny"
    img_size: int = 560

    _model: Optional[nn.Module] = None
    _transform: Optional[object] = None

    def __post_init__(self):
        self._model = ConvNeXtGridExtractor(variant=self.variant)
        self._transform = make_imagenet_square_transform(self.img_size)

    @torch.no_grad()
    def extract_grid(self, img_bgr, device: str):
        if self._model is None:
            raise RuntimeError("ConvNeXt model is not initialized.")

        self._model = self._model.to(device).eval()

        img_pil = bgr_to_pil_rgb(img_bgr)
        x = self._transform(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]

        feat = self._model(x).squeeze(0)  # [C,H,W]
        feat = F.normalize(feat, dim=0)

        _, H, W = feat.shape
        return feat, (H, W)


# ============================================================
# DINOv2 wrapper
# ============================================================

def make_dino_square_transform(img_size=560):
    # DINOv2도 일반적으로 ImageNet 정규화 사용
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dinov2_model(model_name: str, device: str = "cpu"):
    """
    model_name examples:
      - dinov2_vits14
      - dinov2_vitb14
      - dinov2_vitl14
      - dinov2_vitg14
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_dinov2_grid(model, device: str, x: torch.Tensor):
    """
    x: [3,H,W]
    return feat: [C, Hf, Wf]
    """
    x = x.unsqueeze(0).to(device)  # [1,3,H,W]

    # DINOv2 공식 hub 모델은 get_intermediate_layers 지원
    feats = model.get_intermediate_layers(
        x,
        n=1,
        reshape=True,
        return_class_token=False,
    )
    feat = feats[0].squeeze(0)  # [C,Hf,Wf]
    feat = F.normalize(feat, dim=0)

    _, Hf, Wf = feat.shape
    return feat, (Hf, Wf)


@dataclass
class DinoV2Backbone:
    name: str
    model_name: str = "dinov2_vits14"
    img_size: int = 560

    _model: Optional[nn.Module] = None
    _transform: Optional[object] = None

    def __post_init__(self):
        self._model = load_dinov2_model(self.model_name, device="cpu")
        self._transform = make_dino_square_transform(self.img_size)

    @torch.no_grad()
    def extract_grid(self, img_bgr, device: str):
        if self._model is None:
            raise RuntimeError("DINOv2 model is not initialized.")

        self._model = self._model.to(device).eval()

        img_pil = bgr_to_pil_rgb(img_bgr)
        x = self._transform(img_pil)  # [3,H,W]

        feat, feat_hw = extract_dinov2_grid(self._model, device, x)
        feat = F.normalize(feat, dim=0)
        return feat, feat_hw


# ============================================================
# Factory
# ============================================================

def build_local_backbone(backbone_name: str, img_size: int = 560):
    """
    Supported names
    ---------------
    ResNet:
      - resnet18_layer2
      - resnet18_layer3
      - resnet18_layer4

    ConvNeXt:
      - convnext_tiny
      - convnext_small
      - convnext_base

    DINOv2:
      - dinov2_vits14
      - dinov2_vitb14
      - dinov2_vitl14
      - dinov2_vitg14
    """
    name = backbone_name.lower()

    # ---------------- ResNet ----------------
    if name == "resnet18_layer2":
        return ResNetBackbone(
            name=name,
            model_name="resnet18",
            out_layer="layer2",
            img_size=img_size,
        )

    if name == "resnet18_layer3":
        return ResNetBackbone(
            name=name,
            model_name="resnet18",
            out_layer="layer3",
            img_size=img_size,
        )

    if name == "resnet18_layer4":
        return ResNetBackbone(
            name=name,
            model_name="resnet18",
            out_layer="layer4",
            img_size=img_size,
        )

    # ---------------- ConvNeXt ----------------
    if name == "convnext_tiny":
        return ConvNeXtBackbone(
            name=name,
            variant="tiny",
            img_size=img_size,
        )

    if name == "convnext_small":
        return ConvNeXtBackbone(
            name=name,
            variant="small",
            img_size=img_size,
        )

    if name == "convnext_base":
        return ConvNeXtBackbone(
            name=name,
            variant="base",
            img_size=img_size,
        )

    # ---------------- DINOv2 ----------------
    if name in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}:
        return DinoV2Backbone(
            name=name,
            model_name=name,
            img_size=img_size,
        )

    raise ValueError(
        f"Unknown backbone_name={backbone_name}. "
        f"Supported: "
        f"resnet18_layer2, resnet18_layer3, resnet18_layer4, "
        f"convnext_tiny, convnext_small, convnext_base, "
        f"dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14"
    )


# ============================================================
# Simple smoke test
# ============================================================

if __name__ == "__main__":
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    names = [
        "resnet18_layer3",
        "convnext_tiny",
        "dinov2_vits14",
    ]

    for n in names:
        print(f"\n[Test] {n}")
        backbone = build_local_backbone(n, img_size=224)
        feat, (H, W) = backbone.extract_grid(dummy, device)
        print(" feat shape:", tuple(feat.shape))
        print(" grid hw:", (H, W))