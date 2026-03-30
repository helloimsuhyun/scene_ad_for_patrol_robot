# cnn_emb.py
# cnn feature 추출기와 전처리기
# - suhyun

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


class ResNetGridExtractor(nn.Module):
    def __init__(self, out_layer="layer3"):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        if out_layer not in {"layer1", "layer2", "layer3", "layer4"}:
            raise ValueError("out_layer must be one of layer1/layer2/layer3/layer4")

        self.out_layer = out_layer

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.out_layer == "layer1":
            return x

        x = self.layer2(x)
        if self.out_layer == "layer2":
            return x

        x = self.layer3(x)
        if self.out_layer == "layer3":
            return x

        x = self.layer4(x)
        return x


def load_model(model_name="resnet18", out_layer="layer3", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "resnet18":
        model = ResNetGridExtractor(out_layer=out_layer)
    else:
        raise ValueError(f"unknown model_name={model_name}")

    model.eval().to(device)
    return model, device

def make_transform(img_size=560):
    return transforms.Compose([
        transforms.Lambda(lambda img: (
            TF.center_crop(img, (img.height, int(img.height * 4 / 3)))
            if img.width / img.height > 4/3
            else TF.center_crop(img, (int(img.width * 3 / 4), img.width))
        )),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

def make_aligned_local_transform(img_size=560):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

@torch.no_grad()
def extract_grid_layers(model, device, x):
    x = x.unsqueeze(0).to(device)      # [1,3,H,W]
    feat = model(x).squeeze(0)         # [C,H,W]

    C, H, W = feat.shape
    p = feat.permute(1, 2, 0).reshape(H * W, C)   # [P,D]
    p = F.normalize(p, dim=-1)

    return p, (H, W)