#dino feature 추출기와 전처리기
#-suhyun


import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def load_model(model_name="dinov2_vits14", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model, device

def make_transform(img_size=518):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

@torch.no_grad()
def make_embed(model, device, x, mode="cls"):

    # [1,3,H,W]
    x = x.unsqueeze(0).to(device)

    feats = model.forward_features(x)

    cls = feats.get("x_norm_clstoken", None)
    patch = feats.get("x_norm_patchtokens", None)
    storage = feats.get("x_storage_tokens", None)

    if mode == "cls":
        if cls is None:
            raise KeyError(f"cls not found. keys={list(feats.keys())}")
        emb = cls

    elif mode == "patch_mean":
        if patch is None:
            raise KeyError(f"patch not found. keys={list(feats.keys())}")
        emb = patch.mean(dim=1)

    elif mode == "cls+patch_mean":
        if cls is None or patch is None:
            raise KeyError(f"need cls+patch. keys={list(feats.keys())}")
        emb = torch.cat([cls, patch.mean(dim=1)], dim=-1)

    elif mode == "cls+storage+patch_mean":
        if cls is None or patch is None or storage is None:
            raise KeyError(f"need cls+storage+patch. keys={list(feats.keys())}")
        emb = torch.cat(
            [cls, storage.mean(dim=1), patch.mean(dim=1)],
            dim=-1,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0).detach().cpu()



"""

# dinov3 feature extractor + preprocess (torch.hub)
# -suhyun

import torch
import torch.nn.functional as F
from torchvision import transforms

def load_model(model_name="dinov3_vits16", device=None, pretrained=True):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # DINOv3: torch.hub.load('facebookresearch/dinov3', <entrypoint>, ...)
    repo_dir = "/home/choisuhyun/scene_ad_for_patrol_robot/dinov3"
    ckpt = "/home/choisuhyun/scene_ad_for_patrol_robot/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    model = torch.hub.load(repo_dir, model_name, source='local' , weights=ckpt)

    model.eval().to(device)
    return model, device

def make_transform(img_size=224):
    # DINOv3 backbones are defined with img_size=224 by default :contentReference[oaicite:2]{index=2}
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

@torch.no_grad()
def make_embed(model, device, x, mode="cls+patch_mean"):

    x = x.unsqueeze(0).to(device)  # [1,3,H,W]
    feats = model.forward_features(x)

    # 안전하게 get
    cls = feats.get("x_norm_clstoken", None)
    patch = feats.get("x_norm_patchtokens", None)
    storage = feats.get("x_storage_tokens", None)  # 보통 이미 norm된 형태로 나옴(없을 수도)

    if mode == "cls":
        if cls is None:
            raise KeyError(f"cls not found. keys={list(feats.keys())}")
        emb = cls  # [1,D]

    elif mode == "patch_mean":
        if patch is None:
            raise KeyError(f"patch not found. keys={list(feats.keys())}")
        emb = patch.mean(dim=1)  # [1,D]

    elif mode == "cls+patch_mean":
        if cls is None or patch is None:
            raise KeyError(f"need cls+patch. keys={list(feats.keys())}")
        emb = torch.cat([cls, patch.mean(dim=1)], dim=-1)  # [1,2D]

    elif mode == "cls+storage+patch_mean":
        if cls is None or patch is None or storage is None:
            raise KeyError(f"need cls+storage+patch. keys={list(feats.keys())}")
        emb = torch.cat([cls, storage.mean(dim=1), patch.mean(dim=1)], dim=-1)  # [1,3D]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).detach().cpu()

"""