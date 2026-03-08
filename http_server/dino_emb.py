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

def make_transform(img_size=560):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

#feature 추출기 :  cls와 patch token을 return
@torch.no_grad()
def extract_feats(model, device, x):
    x = x.unsqueeze(0).to(device)
    feats = model.forward_features(x)
    cls = feats["x_norm_clstoken"]          # [1,D]
    patch = feats["x_norm_patchtokens"]     # [1,P,D]
    return cls, patch


#모드에 맞추어 emb를 반환
@torch.no_grad()
def make_embed(model, device, x, 
               repr_mode="global",  # "global" | "patch" | "global_patch"
               global_mode="patch_mean"): # "cls" | "patch_mean"
    
    cls, patch = extract_feats(model, device, x)

    # global emb
    if global_mode == "cls":
        gvec = cls.squeeze(0)
    elif global_mode == "patch_mean":
        gvec = patch.mean(dim=1).squeeze(0)
    else:
        raise ValueError("unknown global_mode")
    gvec = F.normalize(gvec, dim=-1)

    p = patch.squeeze(0)          # (P,D)
    p = F.normalize(p, dim=-1)    # (P,D)

    #mode별 emb return
    if repr_mode == "global":
        return {"global": gvec}

    if repr_mode == "patch":
        return {"patch": p}

    if repr_mode in {"global_patch", "patch_global", "global_patch_pool"}:
        return {"global": gvec, "patch": p}

    raise ValueError(f"unknown repr_mode={repr_mode} (use global|patch|global_patch)")

