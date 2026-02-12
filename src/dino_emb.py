#dino feature 추출기와 전처리기
#-suhyun


import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def load_dinov2(model_name="dinov2_vits14", device=None):
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
def make_embed(model, device, x):
    # input x is must be tensor [3,H,W] in range [0,1]
    # output : torch.Tensor [D] cpu

    x = x.unsqueeze(0).to(device)          # [1,3,H,W]
    feats = model.forward_features(x)
    patch = feats["x_norm_patchtokens"]           # [1,N,D]
    emb = patch.mean(dim=1)                       # 패치 토큰들의 평균 임베딩 [1,D]
    emb = F.normalize(emb, dim=-1)                # 크기 1로 정규화 L2 normalize
    return emb.squeeze(0).detach().cpu()          