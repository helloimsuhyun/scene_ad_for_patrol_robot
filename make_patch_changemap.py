"""

input : ref (1개 또는 N개), query 이미지
output : change map  + overlay vis + total change/anomaly score

"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from PIL import Image
import random
import open_clip


# ------------------------------------ model load

def load_dinov2(model_name="dinov2_vitb14", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model, device

sam_ckpt = "/home/choi/capston/zeroshot_scene_ad/ckpt/sam_vit_b_01ec64.pth"

def load_openclip(model_name="ViT-B-16", pretrained="openai", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


# -------------------------
# 2 전처리
#  - DINOv2는 보통 518 입력을 많이 씀
#  - Resize -> CenterCrop(518)로 통일 (patch grid 안정)
# -------------------------
def build_transform(img_size=518):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

def preprocess_518_rgb(pil_img): #SAM
    return np.array(
        transforms.CenterCrop(518)(
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC)(pil_img)
        )
    )



def read_image_rgb(path):
    return Image.open(path).convert("RGB").rotate(0, expand=True)


# -------------------------
# 3) patch tokens 추출
#  - output: [P, D], 그리고 patch grid (Hp, Wp)
# -------------------------
@torch.no_grad()
def extract_patch_tokens(model, x):
    feats = model.forward_features(x)
    pt = feats["x_norm_patchtokens"][0]  # [P, D]

    H, W = x.shape[-2], x.shape[-1]
    patch_size = 14
    Hp, Wp = H // patch_size, W // patch_size
    return pt, Hp, Wp, patch_size



# -------------------------
# 4) change map 계산 (cosine distance 권장)
# -------------------------
def patch_distance_map(q_tokens, r_tokens, Hp, Wp, metric="cosine"):
    """
    q_tokens, r_tokens: [P, D]
    return: [Hp, Wp] float32
    """
    assert q_tokens.shape == r_tokens.shape
    if metric == "cosine":
        qn = F.normalize(q_tokens, dim=-1)
        rn = F.normalize(r_tokens, dim=-1)
        dist = 1.0 - (qn * rn).sum(dim=-1)  # [P]
    elif metric == "l2":
        dist = torch.norm(q_tokens - r_tokens, dim=-1)  # [P]
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")

    dist_map = dist.view(Hp, Wp).float()
    return dist_map


# -------------------------
# 5) robust aggregation (mean/median/trimmed mean)
# -------------------------
def aggregate_maps(maps, mode="mean", trim_ratio=0.1): #여러 ref의 dist map을 합치는 함수
    """
    maps: list of [Hp,Wp] torch tensors
    returns: [Hp,Wp]
    """
    stack = torch.stack(maps, dim=0)  # [N,Hp,Wp]
    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "median":
        return stack.median(dim=0).values
    if mode == "trimmed_mean":
        # 상하 trim_ratio 만큼 잘라내고 평균
        N = stack.shape[0]
        k = int(N * trim_ratio)
        if N < 3 or k == 0:
            return stack.mean(dim=0)
        sorted_stack, _ = torch.sort(stack, dim=0)  # [N,Hp,Wp]
        trimmed = sorted_stack[k:N-k, ...]
        return trimmed.mean(dim=0)
    raise ValueError("mode must be mean/median/trimmed_mean")



# -------------------------
# 6) change map에 대한 후처리
# -------------------------

def constant_threshold_keep_value(m, tau):
    return np.where(m >= tau, m, 0.0)

def postprocess_change_map(change_map_hw,  top_p=None):
    """
    change_map_hw: [H,W] float numpy
    """
    m = change_map_hw.copy() # 528 528
    m = constant_threshold_keep_value(m,0.30) # constamt threshhold

    if top_p is not None:
        # 상위 p%만 남기기 (0~100)
        thr = np.percentile(m, 100 - top_p)
        m = np.where(m >= thr, m, 0.0)
    
    print("changmap original.shape")
    print(m.shape)

    return m


# -------------------------
# heatmap overlay 시각화

def make_overlay(rgb_img, heatmap, alpha=0.45):
    """
    rgb_img: [H,W,3] uint8
    heatmap: [H,W] float
    """
    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-8)

    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    overlay = (rgb_img.astype(np.float32) * (1 - alpha) + hm_color.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


## ----------------- changemap roi
def adaptive_percentile(n_rois):
    if n_rois <= 5:
        return 0      # 전부 유지
    elif n_rois <= 10:
        return 50     # 상위 50%
    elif n_rois <= 20:
        return 65
    else:
        return 75     # 상위 25%

def extract_rois_from_change_map(
    change_map,
    min_area=150,
):
    H, W = change_map.shape

    binary = (change_map > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    rois = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        x0, y0, x1, y1 = x, y, x + w, y + h
        blob_mask = (labels == i)

        mass = float(change_map[blob_mask].sum())
        density = mass / (area + 1e-8)
        score = mass * density

        rois.append({
            "id": i,
            "bbox": (x0, y0, x1, y1),
            "area": int(area),
            "mass": mass,
            "density": density,
            "score": score,
            "mask": blob_mask,
        })

    if not rois:
        return []

    # adaptive percentile
    p = adaptive_percentile(len(rois))
    scores = np.array([r["score"] for r in rois], dtype=np.float32)
    thr = np.percentile(scores, p)
    rois = [r for r in rois if r["score"] >= thr]

    rois.sort(key=lambda r: r["score"], reverse=True)
    return rois


def expand_bbox_xyxy(bbox, scale=2.0, W=518, H=518): # changemap으로 찾은 roi영역의 2배를 sam roi로 사용
    """
    bbox: (x0,y0,x1,y1) in pixel coords
    scale=2.0 => 너가 말한 '2배 영역' (width/height 각각 2배)
    """
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w  = (x1 - x0)
    h  = (y1 - y0)

    w2 = w * scale
    h2 = h * scale

    nx0 = int(round(cx - 0.5 * w2))
    ny0 = int(round(cy - 0.5 * h2))
    nx1 = int(round(cx + 0.5 * w2))
    ny1 = int(round(cy + 0.5 * h2))

    # clip
    nx0 = max(0, min(W - 1, nx0))
    ny0 = max(0, min(H - 1, ny0))
    nx1 = max(1, min(W, nx1))
    ny1 = max(1, min(H, ny1))

    # ensure valid
    if nx1 <= nx0 + 1: nx1 = min(W, nx0 + 2)
    if ny1 <= ny0 + 1: ny1 = min(H, ny0 + 2)

    return (nx0, ny0, nx1, ny1)

# ------------------------- 메인 함수
def compute_change_map(
    query_path,
    ref_paths,
    model_name="dinov2_vitb14",
    metric="cosine",
    agg="mean",
    img_size=518,
    blur_ksize=0,
    morph_ksize=0,
    top_p=None,
):
    model, device = load_dinov2(model_name=model_name)
    tfm = build_transform(img_size=img_size)

    # loag q img
    q_pil = read_image_rgb(query_path)
    q = tfm(q_pil).unsqueeze(0).to(device) #resize and make tensor

    # encoding
    with torch.no_grad():
        q_tokens, Hp, Wp , patch_size = extract_patch_tokens(model, q)

        # ref loop
        maps = []
        last_r_pil = None
        for rp in ref_paths:
            r_pil = read_image_rgb(rp)
            last_r_pil = r_pil

            r = tfm(r_pil).unsqueeze(0).to(device)
            r_tokens, Hp2, Wp2, _  = extract_patch_tokens(model, r)
            assert (Hp, Wp) == (Hp2, Wp2)

            dm = patch_distance_map(q_tokens, r_tokens, Hp, Wp, metric=metric)
            maps.append(dm)

        fused = aggregate_maps(maps, mode=agg)  # 여러개 ref의 distance map을 합침


        # patch 값 그대로 P×P 영역에 복제 (patch 안의 distance가 상수값)
        up = fused.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)  

        H, W = img_size, img_size
        H0, W0 = up.shape
        pad_h = max(0, H - H0)
        pad_w = max(0, W - W0)

        if pad_h > 0 or pad_w > 0:
            up = F.pad(up[None, None], (0, pad_w, 0, pad_h), mode="replicate")[0, 0]

        fused_up = up[:H, :W].detach().cpu().numpy()  # [518,518]


    # 5) postprocess
    fused_up = postprocess_change_map(
        fused_up,
        top_p=top_p
    )

    # fused map(518x518) -> 원본 크기(H0,W0)로 확대 (섞지 않게 nearest) & overlay
    q_rgb0 = np.array(q_pil)
    r_rgb0 = np.array(last_r_pil)

    q_rgb0 = np.array(q_pil)
    r_rgb0 = np.array(last_r_pil)

    # fused_up(518) -> 시각화
    q_rgb_518 = np.array(transforms.CenterCrop(img_size)(
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)(q_pil)
            ))
    r_rgb_518 = np.array(transforms.CenterCrop(img_size)(
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)(last_r_pil)
            ))

    overlay_q = make_overlay(q_rgb_518, fused_up, alpha=0.20)
    overlay_r = make_overlay(r_rgb_518, fused_up, alpha=0.20)

    score = float(np.mean(fused_up))
    return fused_up, overlay_q, overlay_r, score, q_rgb_518 , r_rgb_518

# Clip 
@torch.no_grad()
def clip_text_embeds(clip_model, tokenizer, prompts, device):
    """
    prompts: list[str]
    returns: [K, D] normalized
    """
    tokens = tokenizer(prompts).to(device)
    txt = clip_model.encode_text(tokens)  # [K,D]
    txt = F.normalize(txt, dim=-1)
    return txt

@torch.no_grad()
def clip_patch_tokens(clip_model, pil_img, preprocess, device):
    """
    OpenCLIP ViT visual에서 patch tokens를 안정적으로 추출.
    Returns:
      patch: [P, D]   (CLS 제거)
      Hp, Wp, patch_size, (H, W)
    """
    x = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,H,W] usually 224
    H, W = x.shape[-2], x.shape[-1]

    visual = clip_model.visual

    # --- ViT 계열만 지원 (ViT-B-16, ViT-L-14 등)
    if not (hasattr(visual, "conv1") and hasattr(visual, "transformer")):
        raise RuntimeError("This extractor expects a ViT-based visual backbone (with conv1/transformer).")

    # 1) patch embedding: conv1
    # conv1: [out_dim, 3, patch, patch], stride=patch
    x = visual.conv1(x)  # [1, width, grid_h, grid_w]
    grid_h, grid_w = x.shape[-2], x.shape[-1]

    x = x.reshape(x.shape[0], x.shape[1], grid_h * grid_w)  # [1, width, P]
    x = x.permute(0, 2, 1)  # [1, P, width]

    # 2) prepend CLS
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)  # [1,1,width]
    x = torch.cat([cls, x], dim=1)  # [1, 1+P, width]

    # 3) add positional embedding
    pos = visual.positional_embedding.to(x.dtype)
    if pos.shape[0] != x.shape[1]:
        # 일부 구현에서 입력 해상도가 다르면 pos_emb가 맞지 않을 수 있음.
        # openai pretrained는 보통 224 고정이므로 여기로 오면 preprocess 사이즈를 확인해야 함.
        raise RuntimeError(f"Positional embedding length mismatch: pos {pos.shape[0]} vs tokens {x.shape[1]}. "
                           f"Check preprocess resolution.")
    x = x + pos

    # 4) layer norm pre (있으면)
    if hasattr(visual, "ln_pre") and visual.ln_pre is not None:
        x = visual.ln_pre(x)

    # 5) transformer (open_clip은 보통 [seq, batch, dim] 형태를 씀)
    x = x.permute(1, 0, 2)  # [1+P, 1, width]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)  # [1, 1+P, width]

    # 6) final norm (있으면)
    if hasattr(visual, "ln_post") and visual.ln_post is not None:
        x = visual.ln_post(x)

    # x: [1,1+P,D] (D=width or proj 이후 dim은 모델마다 다름)
    tok = x[0]        # [1+P, D]
    patch = tok[1:]   # [P, D]

    if hasattr(visual, "proj") and visual.proj is not None:
        # visual.proj: [width, embed_dim] or Linear-like
        if isinstance(visual.proj, torch.Tensor):
            patch = patch @ visual.proj  # [P, embed_dim]
        else:
            patch = visual.proj(patch)   # [P, embed_dim]

    patch = F.normalize(patch, dim=-1)

    if hasattr(visual.conv1, "kernel_size"):
        patch_size = int(visual.conv1.kernel_size[0])
    else:
        patch_size = 16

    Hp, Wp = grid_h, grid_w  # 실제 grid
    return patch, Hp, Wp, patch_size, (H, W)


@torch.no_grad()
def clip_roi_diff_scores(
    clip_model, clip_preprocess,
    q_rgb_518, r_rgb_518,
    rois,
    device=None,
    min_side=8,
    use_expand=True,
    expand_scale=1.6
):
    """
    ROI마다 query/ref crop을 CLIP image embedding으로 비교.
    Returns:
      roi_scores: list of dict {id, score_clipdiff}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H, W = q_rgb_518.shape[:2]

    out = []

    for r in rois:
        x0, y0, x1, y1 = r["bbox"]
        roi_w = (x1 - x0)
        roi_h = (y1 - y0)
        roi_area_ratio = (roi_w * roi_h) / float(W * H + 1e-8)

        if use_expand:
            if roi_area_ratio < 0.15:   # 전체의 15% 미만일 때만 확장
                x0, y0, x1, y1 = expand_bbox_xyxy((x0, y0, x1, y1), scale=expand_scale, W=W, H=H)
            
        if (x1 - x0) < min_side or (y1 - y0) < min_side:
            out.append({"id": r["id"], "score_clipdiff": 0.0})
            continue

        q_crop = q_rgb_518[y0:y1, x0:x1]
        r_crop = r_rgb_518[y0:y1, x0:x1]

        q_pil = Image.fromarray(q_crop)
        r_pil = Image.fromarray(r_crop)

        q_in = clip_preprocess(q_pil).unsqueeze(0).to(device)
        r_in = clip_preprocess(r_pil).unsqueeze(0).to(device)

        q_emb = F.normalize(clip_model.encode_image(q_in), dim=-1)
        r_emb = F.normalize(clip_model.encode_image(r_in), dim=-1)

        cos = (q_emb * r_emb).sum(dim=-1).item()
        score = float(1.0 - cos)  # 0(유사) ~ 2(반대), 보통 0~1

        out.append({"id": r["id"], "score_clipdiff": score})

    return out



# ---- 시각화
def draw_rois_on_image(rgb, rois, color=(0, 255, 0), thickness=2):
    """
    rgb: (H,W,3) uint8
    rois: list of roi dicts
    returns: (H,W,3) uint8
    """
    out = rgb.copy()
    for r in rois:
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)

        label = f"id{r['id']} dino={r['score']:.3f} clipd={r.get('clipdiff',0):.3f}"
        print(f"id : {r['id']} , dino score : {r['score']} , clipd score : {r.get('clipdiff',0)}")
        cv2.putText(
            out,
            label,
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


if __name__ == "__main__":
    query = "/home/choi/capston/zeroshot_scene_ad/data/renders_multicam_diff_1/1_change_l.png"
    refs  = ["data/renders_multicam_diff_1/1.png"]

    # 1) DINO change map
    change_map, overlay_q, overlay_r, score, q_rgb_518, r_rgb_518 = compute_change_map(
        query_path=query,
        ref_paths=refs,
        metric="cosine",
        agg="mean",
        blur_ksize=7,
        morph_ksize=3,
        top_p=None,
    )
    print("DINO score =", score)

    # 2) ROI (DINO only)
    rois = extract_rois_from_change_map(change_map, min_area=150)
    print("num rois (DINO):", len(rois))

    # 3) CLIP ROI gate scoring
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess, clip_tokenizer, _ = load_openclip(
        model_name="ViT-B-16",
        pretrained="openai",
        device=device
    )

    # 3-B) CLIP image-encoder 기반 ROI diff scoring
    roi_clipdiff = clip_roi_diff_scores(
        clip_model, clip_preprocess,
        q_rgb_518=q_rgb_518,
        r_rgb_518=r_rgb_518,
        rois=rois,
        device=device,
        use_expand=True,   
        expand_scale=1.6
    )
    print([d["score_clipdiff"] for d in roi_clipdiff])

    clipdiff_map = {d["id"]: float(d["score_clipdiff"]) for d in roi_clipdiff}
    for r in rois:
        r["clipdiff"] = float(clipdiff_map.get(r["id"], 0.0))


    eps_clip = 0.01  # 시작 0.01~0.02 추천 (데이터 보며 조정)
    if len(rois) >= 5:
        dino_keep_thr = float(np.percentile([float(r["score"]) for r in rois], 80))
    else:
        dino_keep_thr = -1e9  # ROI 적으면 dino 예외처리 의미 없으니 사실상 clip만 봄

    keep_ids = set()
    for r in rois:
        cid = r["id"]
        dino_s = float(r["score"])
        clip_s = float(clipdiff_map.get(cid, 0.0))

        # clip이 거의 0인데 dino도 강하지 않으면 제거
        if (clip_s < eps_clip) and (dino_s < dino_keep_thr):
            continue
        keep_ids.add(cid)

    rois_filtered = [r for r in rois if r["id"] in keep_ids]

    print("eps_clip =", eps_clip, "dino_keep_thr =", dino_keep_thr)
    print("rois before/after:", len(rois), "->", len(rois_filtered))


    # 5) 시각화
    roi_vis_all = draw_rois_on_image(q_rgb_518, rois)
    r_roi_vis_all = draw_rois_on_image(r_rgb_518, rois)
    roi_vis_keep = draw_rois_on_image(q_rgb_518, rois_filtered)


    plt.figure(); plt.title("Change Map (DINO)"); plt.imshow(change_map); plt.colorbar(); plt.axis("off")
    plt.figure(); plt.title("Query + DINO heatmap"); plt.imshow(overlay_q); plt.axis("off")
    plt.figure(); plt.title("Query + ROI bboxes (all)"); plt.imshow(roi_vis_all); plt.axis("off")
    plt.figure(); plt.title("r + ROI bboxes (all)"); plt.imshow(r_roi_vis_all); plt.axis("off")
    
    plt.figure(); plt.title("Query + ROI bboxes (CLIP-filtered)"); plt.imshow(roi_vis_keep); plt.axis("off")
    plt.show()

