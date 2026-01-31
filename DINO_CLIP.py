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

def patch_to_grid(pt, Hp, Wp): #patch 토큰을 grid로 펼침
    # pt: [P,D] -> [1,D,Hp,Wp]
    D = pt.shape[-1]
    return pt.view(Hp, Wp, D).permute(2, 0, 1).unsqueeze(0)

def cosine_change_map_from_grids(gr, gq, eps=1e-6): #patch gird를 사용해 코사인 유사도 gird를 만듬
    # gr,gq: [1,D,h,w] -> [1,1,h,w]
    gr = F.normalize(gr, dim=1, eps=eps)
    gq = F.normalize(gq, dim=1, eps=eps)
    cos = (gr * gq).sum(dim=1, keepdim=True)
    return 1.0 - cos  # cosine distance

def multiscale_token_pool_change_map(
    q_tokens, r_tokens, Hp, Wp,
    pool_ks=(1, 2, 4),
    pool_type="avg",
    agg="median",
):
    """
    q_tokens, r_tokens: [P, D]
    returns: fused change map [Hp, Wp] (torch tensor on same device)
    """
    gq = patch_to_grid(q_tokens, Hp, Wp)  # [1,D,Hp,Wp]
    gr = patch_to_grid(r_tokens, Hp, Wp)

    maps = []
    for k in pool_ks:
        if k == 1:
            qk, rk = gq, gr
        else:
            if pool_type == "avg":
                qk = F.avg_pool2d(gq, kernel_size=k, stride=k)
                rk = F.avg_pool2d(gr, kernel_size=k, stride=k)
            elif pool_type == "max":
                qk = F.max_pool2d(gq, kernel_size=k, stride=k)
                rk = F.max_pool2d(gr, kernel_size=k, stride=k)
            else:
                raise ValueError("pool_type must be 'avg' or 'max'")

        cm = cosine_change_map_from_grids(rk, qk)  # [1,1,h,w]

        # back to original token grid size
        cm = F.interpolate(
            cm, size=(Hp, Wp),
            mode="bilinear", align_corners=False
        )
        maps.append(cm)

    stack = torch.cat(maps, dim=1)  # [1,S,Hp,Wp]

    if agg == "median":
        fused = stack.median(dim=1).values          # [1,Hp,Wp]
    elif agg == "top2mean":
        vals, _ = torch.sort(stack, dim=1, descending=True)
        fused = vals[:, :2].mean(dim=1)             # [1,Hp,Wp]
    elif agg == "mean":
        fused = stack.mean(dim=1)                   # [1,Hp,Wp]
    else:
        raise ValueError("agg must be 'median'/'top2mean'/'mean'")

    return fused[0]  # [Hp, Wp]

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
# 6) change map에 대한 후처리
# -------------------------

def constant_threshold_keep_value(m, tau):
    return np.where(m >= tau, m, 0.0)

def postprocess_change_map(change_map_hw,  top_p=None):
    """
    change_map_hw: [H,W] float numpy
    """
    m = change_map_hw.copy() # 528 528
    m = constant_threshold_keep_value(m,0.10) # constamt threshhold

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
    ref_path,                
    model_name="dinov2_vitb14",
    img_size=518,
    top_p=None,
    ms_pool_ks=(1,2,4),
    ms_pool_type="avg",
    ms_agg="median",
):

    model, device = load_dinov2(model_name=model_name)
    tfm = build_transform(img_size=img_size)

    # load q/r
    q_pil = read_image_rgb(query_path)
    q = tfm(q_pil).unsqueeze(0).to(device)

    r_pil = read_image_rgb(ref_path)
    r = tfm(r_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        q_tokens, Hp, Wp, patch_size = extract_patch_tokens(model, q)
        r_tokens, Hp2, Wp2, _        = extract_patch_tokens(model, r)
        assert (Hp, Wp) == (Hp2, Wp2)

        fused = multiscale_token_pool_change_map(
            q_tokens, r_tokens, Hp, Wp,
            pool_ks=ms_pool_ks,
            pool_type=ms_pool_type,
            agg=ms_agg
        )  # [Hp,Wp] torch

        # === 업샘플 방식도 개선 (repeat_interleave 말고 interpolate 추천)
        fused_up = F.interpolate(
            fused.unsqueeze(0).unsqueeze(0),  # [1,1,Hp,Wp]
            size=(img_size, img_size),
            mode="nearest"                    # patch 블록 유지
        )[0,0].detach().cpu().numpy()

    # 5) postprocess
    raw_map = fused_up.copy()
    fused_up = postprocess_change_map(
        fused_up,
        top_p=top_p
    )

    print("[RAW(pre)] min/mean/max =", raw_map.min(), raw_map.mean(), raw_map.max())
    print("[RAW(pre)] p95/p99      =", np.percentile(raw_map,95), np.percentile(raw_map,99))
    print("[POST]     nonzero      =", np.count_nonzero(fused_up))

    q_rgb0 = np.array(q_pil)
    r_rgb0 = np.array(r_pil)

    q_rgb_518 = np.array(transforms.CenterCrop(img_size)(
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)(q_pil)
            ))
    r_rgb_518 = np.array(transforms.CenterCrop(img_size)(
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)(r_pil)
            ))


    overlay_q = make_overlay(q_rgb_518, fused_up, alpha=0.20)
    overlay_r = make_overlay(r_rgb_518, fused_up, alpha=0.20)

    score = float(np.mean(fused_up))
    return fused_up, overlay_q, overlay_r, score, q_rgb_518 , r_rgb_518

# Clip 
@torch.no_grad() #text to embed
def clip_text_embeds(clip_model, tokenizer, prompts, device):
    """
    prompts: list[str]
    returns: [K, D] normalized
    """
    tokens = tokenizer(prompts).to(device)
    txt = clip_model.encode_text(tokens)  # [K,D]
    txt = F.normalize(txt, dim=-1)
    return txt

def build_cpe_class_text_embeds( # 파손 감지 텍스트 임베딩 생성기
    clip_model,
    tokenizer,
    device,
    object_label="rack",
    normal_state_words=None,
    anomalous_state_words=None,
    templates=None,
):
    if normal_state_words is None:
        normal_state_words = [
            "intact", "undamaged", "no defect", "no crack", "no scratch",
            "complete", "not broken", "clean surface"
        ]

    if anomalous_state_words is None:
        anomalous_state_words = [
            "broken", "cracked", "fractured", "chipped", "missing part",
            "dented", "scratched", "torn", "bent", "deformed"
        ]
    if templates is None:
        templates = [
            "a close-up photo of an {obj} that is {state}",
            "a photo of an {obj} with {state}",
            "a detailed inspection photo of an {obj}, {state}",
            "a product photo of an {obj}, {state}",
        ]

    def make_prompts(state_words):
        ps = []
        for st in state_words:
            for tmp in templates:
                ps.append(tmp.format(state=st, obj=object_label))
        return ps

    normal_prompts = make_prompts(normal_state_words)
    anomal_prompts  = make_prompts(anomalous_state_words)

    # normal / anomal --- prompt에 대한 평균 임베딩 추출 ------------------------
    def avg_text_emb(prompts):
        tokens = tokenizer(prompts).to(device)
        txt = clip_model.encode_text(tokens)          # [K,D]
        txt = F.normalize(txt, dim=-1)                # normalize each - 각각을 단위벡터로 만들기
        t = txt.mean(dim=0)                           # [D] - 각 벡터의 평균
        t = F.normalize(t, dim=-1)                    # normalize mean  - 다시 단위벡터로 만들기
        return t

    t_normal = avg_text_emb(normal_prompts)
    t_anom   = avg_text_emb(anomal_prompts)

    debug = {"normal_prompts": normal_prompts, "anomal_prompts": anomal_prompts}

    return t_normal, t_anom, debug

def apply_sliding_mask(rgb_u8, x0, y0, x1, y1):

    H, W, _ = rgb_u8.shape
    out = rgb_u8.copy()
    out[:] = 0
    out[y0:y1, x0:x1, :] = rgb_u8[y0:y1, x0:x1, :]

    return out

def extract_crop(rgb_u8, x0, y0, x1, y1):
    return rgb_u8[y0:y1, x0:x1, :]


@torch.no_grad()
def clip_neighbor_window_encodings_by_patch(
    clip_model,
    clip_preprocess,
    rgb_u8,                 # uint8 input img
    patch_size_px=16,        # model patch size
    win_patches=(3, 3),      # patch nxn window
    device=None
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    H, W = rgb_u8.shape[:2]

    Hp = H // patch_size_px
    Wp = W // patch_size_px
    H_eff = Hp * patch_size_px
    W_eff = Wp * patch_size_px

    win_ph, win_pw = win_patches

    st_ph, st_pw = win_ph, win_pw

    ys = list(range(0, Hp - win_ph + 1, st_ph))
    xs = list(range(0, Wp - win_pw + 1, st_pw))

    base = rgb_u8[:H_eff, :W_eff]

    embs = []
    bboxes = []

    for py in ys:
        row = []
        for px in xs:
            y0 = py * patch_size_px
            x0 = px * patch_size_px
            y1 = (py + win_ph) * patch_size_px
            x1 = (px + win_pw) * patch_size_px

            masked = extract_crop(base, x0, y0, x1, y1)
            pil = Image.fromarray(masked)

            x = clip_preprocess(pil).unsqueeze(0).to(device)
            f = clip_model.encode_image(x)
            f = F.normalize(f, dim=-1)  # [1,D]

            row.append(f.squeeze(0).detach().cpu())
            bboxes.append((x0, y0, x1, y1))
        embs.append(torch.stack(row, dim=0))

    embs = torch.stack(embs, dim=0)  # [Nh,Nw,D]
    return embs, bboxes, (Hp, Wp), (H_eff, W_eff)

@torch.no_grad()
def p_anom_map_softmax2(
    embs_hw_d, t_normal, t_anom,
    clip_model=None,     
    temp=1.0,
    use_logit_scale=True, 
    return_numpy=True
):
    device = t_normal.device
    E = F.normalize(embs_hw_d.to(device), dim=-1)

    sim_normal = (E * t_normal.view(1,1,-1)).sum(dim=-1)
    sim_anom   = (E * t_anom.view(1,1,-1)).sum(dim=-1)

    margin = sim_anom - sim_normal

    if use_logit_scale:
        assert clip_model is not None
        logit_scale = clip_model.logit_scale.exp().clamp(max=100).item()
        margin = margin * logit_scale

    margin = margin / max(1e-8, float(temp))
    p_map = torch.sigmoid(margin)

    if return_numpy:
        return (sim_normal.detach().cpu().numpy(),
                sim_anom.detach().cpu().numpy(),
                p_map.detach().cpu().numpy())
    else:
        return sim_normal, sim_anom, p_map

    

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


def window_map_to_image_heatmap( #clip anmaly map 시각화
    p_map,            # [Nh, Nw] numpy
    bboxes,           # list[(x0,y0,x1,y1)], row-major
    out_hw,            # (H,W) of original image
):
    """
    returns:
      heatmap: [H,W] float32
    """
    H, W = out_hw
    heat = np.zeros((H, W), dtype=np.float32)

    Nh, Nw = p_map.shape
    assert len(bboxes) == Nh * Nw

    idx = 0
    for i in range(Nh):
        for j in range(Nw):
            x0, y0, x1, y1 = bboxes[idx]
            val = float(p_map[i, j])

            # 안전 clip
            x0 = max(0, min(W, x0))
            x1 = max(0, min(W, x1))
            y0 = max(0, min(H, y0))
            y1 = max(0, min(H, y1))

            heat[y0:y1, x0:x1] = val
            idx += 1

    return heat

def normalize_heatmap_for_vis(hm, eps=1e-8):
    hm = hm.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + eps)
    return hm


if __name__ == "__main__":
    query = "/home/choi/capston/zeroshot_scene_ad/data/make/reck_light.png"
    ref   = "/home/choi/capston/zeroshot_scene_ad/data/make/rack_ref.png"

    #DINO change map
    change_map, overlay_q, overlay_r, score, q_rgb_518, r_rgb_518 = compute_change_map(
        query_path=query,
        ref_path=ref,
        top_p=10,
        ms_pool_ks=(1,2,4),
        ms_pool_type="avg",
        ms_agg="median",
    )

    # 2) ROI (DINO only)
    rois = extract_rois_from_change_map(change_map, min_area=150)
    print("num rois (DINO):", len(rois))

    # ----------------------CLIP
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_preprocess, clip_tokenizer, _ = load_openclip(
        model_name="ViT-B-16",
        pretrained="openai",
        device=device
    )

    t_normal, t_anom, dbg = build_cpe_class_text_embeds( #prompt 임베딩 추출
        clip_model=clip_model,
        tokenizer=clip_tokenizer,
        device=device,
        object_label="object"   
    )

    embs, bboxes, (Hp, Wp), (H_eff, W_eff) = clip_neighbor_window_encodings_by_patch(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        rgb_u8=q_rgb_518,
        patch_size_px=16,
        win_patches=(4, 4),
        device=device
    )

    sim_n, sim_a, p_map = p_anom_map_softmax2(
        embs_hw_d=embs,
        t_normal=t_normal,
        t_anom=t_anom,
        clip_model=clip_model,   
        temp=0.25,
        use_logit_scale=True,
        return_numpy=True
    )

    heat = window_map_to_image_heatmap(
        p_map=p_map,
        bboxes=bboxes,
        out_hw=q_rgb_518.shape[:2]
    )
    heat_vis = normalize_heatmap_for_vis(heat)
    clip_overlay_q = make_overlay(q_rgb_518, heat_vis, alpha=0.25)

    # (옵션) DINO change_map이랑 곱해서 "변화영역에서만" CLIP anomaly 보기
    fused = heat * (change_map > 0)
    fused_vis = normalize_heatmap_for_vis(fused)
    fused_overlay_q = make_overlay(q_rgb_518, fused_vis, alpha=0.25)

    # ============================================================
    # 6) 기존 시각화 + CLIP anomaly overlay 추가
    # ============================================================

    roi_vis = draw_rois_on_image(q_rgb_518, rois, color=(0,255,0), thickness=2)

    plt.figure(); plt.title("Query + ROI boxes (from DINO change_map)")
    plt.imshow(roi_vis); plt.axis("off")
    plt.figure(); plt.title("ref"); plt.imshow(r_rgb_518); plt.colorbar(); plt.axis("off")
    plt.figure(); plt.title("Change Map (DINO)"); plt.imshow(change_map); plt.colorbar(); plt.axis("off")
    plt.figure(); plt.title("Query + DINO heatmap"); plt.imshow(overlay_q); plt.axis("off")

    plt.figure(); plt.title("Query + CLIP-CPE anomaly (window) heatmap"); plt.imshow(clip_overlay_q); plt.axis("off")
    plt.figure(); plt.title("Query + (DINO change >0) * CLIP anomaly"); plt.imshow(fused_overlay_q); plt.axis("off")

    plt.show()


