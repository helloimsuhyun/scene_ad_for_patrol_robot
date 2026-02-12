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

from segment_anything import sam_model_registry, SamPredictor


# ------------------------------------ model load

def load_dinov2(model_name="dinov2_vitb14", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model, device

def load_sam2(
    sam_ckpt="/home/choisuhyun/scene_ad_for_patrol_robot/ckpt/sam_vit_b_01ec64.pth",
    model_type="vit_b",
    device="cuda",
):
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


# DINO - 전처리 --------------------------------------------------------------------------

def build_transform(img_size=518):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

def read_image_rgb(path):
    return Image.open(path).convert("RGB").rotate(0, expand=True)


#  DION - patch token 추출 ------------------------------------------------------------
@torch.no_grad()
def extract_patch_tokens(model, x): #model에서 patch 토큰을 추출
    feats = model.forward_features(x)
    pt = feats["x_norm_patchtokens"][0]  # [P, D]

    H, W = x.shape[-2], x.shape[-1]
    patch_size = 14
    Hp, Wp = H // patch_size, W // patch_size
    return pt, Hp, Wp, patch_size

def patch_to_grid(pt, Hp, Wp): # patch 토큰을 grid로 펼침
    # pt: [P,D] -> [1,D,Hp,Wp]
    D = pt.shape[-1]
    return pt.view(Hp, Wp, D).permute(2, 0, 1).unsqueeze(0)

def cosine_change_map_from_grids(gr, gq, eps=1e-6): # patch gird를 사용해 코사인 유사도 gird를 만듬
    # gr,gq: [1,D,h,w] -> [1,1,h,w]
    gr = F.normalize(gr, dim=1, eps=eps)
    gq = F.normalize(gq, dim=1, eps=eps)
    cos = (gr * gq).sum(dim=1, keepdim=True)
    return 1.0 - cos  # cosine distance

def multiscale_token_pool_change_map( #멀티 스케일 풀링 체인지맵
    q_tokens, r_tokens, Hp, Wp,
    pool_ks=(1, 2, 4),
    pool_type="avg",
    agg="top2mean",
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

def patch_distance_map(q_tokens, r_tokens, Hp, Wp, metric="cosine"): #cos 유사도 계산
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
# DINO change map 후처리 -  threshhold , k-p


def constant_threshold_keep_value(m, tau):
    return np.where(m >= tau, m, 0.0)

def postprocess_change_map(m):
    """
    tau_q: 노이즈 바닥 추정 퍼센타일 (튜닝 최소: 85~95 중 하나 고정)
    keep_top_percent: 최종 상위 몇 %만 유지
    """
    raw = m.copy()
    raw = constant_threshold_keep_value(raw, 0.2)

    return raw


# -------------------------



## ----------------- changemap roi

def remove_inside_rois(rois, inside_thr=0.95):
    """
    inside_thr: 작은 bbox가 큰 bbox에 얼마나 들어가면 '포함'으로 볼지
                0.95면 거의 완전 포함일 때만 제거 -> 안정적(튜닝 부담 적음)
    """
    def area(b):
        x0, y0, x1, y1 = b
        return max(0, x1 - x0) * max(0, y1 - y0)

    def inter_area(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        return max(0, ix1 - ix0) * max(0, iy1 - iy0)

    keep = []
    for i, r in enumerate(rois):
        rb = r["bbox"]
        ra = area(rb)
        is_inside = False

        for j, s in enumerate(rois):
            if i == j:
                continue
            sb = s["bbox"]
            sa = area(sb)
            if sa <= ra:
                continue  # s가 더 크지 않으면 r을 포함할 수 없음

            ia = inter_area(rb, sb)
            if ra > 0 and (ia / ra) >= inside_thr:
                is_inside = True
                break

        if not is_inside:
            keep.append(r)

    return keep

def pick_local_maxima_topk(change_map, topk=3, k=3, neigh=11, min_dist=96,
                           min_percentile=90, abs_floor=0.0):
    cm = change_map.astype(np.float32)
    k = k * 14

    E = cv2.boxFilter(cm, -1, (k, k), normalize=False)

    neigh = neigh | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (neigh, neigh))
    E_dil = cv2.dilate(E, kernel)
    local_max = (E == E_dil) & (E > 0)

    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return []

    vals = E[ys, xs]
    thr = max(float(np.percentile(vals, min_percentile)), float(abs_floor))

    keep = vals >= thr
    xs, ys, vals = xs[keep], ys[keep], vals[keep]
    if len(xs) == 0:
        return []

    order = np.argsort(-vals)

    peaks = []
    for idx in order:
        x, y, v = int(xs[idx]), int(ys[idx]), float(vals[idx])
        if any((x - px) ** 2 + (y - py) ** 2 < (min_dist ** 2) for (px, py, _) in peaks):
            continue
        peaks.append((x, y, v))
        if len(peaks) >= topk:
            break

    return peaks


def pick_win_by_mass_score(change_map, x, y, win_levels= (48, 72, 96, 128, 160), alpha=0.5):
    H, W = change_map.shape
    best = None

    for win in win_levels:
        half = win // 2
        x0 = max(0, x - half); x1 = min(W, x + half)
        y0 = max(0, y - half); y1 = min(H, y + half)

        patch = change_map[y0:y1, x0:x1]
        mass = float(patch.sum())
        area = float((y1 - y0) * (x1 - x0))
        score = mass / (area ** alpha + 1e-8)

        if (best is None) or (score > best["score"]):
            best = {"win": win, "bbox": (x0,y0,x1,y1), "mass": mass, "area": area, "score": score}

    return best  # dict

def peaks_to_rois(change_map, peaks, win_levels=(48,64,80,96,128), alpha=0.5):
    rois = []
    for rid, (x, y, _) in enumerate(peaks, start=1):
        best = pick_win_by_mass_score(change_map, x, y, win_levels=win_levels, alpha=alpha)
        x0,y0,x1,y1 = best["bbox"]
        rois.append({
            "id": rid,
            "bbox": (x0,y0,x1,y1),
            "area": int(best["area"]),
            "mass": best["mass"],
            "density": best["mass"]/(best["area"]+1e-8),
            "score": best["mass"],          
            "scale_score": best["score"],  
            "win": best["win"],
            "mask": None,
            "peak_xy": (x, y),
        })
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

def conditional_expand_bbox_xyxy(bbox, W, H, scale_small=2.0, area_ratio_thr=0.12):
    x0, y0, x1, y1 = bbox
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    area_ratio = (bw * bh) / float(W * H + 1e-9)

    scale = 1.0 if area_ratio >= area_ratio_thr else scale_small
    return expand_bbox_xyxy(bbox, scale=scale, W=W, H=H), scale, area_ratio


# ------------------------- 메인 함수
def compute_change_map(
    query_path,
    ref_path,
    model,
    device,                
    img_size=518,
    top_p=None,
    ms_pool_ks=(1,2,4),
    ms_pool_type="avg",
    ms_agg="top2mean",
):

    
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
    fused_up = postprocess_change_map(fused_up)

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

# input to vlm ----------------------------------------------------------
def make_pair_image_lr(
    ref_rgb,
    query_rgb,
    out_h=336,
    border=8,
    add_labels=True,
    label_h=34,
):
    assert ref_rgb.dtype == np.uint8 and query_rgb.dtype == np.uint8
    assert ref_rgb.ndim == 3 and query_rgb.ndim == 3 and ref_rgb.shape[2] == 3 and query_rgb.shape[2] == 3

    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * (target_h / h))))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    ref_r = resize_keep_aspect(ref_rgb, out_h)
    qry_r = resize_keep_aspect(query_rgb, out_h)

    # --- 이중 구분선(white+black)
    b1 = max(1, border // 2)
    b2 = max(1, border - b1)
    sep_white = np.full((out_h, b1, 3), 255, dtype=np.uint8)
    sep_black = np.zeros((out_h, b2, 3), dtype=np.uint8)
    sep = np.concatenate([sep_white, sep_black], axis=1)

    pair = np.concatenate([ref_r, sep, qry_r], axis=1)

    if not add_labels:
        return pair

    # --- 라벨 바
    W = pair.shape[1]
    bar = np.full((label_h, W, 3), 255, dtype=np.uint8)

    # --- 텍스트를 "해당 영역 폭"에 맞게 자동 축소해서 중앙에 넣기
    def put_text_fit(img, text, x0, x1, y, margin=6):
        region_w = max(1, (x1 - x0) - 2 * margin)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 시작값(큰 값에서 줄여나감)
        font_scale = 0.8
        thickness = 2

        # 폭에 맞을 때까지 scale 줄이기
        while True:
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            if tw <= region_w or font_scale <= 0.35:
                break
            font_scale -= 0.05
            thickness = 1 if font_scale < 0.55 else 2

        # 중앙 정렬 x
        x = x0 + (x1 - x0 - tw) // 2
        x = max(x0 + margin, x)  # 왼쪽 margin 보장

        # y(베이스라인) 안전하게
        y = int(y)
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    ref_w = ref_r.shape[1]
    sep_w = sep.shape[1]
    qry_w = qry_r.shape[1]

    # bar 영역 분할: [ref 영역 | sep 영역 | qry 영역]
    y = int(label_h * 0.75)

    put_text_fit(bar, "REF (Left)", 0, ref_w, y)
    put_text_fit(bar, "QRY (Right)", ref_w + sep_w, ref_w + sep_w + qry_w, y)

    out = np.concatenate([bar, pair], axis=0)
    return out


def crop_xyxy(rgb, bbox):
    x0, y0, x1, y1 = bbox
    return rgb[y0:y1, x0:x1].copy()

def prepare_vlm_pairs_from_rois(
    q_rgb,           # np.uint8 (H,W,3)
    r_rgb,           # np.uint8 (H,W,3)
    rois,            # 이미 정리된 ROI list
    expand_scale=1.5,
    img_size=518,
    out_h=336,
    border=4,
):

    if not rois:
        return []

    candidates = []

    for idx, r in enumerate(rois):
        bbox = r["bbox"]
        score = float(r.get("score", 0.0))
        label = r['id']

        bbox_exp, used_scale, area_ratio = conditional_expand_bbox_xyxy(
            bbox, W=img_size, H=img_size,
            scale_small=expand_scale,
            area_ratio_thr=0.12,
        )

        qry_crop = crop_xyxy(q_rgb, bbox_exp)
        ref_crop = crop_xyxy(r_rgb, bbox_exp)

        pair_rgb = make_pair_image_lr(
            ref_rgb=ref_crop,
            query_rgb=qry_crop,
            out_h=out_h,
            border=border,
        )

        candidates.append({
            "label" : label,
            "rank": idx,
            "score": score,
            "bbox": bbox,
            "bbox_exp": bbox_exp,
            "pair_rgb": pair_rgb,   
            "qry_crop": qry_crop,
            "ref_crop": ref_crop
        })

    return candidates


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
        print(f"id : {r['id']} , dino score : {r['score']}")
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

def visualize_vlm_pairs(candidates, max_show=10, cols=5, figsize=(18, 7)):
    """
    candidates: prepare_vlm_pairs_from_rois 출력(list)
    각 candidate["pair_rgb"]를 subplot으로 보여줌.
    """
    if not candidates:
        print("No candidates to show.")
        return

    show = candidates[:max_show]
    n = len(show)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=figsize)
    for i, c in enumerate(show):
        pair = c["pair_rgb"]
        title = f"rank={c['rank']} score={c['score']:.1f}\nexp={c['bbox_exp']}"
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(pair)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()

# ------------------sam

def sam_predict_mask_from_box(predictor, bbox_xyxy, multimask=True):
    box = np.array(bbox_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],              
        multimask_output=multimask
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best]), masks.astype(bool), scores


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

if __name__ == "__main__":
    query = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/017/RGB/1_00.png"
    ref   = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/017/RGB/2_00.png"
    
    #load models
    sam = load_sam2(
        sam_ckpt="/home/choisuhyun/scene_ad_for_patrol_robot/ckpt/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="cuda",
    )

    dino, device = load_dinov2(model_name="dinov2_vitb14")

    #dino change map
    change_map, overlay_q, overlay_r, score, q_rgb_518, r_rgb_518 = compute_change_map(
        query_path=query,
        ref_path=ref,
        model = dino,
        device = device,
        top_p=None,
        ms_pool_ks=(1,2,4),
        ms_pool_type="avg",
        ms_agg="top2mean",
    )

    sam.set_image(q_rgb_518)
    sam.set_image(r_rgb_518)
    
    #change map의 local peak들 중 최고봉 3개를 꼽아 roi를 만들어냄
    peaks = pick_local_maxima_topk(change_map,topk=3,k=3)
    rois = peaks_to_rois(change_map,peaks)
    print("num rois (DINO):", len(rois))

    # VLM 입력(pair) 만들기
    candidates = prepare_vlm_pairs_from_rois(
        q_rgb=q_rgb_518,
        r_rgb=r_rgb_518,
        rois=rois,
        expand_scale=1.6,
        img_size=518,
        out_h=336,
        border=4,
    )

    print("num candidates:", len(candidates))

    # vlm input vis
    visualize_vlm_pairs(candidates, max_show=10, cols=5)

    # 디버깅용 기존 시각화
    roi_vis = draw_rois_on_image(q_rgb_518, rois, color=(0,255,0), thickness=2)
    plt.figure(); plt.title("Query + ROI boxes"); plt.imshow(roi_vis); plt.axis("off")
    plt.figure(); plt.title("Change Map"); plt.imshow(change_map); plt.colorbar(); plt.axis("off")
    plt.figure(); plt.title("Query + heatmap"); plt.imshow(overlay_q); plt.axis("off")
    plt.show()



