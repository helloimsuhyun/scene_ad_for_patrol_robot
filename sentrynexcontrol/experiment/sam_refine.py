# sam_refine.py
# ZeroSCD 아이디어를 반영한 최소 수정형 SAM refinement
# dist_map(coarse) -> blob 추출 -> blob과 가장 잘 겹치는 SAM mask 선택
# -> Query/Ref SAM mask IoU 검증 -> XOR를 blob 주변으로 제한하여 최종 change mask 생성

import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor


# ─────────────────────────────────────────────────────────────────────────────
# SAM 모델 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    device: str = "cuda",
):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Adaptive threshold 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_threshold(
    dist_map: np.ndarray,
    top_p: float = 0.10,
    abs_floor: float = 0.20,
    peak_alpha: float = 0.50,
) -> float:
    """
    threshold = max(percentile(dist, (1-top_p)*100), max(abs_floor, peak_alpha*peak))
    """
    if dist_map.size == 0:
        return abs_floor
    flat = dist_map.flatten().astype(np.float32)
    top_p_val = float(np.percentile(flat, (1.0 - top_p) * 100.0))
    peak_floor = max(abs_floor, peak_alpha * float(flat.max()))
    return max(top_p_val, peak_floor)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Filtered dist map 생성
# ─────────────────────────────────────────────────────────────────────────────

def _build_filtered_map(
    dist_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    fmap = dist_map.astype(np.float32).copy()
    fmap[fmap < threshold] = 0.0
    return fmap


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _blob_peak_xy(blob_mask_img: np.ndarray, dist_map: np.ndarray, img_hw: tuple) -> tuple:
    """
    blob 내부에서 dist peak 위치를 이미지 좌표로 변환
    """
    H_img, W_img = img_hw
    Hf, Wf = dist_map.shape

    blob_feat = cv2.resize(
        blob_mask_img.astype(np.uint8),
        (Wf, Hf),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    vals = dist_map.copy()
    vals[~blob_feat] = -1.0
    idx = int(np.argmax(vals))
    y_f, x_f = np.unravel_index(idx, vals.shape)

    x = int((x_f + 0.5) * W_img / Wf)
    y = int((y_f + 0.5) * H_img / Hf)
    x = max(0, min(W_img - 1, x))
    y = max(0, min(H_img - 1, y))
    return x, y


def _select_best_sam_mask(
    predictor,
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    coarse_blob_mask: np.ndarray,
    peak_xy: tuple,
    min_overlap_blob: float = 0.35,
):
    """
    bbox 내부에서 SAM multimask 후보를 뽑고,
    coarse blob과 가장 잘 겹치는 mask만 선택한다.
    """
    predictor.set_image(image_rgb)

    box = bbox_xyxy.astype(np.float32)

    # 1) point + box prompt
    point_coords = np.array([[peak_xy[0], peak_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box[None, :],
        multimask_output=True,
    )

    # 2) 혹시 위가 불안정하면 box-only도 같이 후보에 포함
    masks2, scores2, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True,
    )

    all_masks = []
    all_scores = []

    if masks is not None and len(masks) > 0:
        for m, s in zip(masks, scores):
            all_masks.append(m.astype(bool))
            all_scores.append(float(s))

    if masks2 is not None and len(masks2) > 0:
        for m, s in zip(masks2, scores2):
            all_masks.append(m.astype(bool))
            all_scores.append(float(s))

    if len(all_masks) == 0:
        return None, {
            "sam_ok": False,
            "reason": "sam_no_mask",
            "blob_overlap_best": 0.0,
            "blob_cover_best": 0.0,
            "sam_score_best": 0.0,
        }

    best_idx = -1
    best_key = (-1.0, -1.0, -1.0)

    blob_area = int(coarse_blob_mask.sum())

    for i, (m, s) in enumerate(zip(all_masks, all_scores)):
        m = m.astype(bool)
        m_area = int(m.sum())
        inter = int(np.logical_and(m, coarse_blob_mask).sum())

        # SAM mask 중 얼마나 coarse blob에 집중되어 있는가
        blob_overlap = _safe_div(inter, m_area)      # precision 성격
        blob_cover = _safe_div(inter, blob_area)     # recall 성격

        # overlap 우선, cover 다음, SAM score 다음
        key = (blob_overlap, blob_cover, s)
        if key > best_key:
            best_key = key
            best_idx = i

    best_mask = all_masks[best_idx].astype(bool)
    best_score = float(all_scores[best_idx])

    inter = int(np.logical_and(best_mask, coarse_blob_mask).sum())
    best_overlap = _safe_div(inter, int(best_mask.sum()))
    best_cover = _safe_div(inter, blob_area)

    if best_overlap < min_overlap_blob and best_cover < min_overlap_blob:
        return None, {
            "sam_ok": False,
            "reason": "low_blob_overlap",
            "blob_overlap_best": best_overlap,
            "blob_cover_best": best_cover,
            "sam_score_best": best_score,
        }

    return best_mask, {
        "sam_ok": True,
        "reason": "ok",
        "blob_overlap_best": best_overlap,
        "blob_cover_best": best_cover,
        "sam_score_best": best_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Connected Component(blob) 추출
# ─────────────────────────────────────────────────────────────────────────────

def _extract_blobs(
    filtered_map: np.ndarray,
    img_hw: tuple,
    min_blob_area_ratio: float = 0.002,
    singleton_weight: float = 0.25,
) -> list:
    Hf, Wf = filtered_map.shape
    H_img, W_img = img_hw

    binary = (filtered_map > 0).astype(np.uint8)

    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    total_patches = Hf * Wf
    min_area = max(1, int(total_patches * min_blob_area_ratio))

    blobs = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        comp_mask_feat = (label_img == label_id)

        vals = filtered_map[comp_mask_feat]
        peak = float(vals.max())
        mean = float(vals.mean())

        if area >= min_area:
            score = float(vals.sum())
        else:
            score = float(singleton_weight * peak)

        comp_mask_img = cv2.resize(
            comp_mask_feat.astype(np.uint8),
            (W_img, H_img),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        y_coords, x_coords = np.where(comp_mask_img)
        if len(y_coords) == 0:
            continue

        # margin 너무 크면 뒤 물체까지 끌고 오므로 축소
        margin = 2
        x1 = max(0, int(x_coords.min()) - margin)
        y1 = max(0, int(y_coords.min()) - margin)
        x2 = min(W_img - 1, int(x_coords.max()) + margin)
        y2 = min(H_img - 1, int(y_coords.max()) + margin)

        blobs.append({
            "mask": comp_mask_img,
            "bbox": np.array([x1, y1, x2, y2], dtype=np.int32),
            "area_patch": area,
            "peak": peak,
            "mean": mean,
            "score": score,
        })

    blobs.sort(key=lambda b: b["score"], reverse=True)
    return blobs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: ZeroSCD 스타일 최소 수정형 SAM refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_with_sam(
    q_crop_rgb: np.ndarray,
    r_crop_rgb: np.ndarray,
    dist_map: np.ndarray,
    predictor,
    top_p: float = 0.10,
    abs_floor: float = 0.20,
    peak_alpha: float = 0.50,
    min_blob_peak: float = 0.30,
    min_blob_mean: float = 0.25,
    dv_iou_thresh: float = 0.60,
    min_blob_area_ratio: float = 0.005,
    min_overlap_blob: float = 0.35,
    xor_restrict_dilate: int = 5,
    # --- DINO crop 검증 추가 ---
    dino_model=None,
    dino_tfm=None,
    device: str = "cuda",
    dino_sim_reject_thresh: float = 0.85,
    dino_sim_bypass_thresh: float = 0.65,
    refine_mode: str = "hybrid",
) -> dict:
    """
    현재 파이프라인 인터페이스는 그대로 두고,
    ZeroSCD의 "coarse overlap -> segment flag -> opposite overlap verification"
    로직을 최대한 비슷하게 반영한 버전.
    """
    H_img, W_img = q_crop_rgb.shape[:2]

    threshold = compute_adaptive_threshold(
        dist_map, top_p=top_p, abs_floor=abs_floor, peak_alpha=peak_alpha
    )

    fmap = _build_filtered_map(dist_map, threshold)
    blobs = _extract_blobs(
        fmap,
        img_hw=(H_img, W_img),
        min_blob_area_ratio=min_blob_area_ratio,
    )

    if not blobs:
        return {
            "change_mask": np.zeros((H_img, W_img), dtype=bool),
            "filtered_map": fmap,
            "blobs": [],
            "num_blobs": 0,
            "num_confirmed_blobs": 0,
            "adaptive_threshold": threshold,
        }

    change_mask = np.zeros((H_img, W_img), dtype=bool)
    num_confirmed = 0

    for i, blob in enumerate(blobs):
        peak = float(blob["peak"])
        mean = float(blob["mean"])

        # coarse blob quality gate
        if peak < min_blob_peak or mean < min_blob_mean:
            blobs[i]["status"] = "rejected_low_blob_score"
            blobs[i]["q_blob_overlap"] = 0.0
            blobs[i]["r_blob_overlap"] = 0.0
            blobs[i]["qr_iou"] = 1.0
            continue

        bbox = blob["bbox"]
        coarse_blob = blob["mask"].astype(bool)
        peak_xy = _blob_peak_xy(coarse_blob, dist_map, (H_img, W_img))

        # [추가] DINOv2 Crop-Level Semantic 검증
        dino_sim = None
        is_hard_anomaly = False # DINO가 다른 물체라고 확신하면 True
        if dino_model is not None and dino_tfm is not None and refine_mode in ["dino", "hybrid"]:
            import torch
            from PIL import Image
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # crop이 너무 작으면 억지 검증 생략
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                cr_q = q_crop_rgb[y1:y2, x1:x2]
                cr_r = r_crop_rgb[y1:y2, x1:x2]
                try:
                    t_q = dino_tfm(Image.fromarray(cr_q)).unsqueeze(0).to(device)
                    t_r = dino_tfm(Image.fromarray(cr_r)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        import torch.nn.functional as F
                        
                        # CLS token 사용
                        f_q = dino_model.get_intermediate_layers(
                            t_q, n=1, return_class_token=True
                        )[0][1].squeeze(0).squeeze(0)


                        f_r = dino_model.get_intermediate_layers(
                            t_r, n=1, return_class_token=True
                        )[0][1].squeeze(0).squeeze(0)

                        f_q = F.normalize(f_q, dim=0)
                        f_r = F.normalize(f_r, dim=0)

                        dino_sim = float((f_q * f_r).sum().item())
                        
                    blobs[i]["dino_sim"] = dino_sim
                    
                    if refine_mode == "dino":
                        # DINO 온리 모드일 때 SAM은 전혀 실행하지 않고 바로 상태 결정
                        if dino_sim > dino_sim_reject_thresh:
                            blobs[i]["status"] = f"rejected_dino_sim_high_{dino_sim:.2f}"
                        else:
                            blobs[i]["status"] = "confirmed"
                            blobs[i]["final_area"] = int(coarse_blob.sum())
                            change_mask[coarse_blob] = True
                            num_confirmed += 1
                        blobs[i]["q_blob_overlap"] = 0.0
                        blobs[i]["r_blob_overlap"] = 0.0
                        blobs[i]["qr_iou"] = 1.0
                        continue
                    
                    # hybrid 모드일 때 
                    elif refine_mode == "hybrid":
                        if dino_sim > dino_sim_reject_thresh:
                            blobs[i]["status"] = f"rejected_dino_sim_high_{dino_sim:.2f}"
                            blobs[i]["q_blob_overlap"] = 0.0
                            blobs[i]["r_blob_overlap"] = 0.0
                            blobs[i]["qr_iou"] = 1.0
                            continue
                        if dino_sim < dino_sim_bypass_thresh:
                            is_hard_anomaly = True
                except Exception as e:
                    print(f"[DINO WARN] feature extraction exception: {e}")
                    pass

        # DINO 단독 모드일 경우 여기서 루프 종료 (SAM 접근 방지)
        if refine_mode == "dino":
            if "status" not in blobs[i]:
                blobs[i]["status"] = "dino_error"
                blobs[i]["q_blob_overlap"] = 0.0
                blobs[i]["r_blob_overlap"] = 0.0
                blobs[i]["qr_iou"] = 1.0
            continue

        # Query SAM
        q_mask, q_meta = _select_best_sam_mask(
            predictor=predictor,
            image_rgb=q_crop_rgb,
            bbox_xyxy=bbox,
            coarse_blob_mask=coarse_blob,
            peak_xy=peak_xy,
            min_overlap_blob=min_overlap_blob,
        )

        if q_mask is None:
            blobs[i]["status"] = f"rejected_q_{q_meta['reason']}"
            blobs[i]["q_blob_overlap"] = float(q_meta["blob_overlap_best"])
            blobs[i]["r_blob_overlap"] = 0.0
            blobs[i]["qr_iou"] = 1.0
            continue

        # Ref SAM
        r_mask, r_meta = _select_best_sam_mask(
            predictor=predictor,
            image_rgb=r_crop_rgb,
            bbox_xyxy=bbox,
            coarse_blob_mask=coarse_blob,
            peak_xy=peak_xy,
            min_overlap_blob=min_overlap_blob,
        )

        if r_mask is None:
            blobs[i]["status"] = f"rejected_r_{r_meta['reason']}"
            blobs[i]["q_blob_overlap"] = float(q_meta["blob_overlap_best"])
            blobs[i]["r_blob_overlap"] = float(r_meta["blob_overlap_best"])
            blobs[i]["qr_iou"] = 1.0
            continue

        # ZeroSCD의 2차 검증에 해당:
        # 대응 segment끼리 너무 비슷하면 change 아님
        qr_iou = _mask_iou(q_mask, r_mask)

        blobs[i]["q_blob_overlap"] = float(q_meta["blob_overlap_best"])
        blobs[i]["r_blob_overlap"] = float(r_meta["blob_overlap_best"])
        blobs[i]["q_blob_cover"] = float(q_meta["blob_cover_best"])
        blobs[i]["r_blob_cover"] = float(r_meta["blob_cover_best"])
        blobs[i]["q_sam_score"] = float(q_meta["sam_score_best"])
        blobs[i]["r_sam_score"] = float(r_meta["sam_score_best"])
        blobs[i]["qr_iou"] = float(qr_iou)
        blobs[i]["q_mask"] = q_mask
        blobs[i]["r_mask"] = r_mask

        # ZeroSCD의 2차 검증(IoU) 우회: DINO가 확신한 질감/의미 변화일 경우 IoU 무시
        if qr_iou >= dv_iou_thresh and not is_hard_anomaly:
            blobs[i]["status"] = "rejected_high_iou"
            continue

        # change region은 XOR이 맞지만,
        # coarse blob 주변으로 다시 제한해서 뒤 물체 leakage 방지
        xor_region = np.logical_xor(q_mask.astype(bool), r_mask.astype(bool))

        if xor_restrict_dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (xor_restrict_dilate, xor_restrict_dilate),
            )
            coarse_support = cv2.dilate(
                coarse_blob.astype(np.uint8),
                kernel,
                iterations=1,
            ).astype(bool)
        else:
            coarse_support = coarse_blob

        xor_region = np.logical_and(xor_region, coarse_support)

        if xor_region.sum() == 0:
            blobs[i]["status"] = "rejected_empty_xor_after_restrict"
            continue

        change_mask[xor_region] = True
        blobs[i]["status"] = "confirmed"
        blobs[i]["final_area"] = int(xor_region.sum())
        num_confirmed += 1

    return {
        "change_mask": change_mask,
        "filtered_map": fmap,
        "blobs": blobs,
        "num_blobs": len(blobs),
        "num_confirmed_blobs": num_confirmed,
        "adaptive_threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_sam_outputs(
    save_dir: Path,
    q_crop_rgb: np.ndarray,
    r_crop_rgb: np.ndarray,
    result: dict,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    change_mask = result["change_mask"]
    fmap = result["filtered_map"]
    blobs = result["blobs"]

    q_bgr = cv2.cvtColor(q_crop_rgb, cv2.COLOR_RGB2BGR)
    r_bgr = cv2.cvtColor(r_crop_rgb, cv2.COLOR_RGB2BGR)
    H, W = q_bgr.shape[:2]

    # 1) binary mask
    mask_u8 = (change_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(save_dir / "sam_change_mask.png"), mask_u8)

    # 2) filtered map
    fmap_rs = cv2.resize(fmap, (W, H), interpolation=cv2.INTER_NEAREST)
    if fmap_rs.max() > 0:
        fmap_vis = np.clip(fmap_rs / fmap_rs.max(), 0, 1)
    else:
        fmap_vis = fmap_rs
    fmap_jet = cv2.applyColorMap((fmap_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(save_dir / "sam_filtered_map.png"), fmap_jet)

    # 3) blob / status overlay
    blob_vis = q_bgr.copy()
    for i, blob in enumerate(blobs):
        status = blob.get("status", "unknown")
        x1, y1, x2, y2 = [int(v) for v in blob["bbox"]]

        if status == "confirmed":
            color = (0, 255, 255)   # yellow
        else:
            color = (0, 0, 255)     # red

        cv2.rectangle(blob_vis, (x1, y1), (x2, y2), color, 2)

        d_sim = blob.get("dino_sim", -1.0)
        if d_sim >= 0:
            txt = f"{i}:{status} (DINO:{d_sim:.2f})"
        else:
            txt = f"{i}:{status}"
            
        cv2.putText(
            blob_vis, txt, (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )

    cv2.imwrite(str(save_dir / "sam_blob_status.png"), blob_vis)

    # 4) q overlay
    overlay_q = q_bgr.copy()
    overlay_q[change_mask] = cv2.addWeighted(
        overlay_q, 0.35, np.full_like(overlay_q, (0, 0, 255)), 0.65, 0
    )[change_mask]
    cv2.imwrite(str(save_dir / "sam_overlay_q.png"), overlay_q)

    # 5) r overlay
    overlay_r = r_bgr.copy()
    overlay_r[change_mask] = cv2.addWeighted(
        overlay_r, 0.35, np.full_like(overlay_r, (0, 0, 255)), 0.65, 0
    )[change_mask]
    cv2.imwrite(str(save_dir / "sam_overlay_r.png"), overlay_r)

    # 6) compare
    divider = np.ones((H, 3, 3), dtype=np.uint8) * 200
    compare = np.hstack([overlay_q, divider, overlay_r])
    cv2.imwrite(str(save_dir / "sam_compare.png"), compare)

    # 7) DINO 오버레이 덤프 (전체 Query 이미지 위 박스)
    has_dino = any("dino_sim" in b for b in blobs)
    if has_dino and len(blobs) > 0:
        dino_q_vis = q_bgr.copy()
        for i, blob in enumerate(blobs):
            if "dino_sim" not in blob:
                continue
            x1, y1, x2, y2 = [int(v) for v in blob["bbox"]]
            status = blob.get("status", "unknown")
            color = (0, 255, 255) if status == "confirmed" else (0, 0, 255)
            
            # 박스 그리기
            cv2.rectangle(dino_q_vis, (x1, y1), (x2, y2), color, 3)
            
            d_sim = blob["dino_sim"]
            text = f"[{i}] Sim: {d_sim:.3f} | {status}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # 가독성을 위한 텍스트 배경
            cv2.rectangle(dino_q_vis, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(dino_q_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
        cv2.imwrite(str(save_dir / "dino_overlay.png"), dino_q_vis)