"""

python -m vision_server.offline_eval --places 00

"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import load_cfg
from .distance import calibrate_place, infer_event
from .distance_util import (
    score_one_pair,
    build_flagged_component_regions,
    verify_bbox_with_local_search,
)
from .matcher import SuperGlueMatcher, SuperGlueMatchConfig
from .dino_emb import load_model as load_global_model
from .backbone_wrapper import build_local_backbone
from vpr_megaloc import MegaLocWrapper



RECV_ROOT = Path("./recv")
OUT_ROOT = Path("./recv/_offline_out")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# basic utils
# =========================================================
def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def safe_read_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def stem_without_frame_idx(path: Path) -> str:
    # ..._000.jpg 같은 프레임 suffix 제거
    stem = path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def label_from_name(path: Path) -> int:
    name = path.name.lower()
    return 1 if name.startswith("abnormal_") else 0


def build_query_events(query_dir: Path) -> List[Dict]:
    """
    query 폴더의 이미지들을 이벤트 단위로 묶는다.
    파일명 마지막 _000, _001, ... 를 같은 이벤트로 간주.
    """
    paths = list_images(query_dir)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        groups[stem_without_frame_idx(p)].append(p)

    events = []
    for key in sorted(groups.keys()):
        frames = sorted(groups[key])
        y_true = label_from_name(frames[0])
        events.append(
            {
                "event_key": key,
                "paths": frames,
                "label": y_true,
            }
        )
    return events


# =========================================================
# plotting
# =========================================================
def plot_calibration_curve(scores: List[float], thr: float, out_path: Path, title: str):
    plt.figure(figsize=(10, 4))
    xs = np.arange(len(scores))
    plt.plot(xs, scores, marker="o", markersize=3)
    plt.axhline(thr, linestyle="--")
    plt.title(title)
    plt.xlabel("sample idx")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_event_curve(frame_scores: List[float], thr: float, out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(frame_scores))
    plt.plot(xs, frame_scores, marker="o")
    plt.axhline(thr, linestyle="--")
    plt.title(title)
    plt.xlabel("frame idx")
    plt.ylabel("frame score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_metrics_report(y_true: List[int], y_pred: List[int], out_path: Path):
    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_pred, dtype=np.int32)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    acc = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    text = (
        f"Total={len(y_t)}\n"
        f"TP={tp} TN={tn} FP={fp} FN={fn}\n"
        f"Precision={precision:.4f}\n"
        f"Recall={recall:.4f}\n"
        f"F1={f1:.4f}\n"
        f"Accuracy={acc:.4f}\n"
    )
    out_path.write_text(text, encoding="utf-8")

    print("\n" + "=" * 40)
    print("[Evaluation Report]")
    print("=" * 40)
    print(text)
    print("=" * 40 + "\n")


# =========================================================
# visualization helpers (adapted from ex.py)
# =========================================================
def save_cc_heatmap(case_dir: Path, q_crop: np.ndarray, dist_map: np.ndarray, valid_mask: np.ndarray):
    h, w = q_crop.shape[:2]
    d = np.clip(dist_map, 0.0, 1.0)
    heat = (d * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    heat_rs = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_rs = cv2.resize(
        valid_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    overlay = q_crop.copy()
    blended = cv2.addWeighted(q_crop, 0.5, heat_rs, 0.5, 0)
    overlay[valid_rs] = blended[valid_rs]
    overlay[~valid_rs] = (128, 128, 128)

    cv2.imwrite(str(case_dir / "cc_heat.png"), heat_rs)
    cv2.imwrite(str(case_dir / "cc_overlay.png"), overlay)


def draw_text_box(img, lines, org=(10, 20), line_h=18):
    out = img.copy()
    x, y = org
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(out, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def colorize_patch_mask(mask_patch, out_h, out_w, color=(0, 0, 255), alpha=0.55, base=None):
    mask_u8 = mask_patch.astype(np.uint8)
    mask_rs = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    if base is None:
        base = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out = base.copy()

    color_img = np.zeros_like(out)
    color_img[:] = color
    out[mask_rs] = cv2.addWeighted(out, 1 - alpha, color_img, alpha, 0)[mask_rs]
    return out, mask_rs


def save_component_summary(case_dir: Path, q_crop: np.ndarray, bin_map: np.ndarray, best_comp_mask: np.ndarray, flagged_regions: list):
    h, w = q_crop.shape[:2]
    hot_all, _ = colorize_patch_mask(bin_map, h, w, color=(0, 165, 255), alpha=0.55, base=q_crop)
    best_only, _ = colorize_patch_mask(best_comp_mask, h, w, color=(0, 0, 255), alpha=0.60, base=q_crop)

    bbox_vis = q_crop.copy()
    for i, reg in enumerate(flagged_regions):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(bbox_vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        txt = f"#{i} s={reg['score']:.3f} a={reg['area']}"
        cv2.putText(bbox_vis, txt, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    panel = np.vstack([np.hstack([q_crop, hot_all]), np.hstack([best_only, bbox_vis])])
    panel = draw_text_box(
        panel,
        ["TL: q_crop", "TR: compound all hot-zone", "BL: best component", "BR: flagged component bbox"],
        org=(10, 20),
    )
    cv2.imwrite(str(case_dir / "stage3_component_summary.png"), panel)


def make_dist_overlay(base_bgr, dist_map, alpha=0.45, abs_min=0.0, abs_max=1.0):
    h, w = base_bgr.shape[:2]
    d = dist_map.astype(np.float32)
    if d.size == 0:
        return base_bgr.copy()

    d_vis = np.clip((d - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
    heat = (d_vis * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(base_bgr, 1 - alpha, heat, alpha, 0)


def save_flagged_region_visuals(case_dir: Path, verifier_results: list):
    for i, reg in enumerate(verifier_results):
        sub_dir = case_dir / f"bbox_{i:02d}"
        ensure_dir(sub_dir)

        q_region = reg["q_region"]
        r_region = reg["r_region"]
        vscore = reg["verifier_score"]
        dist_map = reg["verifier_dist_map"]
        top_p_mask = reg.get("verifier_top_p_mask", None)
        top_p_thr = reg.get("verifier_top_p_thr", None)
        top_k = reg.get("verifier_top_k", None)
        top_p = reg.get("verifier_top_p", None)

        cv2.imwrite(str(sub_dir / "q_region.png"), q_region)
        cv2.imwrite(str(sub_dir / "r_region.png"), r_region)

        q_overlay = make_dist_overlay(q_region, dist_map)
        r_overlay = make_dist_overlay(r_region, dist_map)

        q_overlay = draw_text_box(
            q_overlay,
            [f"bbox verifier score = {vscore:.4f}", f"img_bbox = {reg['img_bbox']}", f"patch_bbox = {reg['patch_bbox']}"],
            org=(8, 18),
        )
        r_overlay = draw_text_box(
            r_overlay,
            [f"component score = {reg['score']:.4f}", f"area = {reg['area']}", f"peak = {reg['peak']:.4f}"],
            org=(8, 18),
        )

        cv2.imwrite(str(sub_dir / "verifier_pair.png"), np.hstack([q_overlay, r_overlay]))

        d = dist_map.astype(np.float32)
        d_vis = np.clip(d, 0.0, 1.0)
        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(str(sub_dir / "verifier_heat.png"), heat)
        np.save(sub_dir / "verifier_dist_map.npy", dist_map)

        if top_p_mask is not None:
            h, w = q_region.shape[:2]
            top_mask_rs = cv2.resize(top_p_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            cv2.imwrite(str(sub_dir / "verifier_top_p_mask.png"), (top_mask_rs.astype(np.uint8) * 255))

            cyan = np.zeros_like(q_region)
            cyan[:] = (255, 255, 0)
            q_top = q_region.copy()
            r_top = r_region.copy()
            q_blend = cv2.addWeighted(q_region, 0.35, cyan, 0.65, 0)
            r_blend = cv2.addWeighted(r_region, 0.35, cyan, 0.65, 0)
            q_top[top_mask_rs] = q_blend[top_mask_rs]
            r_top[top_mask_rs] = r_blend[top_mask_rs]
            q_top = draw_text_box(
                q_top,
                [f"top_p = {top_p}", f"top_k = {top_k}", f"top_thr = {top_p_thr:.4f}" if top_p_thr is not None else "top_thr = NA"],
                org=(8, 18),
            )
            cv2.imwrite(str(sub_dir / "verifier_top_p_pair.png"), np.hstack([q_top, r_top]))


def save_verifier_summary(case_dir: Path, q_crop: np.ndarray, verifier_results: list):
    vis = q_crop.copy()
    for i, reg in enumerate(verifier_results):
        y0, x0, y1, x1 = reg["img_bbox"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
        txt = f"#{i} comp={reg['score']:.3f} ver={reg['verifier_score']:.3f}"
        cv2.putText(vis, txt, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(case_dir / "stage5_verifier_summary.png"), vis)


def save_verified_bbox_overlay(case_dir: Path, q_crop: np.ndarray, verified_regions: list, alpha=0.45, abs_min=0.0, abs_max=1.0, vis_thr_abs=0.30):
    vis = q_crop.copy()

    for reg in verified_regions:
        y0, x0, y1, x1 = reg["img_bbox"]
        dist_map = reg["verifier_dist_map"].astype(np.float32)

        if dist_map.size == 0 or (y1 - y0) <= 0 or (x1 - x0) <= 0:
            continue

        d_vis = np.clip((dist_map - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
        thr_norm = np.clip((vis_thr_abs - abs_min) / max(abs_max - abs_min, 1e-8), 0.0, 1.0)
        mask = d_vis >= thr_norm
        if mask.sum() == 0:
            continue

        heat = (d_vis * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        mask_rs = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)

        roi = vis[y0:y1, x0:x1].copy()
        blended = cv2.addWeighted(roi, 1 - alpha, heat, alpha, 0)
        roi[mask_rs] = blended[mask_rs]
        vis[y0:y1, x0:x1] = roi

        cv2.putText(vis, f"{reg['verifier_score']:.3f}", (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(case_dir / "stage6_verified_bbox_overlay_abs.png"), vis)


# =========================================================
# engine
# =========================================================
def build_engine(cfg, recv_root, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global_model, device = load_global_model(device=device)

    local_model = build_local_backbone(
        backbone_name="resnet18_layer3",
        img_size=int(cfg.get("embed", {}).get("img_size", 560)),
    )

    vpr_model = MegaLocWrapper(device=device)

    sg_matcher = SuperGlueMatcher(...)

    return {
        "bank_root": str(recv_root),
        "global_model": global_model,
        "local_model": local_model,
        "device": device,
        "sg_matcher": sg_matcher,
        "vpr_model": vpr_model, 
    }


# =========================================================
# visualization from representative frame
# =========================================================
def make_case_visualizations(
    case_dir: Path,
    cfg: dict,
    engine: dict,
    query_paths: List[Path],
    infer_out: dict,
):
    """
    대표 프레임 1장과 best ref 1장을 가지고
    ex.py 핵심 시각화(CC heatmap / verifier heatmap)를 다시 계산해 저장.
    """
    if len(query_paths) == 0:
        return

    rep_idx = int(infer_out.get("patch_vis", {}).get("frame_idx", 0))
    rep_idx = max(0, min(rep_idx, len(query_paths) - 1))
    q_path = query_paths[rep_idx]
    q_bgr = safe_read_bgr(q_path)
    if q_bgr is None:
        return

    topk_json = json.loads(infer_out.get("ref_topk_json", "{}"))
    topk_paths_all = topk_json.get("topk_paths", [])
    if rep_idx >= len(topk_paths_all) or len(topk_paths_all[rep_idx]) == 0:
        return

    best_ref_path = Path(topk_paths_all[rep_idx][0])
    r_bgr = safe_read_bgr(best_ref_path)
    if r_bgr is None:
        return

    pcfg = cfg.get("patchcore", {})
    top_p = float(pcfg.get("top_p", 0.05))
    alpha = float(pcfg.get("alpha", 0.6))
    min_cut = float(pcfg.get("min_cut", 0.20))
    singleton_weight = float(pcfg.get("singleton_weight", 0.25))
    component_min_area = int(pcfg.get("component_min_area", 2))

    mode_cfg = cfg.get("repr_modes", {}).get("global_patch_with_aligned", {})
    proposal_cfg = mode_cfg.get("proposal", {})
    ver_cfg = mode_cfg.get("verifier", {})

    proposal_top_k = int(proposal_cfg.get("top_k", 3))
    patch_margin = int(proposal_cfg.get("patch_margin", 1))
    crop_margin_ratio = float(proposal_cfg.get("crop_margin_ratio", 0.20))
    min_patch_area = int(proposal_cfg.get("min_patch_area", 2))
    min_crop_size = int(proposal_cfg.get("min_crop_size", 96))
    ver_radius = int(ver_cfg.get("radius", 1))
    ver_top_p = float(ver_cfg.get("top_p", 0.10))

    score, debug = score_one_pair(
        q_bgr=q_bgr,
        r_bgr=r_bgr,
        sg=engine["sg_matcher"],
        backbone=engine["local_model"],
        device=engine["device"],
        radius=int(pcfg.get("radius", 1)),
        top_p=top_p,
        alpha=alpha,
        min_cut=min_cut,
        singleton_weight=singleton_weight,
        component_min_area=component_min_area,
    )
    if score is None:
        return

    cv2.imwrite(str(case_dir / "rep_query.png"), q_bgr)
    cv2.imwrite(str(case_dir / "rep_ref.png"), r_bgr)
    cv2.imwrite(str(case_dir / "q_crop.png"), debug["q_crop"])
    cv2.imwrite(str(case_dir / "r_crop.png"), debug["r_crop"])

    save_cc_heatmap(case_dir, debug["q_crop"], debug["dist_map"], debug["valid_mask"])

    all_comps = debug.get("all_comp_scores", [])
    topk_comps = sorted(all_comps, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:proposal_top_k]

    flagged_regions = build_flagged_component_regions(
        flagged_comps=topk_comps,
        q_crop=debug["q_crop"],
        r_crop=debug["r_crop"],
        valid_mask=debug["valid_mask"],
        patch_margin=patch_margin,
        crop_margin_ratio=crop_margin_ratio,
        min_patch_area=min_patch_area,
        min_crop_size=min_crop_size,
    )

    verifier_results = []
    for reg in flagged_regions:
        out = verify_bbox_with_local_search(
            q_region=reg["q_region"],
            r_region=reg["r_region"],
            backbone=engine["local_model"],
            device=engine["device"],
            radius=ver_radius,
            top_p=ver_top_p,
        )
        verifier_results.append({
            **reg,
            "verifier_score": float(out["score"]),
            "verifier_dist_map": out["dist_map"],
            "verifier_feat_hw": out["feat_hw"],
            "verifier_top_p_mask": out["top_p_mask"],
            "verifier_top_p_thr": out["top_p_thr"],
            "verifier_top_k": out["top_k"],
            "verifier_top_p": out["top_p"],
        })

    save_component_summary(case_dir, debug["q_crop"], debug["bin_map"], debug["best_comp_mask"], flagged_regions)
    save_flagged_region_visuals(case_dir, verifier_results)
    save_verifier_summary(case_dir, debug["q_crop"], verifier_results)

    thr = float(infer_out["threshold"])
    verified_regions = [r for r in verifier_results if r["verifier_score"] > thr]
    save_verified_bbox_overlay(case_dir, debug["q_crop"], verified_regions, alpha=0.45)

    meta = {
        "rep_idx": rep_idx,
        "query_path": str(q_path),
        "best_ref_path": str(best_ref_path),
        "cc_score": float(score),
        "threshold": float(thr),
        "num_flagged_regions": len(flagged_regions),
        "num_verified_regions": len(verified_regions),
    }
    dump_json(case_dir / "viz_meta.json", meta)


# =========================================================
# per place evaluation
# =========================================================
def evaluate_place(recv_root: Path, out_root: Path, plc: str, engine: dict, cfg: dict):
    place_root = recv_root / plc
    bank_dir = place_root / "bank"
    th_dir = place_root / "th_calib"
    query_dir = place_root / "query"

    if not bank_dir.exists() or not th_dir.exists() or not query_dir.exists():
        print(f"[SKIP] place={plc}: bank/th_calib/query 중 하나가 없음")
        return

    out_dir = out_root / plc
    ensure_dir(out_dir)

    # 1) calibration
    thr, calib_scores, _ = calibrate_place(
        str(recv_root),
        plc,
        engine["global_model"],
        engine["local_model"],   # cc_backbone
        engine["local_model"],   # verifier_backbone
        engine["device"],
        sg_matcher=engine.get("sg_matcher"),
        cfg=cfg,
    )

    plot_calibration_curve(
        [float(x) for x in calib_scores],
        float(thr),
        out_dir / "calib_scores_curve.png",
        title=f"[CALIB] place={plc}",
    )

    # 2) query events
    events = build_query_events(query_dir)
    if len(events) == 0:
        print(f"[SKIP] place={plc}: query event 없음")
        return

    all_results = []
    y_true, y_pred = [], []

    for ev_i, ev in enumerate(events):
        case_dir = out_dir / ev["event_key"]
        ensure_dir(case_dir)

        imgs_bgr = []
        valid_paths = []
        for p in ev["paths"]:
            img = safe_read_bgr(p)
            if img is None:
                continue
            imgs_bgr.append(img)
            valid_paths.append(p)

        if len(imgs_bgr) == 0:
            continue

        out = infer_event(
            imgs_bgr=imgs_bgr,
            bank_root=engine["bank_root"],
            plc_idx=plc,
            cfg=cfg,
            global_model=engine["global_model"],
            cc_backbone=engine["local_model"],
            verifier_backbone=engine["local_model"],
            device=engine["device"],
            sg_matcher=engine.get("sg_matcher"),
        )

        plot_event_curve(
            out["frame_scores"],
            out["threshold"],
            case_dir / "event_frame_scores.png",
            title=f"[EVENT] {ev['event_key']}",
        )

        make_case_visualizations(
            case_dir=case_dir,
            cfg=cfg,
            engine=engine,
            query_paths=valid_paths,
            infer_out=out,
        )

        result = {
            "event_key": ev["event_key"],
            "label": int(ev["label"]),
            "pred": int(out["anomaly_flag"]),
            "threshold": float(out["threshold"]),
            "frame_scores": [float(x) for x in out["frame_scores"]],
            "frame_change_flags": [int(x) for x in out["frame_change_flags"]],
            "event_score": float(out["event_score"]),
            "summary": out.get("summary", ""),
            "query_paths": [str(p) for p in valid_paths],
            "ref_topk_json": json.loads(out.get("ref_topk_json", "{}")),
        }
        dump_json(case_dir / "result.json", result)

        all_results.append(result)
        y_true.append(int(ev["label"]))
        y_pred.append(int(out["anomaly_flag"]))

        label_str = "ANOMALY" if result["pred"] == 1 else "NORMAL"
        gt_str = "abnormal" if ev["label"] == 1 else "normal"
        print(
            f"[EVENT] place={plc} {ev_i+1}/{len(events)} "
            f"gt={gt_str} pred={label_str} "
            f"score={result['event_score']:.2f}"
        )

    dump_json(out_dir / "offline_eval_results.json", all_results)
    save_metrics_report(y_true, y_pred, out_dir / "metrics.txt")


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recv_root", type=str, default=None)
    parser.add_argument("--out_root", type=str, default=None)
    parser.add_argument("--places", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    recv_root = Path(args.recv_root) if args.recv_root is not None else RECV_ROOT
    out_root = Path(args.out_root) if args.out_root is not None else OUT_ROOT
    ensure_dir(out_root)

    cfg = load_cfg(recv_root)
    engine = build_engine(cfg, recv_root)
    cfg["vpr_model"] = engine["vpr_model"] 

    if args.places is not None and len(args.places) > 0:
        places = [str(x) for x in args.places]
    else:
        places = sorted([p.name for p in recv_root.iterdir() if p.is_dir() and p.name.isdigit()])

    print(f"[INFO] recv_root={recv_root}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] places={places}")

    for plc in places:
        print("\n" + "=" * 60)
        print(f"[PLACE] {plc}")
        print("=" * 60)
        try:
            evaluate_place(recv_root, out_root, plc, engine, cfg)
        except Exception as e:
            print(f"[ERROR] place={plc}: {e}")

    print("\n✅ offline_eval done")


if __name__ == "__main__":
    main()