# offline_eval.py
# ref bank 폴더트리 완성되어있는 상황에서
# event 단위 추론/평가 + 서버와 동일한 DB 저장까지 수행하는 스크립트

from __future__ import annotations

import re
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from . import sqlite_db
from . import place_manager
from . import dino_emb
from . import cnn_emb


from .distance import infer_event, calibrate_place
from .config import load_cfg
from .matcher import SuperGlueMatcher, SuperGlueMatchConfig
from .vis import draw_top_p_heatmap, save_aligned_debug_vis, save_patch_match_vis

# =========================
# 설정
# =========================
THIS_DIR = Path(__file__).resolve().parent
RECV_ROOT = THIS_DIR.parent / "recv"

DB_PATH = RECV_ROOT / "events.db"
OUT_ROOTNAME = "_offline_out"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

USE_SAME_DB = True
OFFLINE_DB_PATH = RECV_ROOT / "offline_eval.db"


# =========================
# batch 구조
# =========================
@dataclass
class Batch:
    place_id: str
    safe_ts: str
    label: Optional[str]
    paths: List[Path]


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def normalize_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"normal", "nomal"}:
        return "normal"
    if s in {"abnormal", "unnomal", "unnormal", "anomaly", "anomal"}:
        return "abnormal"
    return s


# =========================
# DB helper
# =========================
def get_db_path() -> Path:
    return DB_PATH if USE_SAME_DB else OFFLINE_DB_PATH


def sync_places_from_fs(db, save_root: Path):
    for p in save_root.iterdir():
        if not p.is_dir():
            continue
        sqlite_db.ensure_place(db, p.name)


# =========================
# model-view helper
# =========================
def load_pil_for_model_view(img_path: Path, img_size: int) -> Image.Image:
    img_pil = Image.open(img_path).convert("RGB")
    img_pil = transforms.Resize(img_size)(img_pil)
    img_pil = transforms.CenterCrop(img_size)(img_pil)
    return img_pil


def load_bgr_for_model_view(img_path: Path, img_size: int) -> np.ndarray:
    img_pil = load_pil_for_model_view(img_path, img_size=img_size)
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def save_loss_map_vis(
    query_path: Path,
    loss_vis: Dict[str, Any],
    out_path: Path,
    img_size: int = 560,
):
    aggr_loss = loss_vis.get("aggr_loss_map", None)
    aggr_valid = loss_vis.get("aggr_valid_map", None)

    if aggr_loss is None:
        return

    # query image 로드
    q_img = load_bgr_for_model_view(query_path, img_size=img_size)

    # 크기 맞추기
    if aggr_loss.shape[:2] != q_img.shape[:2]:
        aggr_loss = cv2.resize(
            aggr_loss,
            (q_img.shape[1], q_img.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if aggr_valid is not None and aggr_valid.shape[:2] != q_img.shape[:2]:
        aggr_valid = cv2.resize(
            aggr_valid,
            (q_img.shape[1], q_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # uint8 보장
    if aggr_loss.dtype != np.uint8:
        aggr_loss = np.clip(aggr_loss, 0, 255).astype(np.uint8)

    # loss map 단독 시각화
    loss_color = cv2.applyColorMap(aggr_loss, cv2.COLORMAP_JET)

    # valid mask 밖은 검정 처리
    if aggr_valid is not None:
        vm = (aggr_valid > 0)
        loss_color = loss_color.copy()
        loss_color[~vm] = 0

    # 제목 바
    bar_h = 36
    H, W = q_img.shape[:2]

    def add_title(img, title):
        canvas = np.full((H + bar_h, W, 3), 255, dtype=np.uint8)
        canvas[bar_h:] = img
        cv2.putText(
            canvas,
            title,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    panel_loss = add_title(loss_color, "loss map")
    panel_query = add_title(q_img, "query")

    vis = np.concatenate([panel_loss, panel_query], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def parse_server_filename(
    p: Path
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
    stem = p.stem

    m = re.match(
        r"^(?P<label>[^_]+)_(?P<place>[^_]+)_(?P<mode>bank|th_calib|query)_(?P<ts>.+)_(?P<idx>\d+)$",
        stem
    )
    if m:
        return (
            m.group("label"),
            m.group("place"),
            m.group("mode"),
            m.group("ts"),
            int(m.group("idx")),
        )

    m = re.match(
        r"^(?P<place>[^_]+)_(?P<mode>bank|th_calib|query)_(?P<ts>.+)_(?P<idx>\d+)$",
        stem
    )
    if m:
        return (
            None,
            m.group("place"),
            m.group("mode"),
            m.group("ts"),
            int(m.group("idx")),
        )

    return None, None, None, None, None


def collect_batches_for_place(place_dir: Path, mode: str) -> List[Batch]:
    assert mode in {"query", "th_calib", "bank"}

    target_dir = place_dir / mode
    if not target_dir.exists():
        return []

    groups: Dict[Tuple[str, Optional[str]], List[Tuple[int, Path]]] = {}
    place_id = place_dir.name

    for p in target_dir.iterdir():
        if not p.is_file() or not is_image(p):
            continue
        label, plc, md, safe_ts, idx = parse_server_filename(p)
        if plc != place_id or md != mode or safe_ts is None or idx is None:
            continue
        label = normalize_label(label)
        key = (safe_ts, label)
        groups.setdefault(key, []).append((idx, p))

    out: List[Batch] = []
    for (safe_ts, label), items in groups.items():
        items = sorted(items, key=lambda x: x[0])
        out.append(
            Batch(
                place_id=place_id,
                safe_ts=safe_ts,
                label=label,
                paths=[p for _, p in items],
            )
        )

    out.sort(key=lambda b: b.safe_ts)
    return out


def load_imgs_bgr(paths: List[Path]) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        imgs.append(img)
    return imgs


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))

    n = max(1, len(y_true))
    acc = (tp + tn) / n
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "N_total": float(len(y_true)),
        "N_pos": float(sum(y_true)),
        "N_neg": float(len(y_true) - sum(y_true)),
    }

def load_bgr_for_cnn_view(img_bgr, img_size=560):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img_pil = transforms.Resize((img_size, img_size))(img_pil)

    img = np.array(img_pil)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_pair_vis_k(
    topk_paths: List[Path],
    query_path: Path,
    out_path: Path,
    score: float,
    topk_sims: List[float],
    is_change: bool,
    vis_size: int = 384,
    model_view_img_size: int = 560,
):
    k = len(topk_paths)

    ref_imgs = [
        load_pil_for_model_view(p, img_size=model_view_img_size).resize(
            (vis_size, vis_size), Image.BICUBIC
        )
        for p in topk_paths
    ]
    qry_img = load_pil_for_model_view(
        query_path, img_size=model_view_img_size
    ).resize((vis_size, vis_size), Image.BICUBIC)

    if is_change:
        draw_q = ImageDraw.Draw(qry_img)
        draw_q.rectangle(
            [0, 0, vis_size - 1, vis_size - 1],
            outline=(255, 0, 0),
            width=6,
        )

    canvas_w = vis_size * (k + 1)
    canvas = Image.new("RGB", (canvas_w, vis_size))
    for i, im in enumerate(ref_imgs):
        canvas.paste(im, (i * vis_size, 0))
    canvas.paste(qry_img, (k * vis_size, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas_w, 24], fill=(0, 0, 0))
    sim_txt = " | ".join([f"{s:.3f}" for s in topk_sims]) if topk_sims else "-"
    txt = f"score={score:.4f}  sims=[{sim_txt}]"
    draw.text((5, 4), txt, fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_top_p_patch_vis(
    query_path: Path,
    patch_vis: Dict[str, Any],
    out_path: Path,
):
    if not patch_vis:
        return

    top_patch_idx = patch_vis.get("top_patch_idx", [])
    top_patch_vals = patch_vis.get("top_patch_vals", [])
    img_size = int(patch_vis.get("img_size", 560))
    patch_size = int(patch_vis.get("patch_size", 14))

    if len(top_patch_idx) == 0:
        return

    img = load_bgr_for_model_view(query_path, img_size=img_size)

    print(
        "top_patch_vals:",
        float(np.min(top_patch_vals)),
        float(np.mean(top_patch_vals)),
        float(np.max(top_patch_vals)),
    )

    vis_img = draw_top_p_heatmap(
        img_bgr=img,
        top_patch_idx=top_patch_idx,
        top_patch_vals=top_patch_vals,
        img_size=img_size,
        patch_size=patch_size,
        alpha=0.45,
        blur_ksize=0,
        normalize_each=False,
        abs_min=0.2,
        abs_max=0.80,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_img)


def _resolve_path(p: str, bank_root: Path) -> Path:
    p = str(p)
    pp = Path(p)

    if pp.is_absolute():
        return pp

    cand1 = bank_root / pp
    if cand1.exists():
        return cand1.resolve()

    cand2 = pp.resolve()
    if cand2.exists():
        return cand2

    br_name = bank_root.name
    parts = pp.parts
    if len(parts) > 0 and parts[0] == br_name:
        cand3 = bank_root / Path(*parts[1:])
        if cand3.exists():
            return cand3.resolve()

    return cand1.resolve()


def extract_topk_from_out(
    out: dict,
    bank_root: Path,
    rep_idx_fallback: int = 0,
    k: int = 3,
) -> Tuple[List[Path], List[float], int]:
    rj: Any = out.get("ref_topk_json")
    if rj is None:
        return [], [], rep_idx_fallback

    if isinstance(rj, str):
        try:
            rj = json.loads(rj)
        except Exception:
            return [], [], rep_idx_fallback

    if isinstance(rj, dict) and ("topk_paths" in rj):
        topk_paths_all = rj.get("topk_paths") or []
        topk_sims_all = rj.get("topk_sims") or []
        rep = rj.get("rep")

        rep_idx = rep_idx_fallback
        if isinstance(rep, dict) and isinstance(rep.get("frame_idx"), int):
            rep_idx = int(rep["frame_idx"])

        if not isinstance(topk_paths_all, list) or len(topk_paths_all) == 0:
            return [], [], rep_idx

        rep_idx = max(0, min(rep_idx, len(topk_paths_all) - 1))

        p_list = topk_paths_all[rep_idx]
        s_list = (
            topk_sims_all[rep_idx]
            if isinstance(topk_sims_all, list) and rep_idx < len(topk_sims_all)
            else None
        )

        if not isinstance(p_list, list) or len(p_list) == 0:
            return [], [], rep_idx

        pths = [_resolve_path(str(p), bank_root) for p in p_list[:k]]
        if isinstance(s_list, list):
            ss = [float(x) for x in s_list[:k]]
        else:
            ss = [float("nan")] * len(pths)

        return pths, ss, rep_idx

    return [], [], rep_idx_fallback


def plot_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    thr: Optional[float],
    title: str,
    out_path: Path,
):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    if thr is not None:
        plt.axhline(thr)
    plt.xlabel("index")
    plt.ylabel("score")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def safe_ts_to_iso(safe_ts: str) -> str:
    if "T" not in safe_ts:
        return safe_ts

    date_part, time_part = safe_ts.split("T", 1)
    time_part = time_part.replace("-", ":")
    return f"{date_part}T{time_part}"


def main(target_places: Optional[List[str]] = None):
    recv_root = Path(args.bank_root) if args.bank_root is not None else recv_root

    if not recv_root.exists():
        raise FileNotFoundError(f"RECV_ROOT not found: {recv_root.resolve()}")

    cfg = load_cfg(recv_root)
    override_cfg = {}
    if args.config is not None:
        with open(args.config, "r") as f:
            override_cfg = json.load(f)

    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = merge(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    cfg = merge(cfg, override_cfg)

    # override merge
    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = merge(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    cfg = merge(cfg, override_cfg)

    calib_cfg = cfg.get("calib", {})
    infer_cfg = cfg.get("infer", {})
    embed_cfg = cfg.get("embed", {})

    k_cfg = int(calib_cfg.get("k", 3))
    percentile_cfg = int(calib_cfg.get("percentile", 97))
    method_cfg = str(calib_cfg.get("method", "robust"))
    event_rule_cfg = str(infer_cfg.get("event_rule", "max"))
    use_vlm_cfg = bool(infer_cfg.get("use_two_stage_vlm", False))
    img_size_cfg = int(embed_cfg.get("img_size", 560))

    print("[CFG] calib:", {"k": k_cfg, "percentile": percentile_cfg, "method": method_cfg})
    print("[CFG] infer :", {"event_rule": event_rule_cfg, "use_two_stage_vlm": use_vlm_cfg})
    print("[CFG] embed :", {"img_size": img_size_cfg})

    db_path = get_db_path()
    db = sqlite_db.connect_db(db_path)
    sqlite_db.init_db(db)
    sync_places_from_fs(db, recv_root)
    print(f"[DB] connected: {db_path.resolve()}")

    global_model, device = dino_emb.load_model()
    local_model, device = cnn_emb.load_model(
        model_name="resnet18",
        out_layer="layer3",
        device=device,
    )

    sg_raw = cfg["superglue"]
    sg_cfg = SuperGlueMatchConfig(
        resize_long_side=sg_raw["resize_long_side"],
        weights=sg_raw["weights"],
        max_keypoints=sg_raw["max_keypoints"],
        keypoint_threshold=sg_raw["keypoint_threshold"],
        match_threshold=sg_raw["match_threshold"],
        sinkhorn_iterations=sg_raw["sinkhorn_iterations"],
    )
    sg_matcher = SuperGlueMatcher(sg_cfg, device=device)

    engine = {
        "global_model": global_model,
        "local_model": local_model,
        "device": device,
        "bank_root": recv_root,
        "sg_matcher": sg_matcher,
    }

    place_dirs = sorted([p for p in recv_root.iterdir() if p.is_dir()])
    place_dirs = [
        p for p in place_dirs
        if (p / "bank").exists() or (p / "query").exists() or (p / "th_calib").exists()
    ]

    if target_places is not None and len(target_places) > 0:
        target_set = set(target_places)
        place_dirs = [p for p in place_dirs if p.name in target_set]

    print(f"[INFO] filtered places: {target_places}")
    print(f"[INFO] places: {[p.name for p in place_dirs]}")

    try:
        for place_dir in place_dirs:
            plc = place_dir.name
            if args.output_dir:
                out_dir = Path(args.output_dir) / plc
            else:
                out_dir = place_dir / OUT_ROOTNAME
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n===== PLACE {plc} =====")

            sqlite_db.ensure_place(db, plc)

            try:
                place_manager.set_place_mode(db, plc, "query")
            except Exception:
                pass

            thr, calib_scores, _ = calibrate_place(
                str(recv_root),
                plc,
                engine["global_model"],
                engine["local_model"],
                engine["device"],
                sg_matcher=engine.get("sg_matcher"),
            )
            print("[CALIB] thr =", thr, " (#scores=", len(calib_scores), ")")

            try:
                place_manager.set_need_calibration(db, plc, False)
            except Exception:
                pass

            calib_scores_arr = np.array(calib_scores, dtype=np.float32)
            plot_curve(
                xs=np.arange(len(calib_scores_arr)),
                ys=calib_scores_arr,
                thr=thr,
                title=f"th_calib score curve (place={plc}, thr={thr:.4f})",
                out_path=out_dir / "calib_score_curve.png",
            )
            print("[OK] saved:", out_dir / "calib_score_curve.png")

            query_batches = collect_batches_for_place(place_dir, mode="query")
            print(f"[INFO] query batches: {len(query_batches)}")

            pair_dir = out_dir / "pairs"
            pair_dir.mkdir(parents=True, exist_ok=True)

            y_true: List[int] = []
            y_pred: List[int] = []
            event_scores: List[float] = []
            per_place_results: List[Dict[str, Any]] = []

            for bi, b in enumerate(query_batches):
                gt = normalize_label(b.label)
                imgs_bgr = load_imgs_bgr(b.paths)

                captured_at = safe_ts_to_iso(b.safe_ts)

                event_id = sqlite_db.insert_event(
                    db=db,
                    place_id=plc,
                    captured_at=captured_at,
                )

                sqlite_db.insert_frames(
                    db=db,
                    event_id=event_id,
                    image_paths=[str(p) for p in b.paths],
                    frame_scores=None,
                    capture_times=captured_at,
                )

                out = infer_event(
                    imgs_bgr=imgs_bgr,
                    bank_root=engine["bank_root"],
                    plc_idx=plc,
                    cfg=cfg,
                    global_model=engine["global_model"],
                    local_model=engine["local_model"],
                    device=engine["device"],
                    sg_matcher=engine.get("sg_matcher"),
                )

                pred_flag = int(out["anomaly_flag"])
                event_score = float(out["event_score"])
                threshold_used = float(out["threshold"])

                sqlite_db.update_frame_scores(
                    db,
                    event_id,
                    out["frame_scores"],
                )
                sqlite_db.update_event_result(
                    db=db,
                    event_id=event_id,
                    anomaly_flag=pred_flag,
                    anomaly_score=event_score,
                    threshold_used=threshold_used,
                    ref_bank_id=out.get("ref_bank_id"),
                    ref_topk_json=out.get("ref_topk_json"),
                    summary_text=out.get("summary"),
                )

                if gt in {"normal", "abnormal"}:
                    true_flag = 0 if gt == "normal" else 1
                    y_true.append(true_flag)
                    y_pred.append(pred_flag)
                    event_scores.append(event_score)
                else:
                    print(
                        f"[WARN] unlabeled/unknown gt batch -> DB 저장만 수행 "
                        f"(place={plc}, ts={b.safe_ts}, label={b.label})"
                    )

                patch_vis = out.get("patch_vis") or {}
                rep_idx_patch = int(patch_vis.get("frame_idx", 0))
                rep_idx_patch = max(0, min(rep_idx_patch, len(b.paths) - 1))

                topk_paths, topk_sims, rep_idx = extract_topk_from_out(
                    out,
                    engine["bank_root"],
                    rep_idx_fallback=rep_idx_patch,
                    k=k_cfg,
                )
                rep_idx = max(0, min(rep_idx, len(b.paths) - 1))
                rep_q = b.paths[rep_idx]

                if len(topk_paths) > 0:
                    out_pair = pair_dir / f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}.png"
                    save_pair_vis_k(
                        topk_paths=topk_paths[:k_cfg],
                        query_path=rep_q,
                        out_path=out_pair,
                        score=event_score,
                        topk_sims=topk_sims[:k_cfg],
                        is_change=bool(pred_flag),
                        vis_size=384,
                        model_view_img_size=img_size_cfg,
                    )

                align_vis = out.get("align_vis")

                if align_vis:
                    rep_idx_align = int(align_vis.get("frame_idx", rep_idx_patch))
                    rep_idx_align = max(0, min(rep_idx_align, len(b.paths) - 1))
                    rep_q_align = b.paths[rep_idx_align]

                    out_align_prefix = pair_dir / (
                        f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}_aligned"
                    )
                    save_aligned_debug_vis(
                        query_path=rep_q_align,
                        align_vis=align_vis,
                        out_prefix=out_align_prefix,
                        img_size=img_size_cfg,
                        patch_size=None,
                    )

                    best_ref_img_path = patch_vis.get("best_ref_img_path", None)
                    if best_ref_img_path:
                        out_match = pair_dir / (
                            f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}_aligned_matchlines.png"
                        )
                        save_patch_match_vis(
                            query_path=rep_q_align,
                            ref_path=best_ref_img_path,
                            patch_vis=patch_vis,
                            out_path=out_match,
                        )

                elif patch_vis:
                    rep_q_patch = b.paths[rep_idx_patch]
                    out_patch = pair_dir / f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}_heatmap.png"
                    save_top_p_patch_vis(
                        query_path=rep_q_patch,
                        patch_vis=patch_vis,
                        out_path=out_patch,
                    )

                    best_ref_img_path = patch_vis.get("best_ref_img_path", None)
                    if best_ref_img_path:
                        out_match = pair_dir / f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}_matchlines.png"
                        save_patch_match_vis(
                            query_path=rep_q_patch,
                            ref_path=best_ref_img_path,
                            patch_vis=patch_vis,
                            out_path=out_match,
                        )

                print(
                    f"[EVAL] ts={b.safe_ts} gt={gt} pred={pred_flag} "
                    f"score={event_score:.4f} thr={threshold_used:.4f} event_id={event_id}"
                )

                loss_vis = out.get("loss_vis")

                if loss_vis:
                    rep_idx_loss = int(loss_vis.get("frame_idx", rep_idx_patch))
                    rep_idx_loss = max(0, min(rep_idx_loss, len(b.paths) - 1))
                    rep_q_loss = b.paths[rep_idx_loss]

                    out_loss = pair_dir / (
                        f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}_lossmap.png"
                    )
                    save_loss_map_vis(
                        query_path=rep_q_loss,
                        loss_vis=loss_vis,
                        out_path=out_loss,
                        img_size=img_size_cfg,
                    )

                per_place_results.append({
                    "event_id": event_id,
                    "place_id": plc,
                    "safe_ts": b.safe_ts,
                    "captured_at": captured_at,
                    "gt": gt,
                    "pred_flag": pred_flag,
                    "event_score": event_score,
                    "threshold": threshold_used,
                    "ref_bank_id": out.get("ref_bank_id"),
                    "rep_idx": int(rep_idx),
                })

            if event_scores:
                scores_arr = np.array(event_scores, dtype=np.float32)
                plot_curve(
                    xs=np.arange(len(scores_arr)),
                    ys=scores_arr,
                    thr=thr,
                    title=f"query event_score curve (place={plc}, thr={thr:.4f})",
                    out_path=out_dir / "query_event_score_curve.png",
                )
                print("[OK] saved:", out_dir / "query_event_score_curve.png")

            metrics = compute_metrics(y_true, y_pred) if y_true else {}
            fps = [
                r for r, t, p in zip(per_place_results, y_true, y_pred)
                if (t == 0 and p == 1)
            ]
            fns = [
                r for r, t, p in zip(per_place_results, y_true, y_pred)
                if (t == 1 and p == 0)
            ]

            txt_path = out_dir / "metrics.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"place={plc}\n")
                f.write(f"db_path={str(db_path.resolve())}\n")
                f.write(f"cfg.calib.k={k_cfg}\n")
                f.write(f"cfg.calib.percentile={percentile_cfg}\n")
                f.write(f"cfg.calib.method={method_cfg}\n")
                f.write(f"cfg.infer.event_rule={event_rule_cfg}\n")
                f.write(f"cfg.infer.use_two_stage_vlm={use_vlm_cfg}\n")
                f.write(f"cfg.embed.img_size={img_size_cfg}\n")
                f.write(f"threshold={thr:.6f}\n\n")

                f.write("=== Metrics ===\n")
                if metrics:
                    for kk, vv in metrics.items():
                        f.write(f"{kk}: {vv}\n")
                else:
                    f.write("No labeled query batches.\n")

                f.write("\n=== FP batches ===\n")
                for r in fps:
                    f.write(
                        f"  event_id={r['event_id']} ts={r['safe_ts']} "
                        f"score={r['event_score']:.6f}\n"
                    )

                f.write("\n=== FN batches ===\n")
                for r in fns:
                    f.write(
                        f"  event_id={r['event_id']} ts={r['safe_ts']} "
                        f"score={r['event_score']:.6f}\n"
                    )

            print("[OK] saved:", txt_path)

            out_json = out_dir / "offline_eval_results.json"
            out_json.write_text(
                json.dumps(
                    {
                        "db_path": str(db_path.resolve()),
                        "metrics": metrics,
                        "results": per_place_results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print("[OK] saved:", out_json)

    finally:
        db.close()
        print("[DB] closed")

    print("\nDONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--places", nargs="*", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--bank_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(target_places=args.places)