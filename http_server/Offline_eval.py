# offline_eval.py
# ref bank 폴더트리 완성되어있는 상황에서 event 단위별 추론/평가하는 스크립트
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import dino_emb
from distance import infer_event, calibrate_place   # 너의 최신 distance.py 기준
from config import load_cfg


# =========================
# 설정
# =========================
RECV_ROOT = Path("./recv")          # http_server.py의 SAVE_ROOT
OUT_ROOTNAME = "_offline_out"       # place_dir 하위 출력 폴더명
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# batch 구조
# =========================
@dataclass
class Batch:
    place_id: str
    safe_ts: str
    label: Optional[str]     # normal/abnormal or None
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


def parse_server_filename(p: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    서버 저장 파일명에서 meta 추출.
    return (label, place_id, mode, safe_ts, idx)
    """
    stem = p.stem

    # label 있는 케이스: "{label}_{place_id}_{mode}_{safe_ts}_{i:03d}"
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

    # label 없는 케이스: "{place_id}_{mode}_{safe_ts}_{i:03d}"
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
    """
    place_dir/query 에서 (safe_ts, label)로 묶어서 배치 생성.
    """
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
        out.append(Batch(place_id=place_id, safe_ts=safe_ts, label=label, paths=[p for _, p in items]))

    out.sort(key=lambda b: b.safe_ts)
    return out


def load_imgs_bgr(paths: List[Path]) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR
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
        "TP": float(tp), "TN": float(tn), "FP": float(fp), "FN": float(fn),
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "N_total": float(len(y_true)),
        "N_pos": float(sum(y_true)), "N_neg": float(len(y_true) - sum(y_true)),
    }


def save_pair_vis_k(
    topk_paths: List[Path],
    query_path: Path,
    out_path: Path,
    score: float,
    topk_sims: List[float],
    is_change: bool,
    vis_size: int = 384,
):
    k = len(topk_paths)

    ref_imgs = [
        Image.open(p).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)
        for p in topk_paths
    ]
    qry_img = Image.open(query_path).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)

    if is_change:
        draw_q = ImageDraw.Draw(qry_img)
        draw_q.rectangle([0, 0, vis_size - 1, vis_size - 1], outline=(255, 0, 0), width=6)

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


def _resolve_path(p: str, bank_root: Path) -> Path:
    """
    p가
      - 절대경로면 그대로
      - 상대경로면 다음 우선순위로 해석:
        1) bank_root/p
        2) CWD 기준 resolve
        3) "recv/00/..." 처럼 bank_root 이름이 중복 prefix면 제거 후 bank_root에 붙임
        4) fallback: bank_root/p
    """
    p = str(p)
    pp = Path(p)

    if pp.is_absolute():
        return pp

    cand1 = (bank_root / pp)
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


def extract_topk_from_out(out: dict, bank_root: Path, k: int = 3) -> Tuple[List[Path], List[float]]:
    """
    infer_event 결과(ref_topk_json) 구조 지원:
      ref_topk_json = {
        "topk_paths": [[...], [...], ...],
        "topk_sims":  [[...], [...], ...],
        "rep": {"frame_idx": i, "ref_img_path": "..."} or None
      }
    return: (rep_frame_topk_paths, rep_frame_topk_sims)
    """
    rj: Any = out.get("ref_topk_json")
    if rj is None:
        return [], []

    if isinstance(rj, str):
        try:
            rj = json.loads(rj)
        except Exception:
            return [], []

    if isinstance(rj, dict) and ("topk_paths" in rj):
        topk_paths_all = rj.get("topk_paths") or []
        topk_sims_all = rj.get("topk_sims") or []
        rep = rj.get("rep")

        rep_idx = 0
        if isinstance(rep, dict) and isinstance(rep.get("frame_idx"), int):
            rep_idx = int(rep["frame_idx"])

        if not isinstance(topk_paths_all, list) or len(topk_paths_all) == 0:
            return [], []

        rep_idx = max(0, min(rep_idx, len(topk_paths_all) - 1))

        p_list = topk_paths_all[rep_idx]
        s_list = topk_sims_all[rep_idx] if isinstance(topk_sims_all, list) and rep_idx < len(topk_sims_all) else None

        if not isinstance(p_list, list) or len(p_list) == 0:
            return [], []

        pths = [_resolve_path(str(p), bank_root) for p in p_list[:k]]
        if isinstance(s_list, list):
            ss = [float(x) for x in s_list[:k]]
        else:
            ss = [float("nan")] * len(pths)

        return pths, ss

    return [], []


def plot_curve(xs: np.ndarray, ys: np.ndarray, thr: Optional[float], title: str, out_path: Path):
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


def main():
    if not RECV_ROOT.exists():
        raise FileNotFoundError(f"RECV_ROOT not found: {RECV_ROOT.resolve()}")

    # cfg는 "참고용 출력" + distance.py 내부에서 사용할 값은 distance.py가 다시 load_cfg(bank_root)로 읽음
    cfg = load_cfg(RECV_ROOT)

    calib_cfg = cfg.get("calib", {})
    infer_cfg = cfg.get("infer", {})
    k_cfg = int(calib_cfg.get("k", 3))
    percentile_cfg = int(calib_cfg.get("percentile", 97))
    method_cfg = str(calib_cfg.get("method", "robust"))
    event_rule_cfg = str(infer_cfg.get("event_rule", "max"))
    use_vlm_cfg = bool(infer_cfg.get("use_two_stage_vlm", False))

    print("[CFG] calib:", {"k": k_cfg, "percentile": percentile_cfg, "method": method_cfg})
    print("[CFG] infer :", {"event_rule": event_rule_cfg, "use_two_stage_vlm": use_vlm_cfg})

    model, device = dino_emb.load_model()
    engine = {"model": model, "device": device, "bank_root": RECV_ROOT}

    place_dirs = sorted([p for p in RECV_ROOT.iterdir() if p.is_dir()])
    place_dirs = [p for p in place_dirs if (p / "bank").exists() or (p / "query").exists() or (p / "th_calib").exists()]
    print(f"[INFO] places: {[p.name for p in place_dirs]}")

    for place_dir in place_dirs:
        plc = place_dir.name
        out_dir = place_dir / OUT_ROOTNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== PLACE {plc} =====")

        # 1) threshold 계산 (cfg 기반)
        thr, calib_scores, _ = calibrate_place(
            str(RECV_ROOT), plc, model, device
        )
        print("[CALIB] thr =", thr, " (#scores=", len(calib_scores), ")")

        # 2) th_calib curve
        calib_scores_arr = np.array(calib_scores, dtype=np.float32)
        plot_curve(
            xs=np.arange(len(calib_scores_arr)),
            ys=calib_scores_arr,
            thr=thr,
            title=f"th_calib score curve (place={plc}, thr={thr:.4f})",
            out_path=out_dir / "calib_score_curve.png",
        )
        print("[OK] saved:", out_dir / "calib_score_curve.png")

        # 3) query batches
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
            if gt not in {"normal", "abnormal"}:
                print(f"[SKIP] batch(no/unknown label) place={plc} ts={b.safe_ts} label={b.label}")
                continue

            imgs_bgr = load_imgs_bgr(b.paths)

            # infer_event도 cfg/threshold.json 기반으로 내부에서 처리
            out = infer_event(
                imgs_bgr,
                plc,
                engine["bank_root"],
                engine["model"],
                engine["device"],
            )

            pred_flag = int(out["anomaly_flag"])
            true_flag = 0 if gt == "normal" else 1

            y_true.append(true_flag)
            y_pred.append(pred_flag)
            event_scores.append(float(out["event_score"]))

            # 대표 query 이미지: 기본은 첫 장 (VLM gate에서 rep을 따로 만들면 extract_topk_from_out가 rep_idx 따라감)
            rep_q = b.paths[0]

            # infer_event 포맷에 맞게 topk 복원
            topk_paths, topk_sims = extract_topk_from_out(out, engine["bank_root"], k=k_cfg)

            if len(topk_paths) > 0:
                out_pair = pair_dir / f"{bi:04d}_{plc}_{b.safe_ts}_gt{gt}_pred{pred_flag}.png"
                save_pair_vis_k(
                    topk_paths=topk_paths[:k_cfg],
                    query_path=rep_q,
                    out_path=out_pair,
                    score=float(out["event_score"]),
                    topk_sims=topk_sims[:k_cfg],
                    is_change=bool(pred_flag),
                    vis_size=384,
                )

            print(f"[EVAL] ts={b.safe_ts} gt={gt} pred={pred_flag} score={out['event_score']:.4f} thr={out['threshold']:.4f}")

            per_place_results.append({
                "place_id": plc,
                "safe_ts": b.safe_ts,
                "gt": gt,
                "pred_flag": pred_flag,
                "event_score": float(out["event_score"]),
                "threshold": float(out["threshold"]),
                "ref_bank_id": out.get("ref_bank_id"),
            })

        # 4) query event_score curve
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

        # 5) metrics + FP/FN
        metrics = compute_metrics(y_true, y_pred) if y_true else {}
        fps = [r for r, t, p in zip(per_place_results, y_true, y_pred) if (t == 0 and p == 1)]
        fns = [r for r, t, p in zip(per_place_results, y_true, y_pred) if (t == 1 and p == 0)]

        txt_path = out_dir / "metrics.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"place={plc}\n")
            f.write(f"cfg.calib.k={k_cfg}\n")
            f.write(f"cfg.calib.percentile={percentile_cfg}\n")
            f.write(f"cfg.calib.method={method_cfg}\n")
            f.write(f"cfg.infer.event_rule={event_rule_cfg}\n")
            f.write(f"cfg.infer.use_two_stage_vlm={use_vlm_cfg}\n")
            f.write(f"threshold={thr:.6f}\n\n")
            f.write("=== Metrics ===\n")
            if metrics:
                for kk, vv in metrics.items():
                    f.write(f"{kk}: {vv}\n")
            else:
                f.write("No labeled query batches.\n")
            f.write("\n=== FP batches ===\n")
            for r in fps:
                f.write(f"  ts={r['safe_ts']} score={r['event_score']:.6f}\n")
            f.write("\n=== FN batches ===\n")
            for r in fns:
                f.write(f"  ts={r['safe_ts']} score={r['event_score']:.6f}\n")

        print("[OK] saved:", txt_path)

        out_json = out_dir / "offline_eval_results.json"
        out_json.write_text(json.dumps({"metrics": metrics, "results": per_place_results}, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[OK] saved:", out_json)

    print("\nDONE.")


if __name__ == "__main__":
    main()