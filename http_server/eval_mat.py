#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_report.py (finalized)

Outputs:
- eval_out/index.html (+ per-place index.html)
- eval_out/<place>/<event_id>.jpg   : montage [Q(rep) | REF0..K-1]
- eval_out/plots/events/<place>/<event_id>.png : per-event plot (frame_score vs frame_idx + threshold)
- eval_out/plots/places/place_<place>.png      : per-place plot (rep_frame_score vs event_index + threshold)

Assumptions:
- events table has: event_id, place_id, captured_at, anomaly_flag, anomaly_score, threshold_used, ref_topk_json, (optional gt_label/label)
- frames table has: event_id, idx, image_path, (optional frame_score)
- ref_topk_json format:
  {
    "topk_paths": List[List[str]],
    "topk_sims":  List[List[float]],
    "rep": {"frame_idx": int, "ref_img_path": str}
  }
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================
# USER CONFIG
# =========================
DB_PATH = Path("./recv/events.db")
RECV_ROOT = Path("./recv")
OUT_DIR = Path("./recv/eval_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_K = 5
TILE_W = 320
TILE_H = 240

DRAW_MIN_TEXT = False          # Q / R0 라벨을 넣고 싶으면 True
SHOW_MISSING_TEXT = False      # missing 타일에 텍스트를 넣고 싶으면 True

SAVE_PLOTS = True
# =========================


# -------------------------
# DB / label helpers
# -------------------------
def _table_has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols


def normalize_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("none", ""):
        return None
    if s in ("normal", "norm", "0", "ok"):
        return "normal"
    if s in ("abnormal", "abnorm", "anomaly", "1", "ng", "unnomal", "unomal", "unormal"):
        return "abnormal"
    return None


def gt_from_filename(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    name = Path(path).name.lower()
    if name.startswith("normal_"):
        return "normal"
    if name.startswith("abnormal_"):
        return "abnormal"
    if name.startswith(("unnomal_", "unomal_", "unormal_")):
        return "abnormal"
    return None


def pred_from_flag(anomaly_flag: Optional[int]) -> Optional[str]:
    if anomaly_flag is None:
        return None
    return "abnormal" if int(anomaly_flag) == 1 else "normal"


@dataclass
class EventRow:
    event_id: str
    place_id: str
    captured_at: str
    anomaly_flag: Optional[int]
    anomaly_score: Optional[float]
    threshold_used: Optional[float]
    ref_topk_json: Optional[str]
    summary_text: Optional[str]
    gt_label_db: Optional[str]


@dataclass
class FrameRow:
    idx: int
    image_path: str
    frame_score: Optional[float]


def load_events(con: sqlite3.Connection) -> List[EventRow]:
    gt_col = "gt_label" if _table_has_column(con, "events", "gt_label") else (
        "label" if _table_has_column(con, "events", "label") else None
    )

    if gt_col:
        q = f"""
        SELECT
            e.event_id, e.place_id, e.captured_at,
            e.anomaly_flag, e.anomaly_score, e.threshold_used,
            e.ref_topk_json, e.summary_text,
            e.{gt_col} as gt_label_db
        FROM events e
        ORDER BY e.captured_at ASC
        """
    else:
        q = """
        SELECT
            e.event_id, e.place_id, e.captured_at,
            e.anomaly_flag, e.anomaly_score, e.threshold_used,
            e.ref_topk_json, e.summary_text,
            NULL as gt_label_db
        FROM events e
        ORDER BY e.captured_at ASC
        """

    cur = con.execute(q)
    out: List[EventRow] = []
    for r in cur.fetchall():
        out.append(EventRow(
            event_id=str(r[0]),
            place_id=str(r[1]),
            captured_at=str(r[2]),
            anomaly_flag=r[3],
            anomaly_score=r[4],
            threshold_used=r[5],
            ref_topk_json=r[6],
            summary_text=r[7],
            gt_label_db=r[8],
        ))
    return out


def load_frames_for_event(con: sqlite3.Connection, event_id: str) -> List[FrameRow]:
    has_frame_score = _table_has_column(con, "frames", "frame_score")
    if has_frame_score:
        q = "SELECT idx, image_path, frame_score FROM frames WHERE event_id=? ORDER BY idx ASC"
    else:
        q = "SELECT idx, image_path, NULL as frame_score FROM frames WHERE event_id=? ORDER BY idx ASC"

    cur = con.execute(q, (event_id,))
    frames: List[FrameRow] = []
    for r in cur.fetchall():
        idx = int(r[0])
        imgp = str(r[1])
        fs = None
        if r[2] is not None:
            try:
                fs = float(r[2])
            except Exception:
                fs = None
        frames.append(FrameRow(idx=idx, image_path=imgp, frame_score=fs))
    return frames


# -------------------------
# ref_topk_json parsing (핵심)
# -------------------------
def parse_rep_frame_idx(ref_topk_json: Optional[str]) -> Optional[int]:
    if not ref_topk_json:
        return None
    try:
        obj = json.loads(ref_topk_json)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    rep = obj.get("rep", None)
    if isinstance(rep, dict) and isinstance(rep.get("frame_idx", None), int):
        return int(rep["frame_idx"])
    return None


def parse_refs_for_rep_frame(ref_topk_json: Optional[str], k: int = REF_K) -> List[str]:
    """
    대표 query frame(rep.frame_idx)에 대응되는 top-k ref 경로만 뽑는다.
    """
    if not ref_topk_json:
        return []
    try:
        obj = json.loads(ref_topk_json)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []

    topk_paths = obj.get("topk_paths", None)
    rep = obj.get("rep", None)

    if isinstance(topk_paths, list) and isinstance(rep, dict):
        fi = rep.get("frame_idx", None)
        if isinstance(fi, int) and 0 <= fi < len(topk_paths):
            row = topk_paths[fi]
            if isinstance(row, list):
                out = [str(x) for x in row if isinstance(x, str)]
                return out[:k]

    # fallback: rep.ref_img_path만이라도
    if isinstance(rep, dict):
        rp = rep.get("ref_img_path", None)
        if isinstance(rp, str) and rp:
            return [rp]

    return []


# -------------------------
# path resolve + image read
# -------------------------
def _first_existing(cands: List[Path]) -> Optional[Path]:
    for p in cands:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def resolve_image_path(path_str: str, place_id: Optional[str] = None) -> Optional[Path]:
    if not path_str:
        return None

    p = Path(path_str)
    cands: List[Path] = []

    # 1) as-is
    if p.is_absolute():
        cands.append(p)
    else:
        cands.append(p.resolve())

    # 2) RECV_ROOT / path
    if not p.is_absolute():
        cands.append((RECV_ROOT / p).resolve())

    # 3) strip to inside recv
    s = str(path_str).replace("\\", "/")
    if "/recv/" in s:
        inside = s.split("/recv/", 1)[1]
        cands.append((RECV_ROOT / inside).resolve())
    elif s.startswith("recv/"):
        inside = s.split("recv/", 1)[1]
        cands.append((RECV_ROOT / inside).resolve())

    # 4) filename fallback
    name = p.name
    if ("/" not in s and "\\" not in path_str) and name:
        if place_id:
            cands.append((RECV_ROOT / place_id / "bank" / name).resolve())
            cands.append((RECV_ROOT / place_id / "query" / name).resolve())
            cands.append((RECV_ROOT / place_id / "th_calib" / name).resolve())
        cands.append((RECV_ROOT / name).resolve())

    return _first_existing(cands)


def safe_read_img_resolved(path_str: str, place_id: Optional[str] = None) -> Optional[np.ndarray]:
    p = resolve_image_path(path_str, place_id=place_id)
    if p is None:
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


# -------------------------
# montage helpers
# -------------------------
def resize_keep_aspect(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def put_min_text(img: np.ndarray, text: str) -> np.ndarray:
    if not DRAW_MIN_TEXT:
        return img
    out = img.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def blank_tile(label: Optional[str] = None) -> np.ndarray:
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    if SHOW_MISSING_TEXT and label:
        cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return tile


def draw_border(img: np.ndarray, color: Tuple[int, int, int], thickness: int = 6) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)
    return out


def pick_rep_query_frame_path(frames: List[FrameRow], rep_frame_idx: Optional[int]) -> Optional[str]:
    """
    대표 query 이미지: rep.frame_idx가 frames에 있으면 그 프레임.
    없으면 idx=0(최소 idx) 사용.
    """
    if not frames:
        return None
    if rep_frame_idx is not None:
        for fr in frames:
            if fr.idx == rep_frame_idx:
                return fr.image_path
    return frames[0].image_path


def make_montage_repq_refs(
    rep_query_path: Optional[str],
    ref_paths: List[str],
    out_path: Path,
    place_id: str,
    pred_label: Optional[str],
    k: int = REF_K,
) -> bool:
    tiles: List[np.ndarray] = []

    # Q(rep)
    qimg = safe_read_img_resolved(rep_query_path, place_id=place_id) if rep_query_path else None
    if qimg is None:
        qtile = blank_tile("Q missing")
    else:
        qtile = resize_keep_aspect(qimg, TILE_W, TILE_H)
        qtile = put_min_text(qtile, "Q")
        # Query border (pred)
        if pred_label == "normal":
            qtile = draw_border(qtile, (255, 0, 0), thickness=6)  # blue (BGR)
        elif pred_label == "abnormal":
            qtile = draw_border(qtile, (0, 0, 255), thickness=6)  # red
        else:
            qtile = draw_border(qtile, (128, 128, 128), thickness=4)
    tiles.append(qtile)

    # Refs (top-k for rep frame)
    for i in range(k):
        rp = ref_paths[i] if i < len(ref_paths) else ""
        rimg = safe_read_img_resolved(rp, place_id=place_id) if rp else None
        if rimg is None:
            tiles.append(blank_tile(f"R{i} missing"))
        else:
            rt = resize_keep_aspect(rimg, TILE_W, TILE_H)
            rt = put_min_text(rt, f"R{i}")
            tiles.append(rt)

    montage = np.concatenate(tiles, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), montage))


# -------------------------
# metrics + IO
# -------------------------
def confusion_counts(pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    tn = fp = fn = tp = 0
    for gt, pred in pairs:
        if gt == "normal" and pred == "normal":
            tn += 1
        elif gt == "normal" and pred == "abnormal":
            fp += 1
        elif gt == "abnormal" and pred == "normal":
            fn += 1
        elif gt == "abnormal" and pred == "abnormal":
            tp += 1
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def prf_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_html_index(path: Path, title: str, items: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; padding:16px;}",
        ".grid{display:flex; flex-wrap:wrap; gap:14px;}",
        ".card{width:1100px; border:1px solid #ddd; padding:10px; border-radius:10px;}",
        ".cap{font-size:13px; color:#333; margin-top:8px; white-space:pre-wrap;}",
        "img{max-width:100%; border-radius:8px;}",
        "</style></head><body>",
        f"<h2>{title}</h2>",
        "<div class='grid'>",
    ]
    for rel, cap in items:
        html += [
            "<div class='card'>",
            f"<a href='{rel}' target='_blank'><img src='{rel}'/></a>",
            f"<div class='cap'>{cap}</div>",
            "</div>",
        ]
    html += ["</div></body></html>"]
    path.write_text("\n".join(html), encoding="utf-8")


# -------------------------
# plotting: event 1개 / place 1개 (너 요구에 맞춘 핵심)
# -------------------------
def save_event_and_place_plots(con: sqlite3.Connection, events: List[EventRow]) -> None:
    """
    Event plot (1 per event):
      - x: frame_idx
      - y: frame_score (all frames)
      - horizontal line: threshold_used

    Place plot (1 per place):
      - x: event_index (time order)
      - y1: rep_frame_score (frame_score at rep.frame_idx)
      - y2: threshold_used (event-wise)
    """
    import matplotlib.pyplot as plt

    plot_root = OUT_DIR / "plots"
    ev_dir = plot_root / "events"
    pl_dir = plot_root / "places"
    ev_dir.mkdir(parents=True, exist_ok=True)
    pl_dir.mkdir(parents=True, exist_ok=True)

    has_frame_score = _table_has_column(con, "frames", "frame_score")
    if not has_frame_score:
        print("[PLOT] frames.frame_score 없음 -> event/place plot에서 frame_score 라인은 스킵됩니다.")
        return

    # ---- per-event plots
    for e in events:
        frames = load_frames_for_event(con, e.event_id)
        xs = [fr.idx for fr in frames if fr.frame_score is not None]
        ys = [fr.frame_score for fr in frames if fr.frame_score is not None]
        if len(xs) == 0:
            continue

        thr = e.threshold_used if e.threshold_used is not None else np.nan

        outp = ev_dir / e.place_id
        outp.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(xs, ys, marker="o", linestyle="-", label="frame_score")
        if not np.isnan(thr):
            plt.axhline(thr, linestyle="--", label="threshold")
        plt.xlabel("frame_idx")
        plt.ylabel("score")
        plt.title(f"place={e.place_id} event={e.event_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(outp / f"{e.event_id}.png"), dpi=200)
        plt.close()

    # ---- per-place plots
    # collect per place: rep_frame_score + threshold per event (time order)
    by_place: Dict[str, List[Tuple[float, float]]] = {}
    # value: list of (rep_score, thr)

    for e in events:
        rep_idx = parse_rep_frame_idx(e.ref_topk_json)
        frames = load_frames_for_event(con, e.event_id)
        if not frames:
            continue

        thr = e.threshold_used if e.threshold_used is not None else np.nan

        rep_score = None
        if rep_idx is not None:
            for fr in frames:
                if fr.idx == rep_idx:
                    rep_score = fr.frame_score
                    break
        if rep_score is None:
            # fallback: idx0 score
            rep_score = frames[0].frame_score

        if rep_score is None:
            continue

        by_place.setdefault(e.place_id, []).append((float(rep_score), float(thr) if thr is not None else np.nan))

    for place_id, items in by_place.items():
        if len(items) == 0:
            continue
        xs = list(range(len(items)))
        rep_scores = [it[0] for it in items]
        thrs = [it[1] for it in items]

        plt.figure()
        plt.plot(xs, rep_scores, marker="o", linestyle="-", label="rep_frame_score")
        plt.plot(xs, thrs, marker="o", linestyle="--", label="threshold")
        plt.xlabel("event_index (captured_at order)")
        plt.ylabel("score")
        plt.title(f"place={place_id} rep_score vs threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(pl_dir / f"place_{place_id}.png"), dpi=200)
        plt.close()


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row

    events = load_events(con)
    print("[DB] loaded events:", len(events))
    if len(events) == 0:
        con.close()
        print("⚠️ events 0 -> nothing to do.")
        return

    # plots (event 1개, place 1개)
    if SAVE_PLOTS:
        save_event_and_place_plots(con, events)

    # eval + montage
    per_event_rows: List[List[Any]] = []
    valid_pairs_by_place: Dict[str, List[Tuple[str, str]]] = {}
    global_pairs: List[Tuple[str, str]] = []
    html_items_by_place: Dict[str, List[Tuple[str, str]]] = {}

    for e in events:
        frames = load_frames_for_event(con, e.event_id)

        # rep frame idx
        rep_idx = parse_rep_frame_idx(e.ref_topk_json)
        rep_q_path = pick_rep_query_frame_path(frames, rep_idx)

        # GT는 대표 query 파일명 기준으로도 복원 가능(없으면 None)
        gt = normalize_label(e.gt_label_db) or gt_from_filename(rep_q_path)
        pred = pred_from_flag(e.anomaly_flag)

        eval_ok = (gt in ("normal", "abnormal")) and (pred in ("normal", "abnormal"))

        per_event_rows.append([
            e.event_id,
            e.place_id,
            e.captured_at,
            gt if gt else "",
            pred if pred else "",
            int(e.anomaly_flag) if e.anomaly_flag is not None else "",
            float(e.anomaly_score) if e.anomaly_score is not None else "",
            float(e.threshold_used) if e.threshold_used is not None else "",
            rep_q_path or "",
        ])

        if eval_ok:
            valid_pairs_by_place.setdefault(e.place_id, []).append((gt, pred))
            global_pairs.append((gt, pred))

        # refs = rep frame에 대응되는 top-k만
        ref_paths = parse_refs_for_rep_frame(e.ref_topk_json, k=REF_K)

        out_rel = f"{e.place_id}/{e.event_id}.jpg"
        out_path = OUT_DIR / out_rel

        ok = make_montage_repq_refs(
            rep_query_path=rep_q_path,
            ref_paths=ref_paths,
            out_path=out_path,
            place_id=e.place_id,
            pred_label=pred,
            k=REF_K,
        )

        caption = f"event={e.event_id} place={e.place_id} GT={gt} PRED={pred} ev_score={e.anomaly_score} thr={e.threshold_used}"
        if not ok:
            caption += " [WRITE_FAIL]"
        html_items_by_place.setdefault(e.place_id, []).append((out_rel, caption))

    # metrics summary
    summary_rows: List[List[Any]] = []
    g_counts = confusion_counts(global_pairs) if global_pairs else {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    g_prf = prf_from_counts(g_counts)
    summary_rows.append([
        "__ALL__",
        len(global_pairs),
        g_counts["TN"], g_counts["FP"], g_counts["FN"], g_counts["TP"],
        g_prf["precision"], g_prf["recall"], g_prf["f1"], g_prf["accuracy"],
    ])
    for place_id, pairs in sorted(valid_pairs_by_place.items(), key=lambda x: x[0]):
        c = confusion_counts(pairs)
        prf = prf_from_counts(c)
        summary_rows.append([
            place_id,
            len(pairs),
            c["TN"], c["FP"], c["FN"], c["TP"],
            prf["precision"], prf["recall"], prf["f1"], prf["accuracy"],
        ])

    write_csv(
        OUT_DIR / "per_event.csv",
        header=[
            "event_id", "place_id", "captured_at",
            "gt_label", "pred_label", "anomaly_flag",
            "event_score", "threshold_used", "rep_query_image_path",
        ],
        rows=per_event_rows,
    )
    write_csv(
        OUT_DIR / "summary_by_place.csv",
        header=[
            "place_id", "n_eval",
            "TN", "FP", "FN", "TP",
            "precision", "recall", "f1", "accuracy",
        ],
        rows=summary_rows,
    )

    # HTML viewer
    global_items: List[Tuple[str, str]] = []
    for place_id, items in sorted(html_items_by_place.items(), key=lambda x: x[0]):
        global_items.extend(items[:200])
        write_html_index(OUT_DIR / place_id / "index.html", f"Eval Viewer (place={place_id})", items)
    write_html_index(OUT_DIR / "index.html", "Eval Viewer (ALL)", global_items)

    con.close()

    print("✅ Done")
    print(f"- HTML:  {OUT_DIR / 'index.html'}")
    print(f"- Montage images: {OUT_DIR}/<place>/<event>.jpg")
    print(f"- Plots:")
    print(f"    - per-event: {OUT_DIR}/plots/events/<place>/<event>.png")
    print(f"    - per-place: {OUT_DIR}/plots/places/place_<place>.png")
    print(f"- CSV:")
    print(f"    - {OUT_DIR / 'summary_by_place.csv'}")
    print(f"    - {OUT_DIR / 'per_event.csv'}")


if __name__ == "__main__":
    main()