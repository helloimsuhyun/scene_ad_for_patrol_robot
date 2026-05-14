#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline evaluation 부산물 기반 통계/곡선 리포트 생성 스크립트

핵심 정의:
1) ROC/PR curve:
   - frame 단위
   - 각 frame_score를 하나의 샘플로 사용
   - frame label은 해당 event label을 그대로 부여
     예: abnormal event 안의 frame들은 모두 positive frame으로 간주

2) Best threshold:
   - frame_score threshold 기준
   - best-F1 threshold 1개
   - 오탐 방지용 target precision threshold 1개, 기본 Precision >= 0.95

3) 최종 평가지표:
   - event 단위
   - 선택된 frame threshold를 각 event의 frame_scores에 적용
   - frame_flags -> vote -> event_pred
   - event label과 event_pred로 TP/TN/FP/FN/Accuracy/Precision/Recall/F1 계산

전제:
recv/_offline_out/<place>/offline_eval_results.json 구조가 있어야 함.

사용 예시:
cd sentrynexcontrol

python -m vision_server.offline_report \
  --out_root ./recv/_offline_out \
  --report_root ./recv/val_offline_report \
  --places 01 06 07 08 \
  --vote_rule majority \
  --target_precision 0.95


python -m vision_server.offline_report \
  --out_root ./recv/_offline_out \
  --report_root ./recv/test_offline_report \
  --places P001 P002 P003 \
  --vote_rule majority \
  --target_precision 0.95
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# I/O
# =========================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_csv_safe_events(events: List[Dict]) -> List[Dict]:
    out = []
    for ev in events:
        row = dict(ev)
        row["frame_scores"] = json.dumps(row.get("frame_scores", []), ensure_ascii=False)
        row["frame_change_flags"] = json.dumps(row.get("frame_change_flags", []), ensure_ascii=False)
        row["query_paths"] = json.dumps(row.get("query_paths", []), ensure_ascii=False)
        out.append(row)
    return out


def make_frame_sample_rows(events: List[Dict]) -> List[Dict]:
    rows = []
    for ev in events:
        label = int(ev["label"])
        place = ev.get("place", "")
        event_key = ev.get("event_key", "")
        for i, s in enumerate(ev.get("frame_scores", [])):
            rows.append({
                "place": place,
                "event_key": event_key,
                "frame_idx": i,
                "frame_label_from_event": label,
                "frame_score": float(s),
            })
    return rows


# =========================================================
# Metric utils
# =========================================================

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp = c["tp"]
    tn = c["tn"]
    fp = c["fp"]
    fn = c["fn"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    c = confusion_counts(y_true, y_pred)
    m = metrics_from_counts(c)
    return {**c, **m, "n": int(len(y_true))}


def safe_auc(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    return float(np.trapz(y_sorted, x_sorted))


def threshold_candidates(scores: np.ndarray) -> List[float]:
    """
    threshold sweep 후보.
    실제 infer_event는 dist > threshold 구조이므로,
    equality ambiguity를 줄이기 위해 unique score 사이 midpoint를 사용한다.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return []

    unique = sorted(set(float(x) for x in scores), reverse=True)

    if len(unique) == 1:
        v = unique[0]
        eps = max(abs(v) * 1e-9, 1e-9)
        return [v + eps, v - eps]

    thrs = []
    eps_hi = max(abs(unique[0]) * 1e-9, 1e-9)
    eps_lo = max(abs(unique[-1]) * 1e-9, 1e-9)

    thrs.append(unique[0] + eps_hi)

    for a, b in zip(unique[:-1], unique[1:]):
        thrs.append((a + b) / 2.0)

    thrs.append(unique[-1] - eps_lo)
    return [float(t) for t in thrs]


def best_threshold_by_f1(sweep_rows: List[Dict]) -> Optional[Dict]:
    if len(sweep_rows) == 0:
        return None

    return max(
        sweep_rows,
        key=lambda r: (
            float(r["f1"]),
            float(r["recall"]),
            float(r["precision"]),
            -float(r["threshold"]),
        )
    )


def best_threshold_by_min_recall(
    sweep_rows: List[Dict],
    min_recall: float = 0.90,
) -> Optional[Dict]:
    candidates = [r for r in sweep_rows if float(r["recall"]) >= float(min_recall)]
    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: (
            float(r["precision"]),
            float(r["f1"]),
            float(r["recall"]),
            float(r["threshold"]),
        )
    )


def best_threshold_by_min_precision(
    sweep_rows: List[Dict],
    min_precision: float = 0.95,
) -> Optional[Dict]:
    candidates = [r for r in sweep_rows if float(r["precision"]) >= float(min_precision)]
    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: (
            float(r["recall"]),
            float(r["f1"]),
            float(r["precision"]),
            -float(r["threshold"]),
        )
    )


def average_precision_score_manual(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    sklearn 없이 Average Precision 계산.
    여기서는 frame-level AP 계산에 사용한다.
    """
    y_true = y_true.astype(int)
    scores = scores.astype(float)

    num_pos = int(np.sum(y_true == 1))
    if num_pos == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    tp = 0
    precisions_at_pos = []

    for i, y in enumerate(y_sorted, start=1):
        if y == 1:
            tp += 1
            precisions_at_pos.append(tp / i)

    return float(np.mean(precisions_at_pos)) if precisions_at_pos else 0.0


# =========================================================
# Vote logic
# =========================================================

def required_votes_for_event(
    num_frames: int,
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> int:
    """
    event가 abnormal이 되기 위해 threshold를 초과해야 하는 frame 개수.

    실제 infer_event의 vote는:
      anomaly_flag = 1 if n_ab > (n / 2) else 0

    따라서 majority는 floor(n/2)+1.
    """
    if num_frames <= 0:
        return 1

    if vote_rule == "any":
        return 1

    if vote_rule == "majority":
        return int(math.floor(num_frames * 0.5)) + 1

    if vote_rule == "ratio":
        return max(1, int(math.ceil(num_frames * float(vote_ratio))))

    if vote_rule == "min_count":
        return max(1, int(min_positive_frames))

    raise ValueError(f"Unknown vote_rule: {vote_rule}")


def pred_from_frame_threshold(
    frame_scores: List[float],
    threshold: float,
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> int:
    """
    실제 판정 방식과 맞춤:
    frame_score > threshold 인 frame 개수를 세고,
    vote 조건을 만족하면 event abnormal.
    """
    if len(frame_scores) == 0:
        return 0

    required = required_votes_for_event(
        num_frames=len(frame_scores),
        vote_rule=vote_rule,
        vote_ratio=vote_ratio,
        min_positive_frames=min_positive_frames,
    )

    num_positive = int(np.sum(np.array(frame_scores, dtype=float) > float(threshold)))
    return 1 if num_positive >= required else 0


def vote_score_from_frame_scores(
    frame_scores: List[float],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> float:
    """
    참고용 event-level score.
    frame threshold + vote와 대응되는 kth largest score.
    ROC/PR/best threshold에는 사용하지 않는다.
    """
    if len(frame_scores) == 0:
        return float("-inf")

    scores = sorted([float(s) for s in frame_scores], reverse=True)

    required = required_votes_for_event(
        num_frames=len(scores),
        vote_rule=vote_rule,
        vote_ratio=vote_ratio,
        min_positive_frames=min_positive_frames,
    )

    required = min(max(required, 1), len(scores))
    return float(scores[required - 1])


def add_vote_scores(
    events: List[Dict],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> List[Dict]:
    out = []

    for ev in events:
        row = dict(ev)
        frame_scores = row.get("frame_scores", [])

        row["required_votes"] = required_votes_for_event(
            num_frames=len(frame_scores),
            vote_rule=vote_rule,
            vote_ratio=vote_ratio,
            min_positive_frames=min_positive_frames,
        )

        row["vote_score"] = vote_score_from_frame_scores(
            frame_scores=frame_scores,
            vote_rule=vote_rule,
            vote_ratio=vote_ratio,
            min_positive_frames=min_positive_frames,
        )

        out.append(row)

    return out


# =========================================================
# Frame-level threshold sweep
# =========================================================

def flatten_frame_samples(events: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 frame_score를 ROC/PR 샘플로 사용한다.

    주의:
    현재 offline 결과에는 frame-level label이 없으므로,
    event label을 해당 event의 모든 frame에 동일하게 부여한다.
    """
    y_frames = []
    s_frames = []

    for ev in events:
        label = int(ev["label"])
        for s in ev.get("frame_scores", []):
            y_frames.append(label)
            s_frames.append(float(s))

    return (
        np.array(y_frames, dtype=int),
        np.array(s_frames, dtype=float),
    )


def sweep_frame_scores(
    y_frame_true: np.ndarray,
    frame_scores: np.ndarray,
) -> List[Dict]:
    """
    각 frame_score를 직접 threshold sweep한다.
    """
    rows = []

    for thr in threshold_candidates(frame_scores):
        y_frame_pred = (frame_scores > float(thr)).astype(int)
        met = compute_metrics(y_frame_true, y_frame_pred)

        rows.append({
            "threshold": float(thr),
            **met,
        })

    return rows


def event_metrics_at_frame_threshold(
    events: List[Dict],
    threshold: float,
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> Dict:
    """
    frame 기준으로 선택된 threshold를 event별 frame_scores에 적용한 뒤,
    vote로 event prediction을 만들고 event-level metric을 계산한다.
    """
    y_true = np.array([int(ev["label"]) for ev in events], dtype=int)

    y_pred = np.array([
        pred_from_frame_threshold(
            frame_scores=ev.get("frame_scores", []),
            threshold=float(threshold),
            vote_rule=vote_rule,
            vote_ratio=vote_ratio,
            min_positive_frames=min_positive_frames,
        )
        for ev in events
    ], dtype=int)

    out = compute_metrics(y_true, y_pred)
    out["threshold"] = float(threshold)
    return out


def roc_curve_points(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rows = sweep_frame_scores(y_true, scores)

    fpr = np.array([r["fpr"] for r in rows], dtype=np.float32)
    tpr = np.array([r["recall"] for r in rows], dtype=np.float32)
    thrs = [float(r["threshold"]) for r in rows]

    order = np.argsort(fpr)
    return fpr[order], tpr[order], [thrs[i] for i in order]


def pr_curve_points(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rows = sweep_frame_scores(y_true, scores)

    precision = np.array([r["precision"] for r in rows], dtype=np.float32)
    recall = np.array([r["recall"] for r in rows], dtype=np.float32)
    thrs = [float(r["threshold"]) for r in rows]

    order = np.argsort(recall)
    return recall[order], precision[order], [thrs[i] for i in order]


# =========================================================
# Data loading
# =========================================================

def collect_events(out_root: Path, places: Optional[List[str]] = None) -> List[Dict]:
    rows = []

    if not out_root.exists():
        print(f"[ERROR] out_root does not exist: {out_root}")
        return rows

    if places is None or len(places) == 0:
        place_dirs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    else:
        place_dirs = [out_root / str(p) for p in places]

    for place_dir in place_dirs:
        plc = place_dir.name
        result_path = place_dir / "offline_eval_results.json"

        if not result_path.exists():
            print(f"[WARN] missing: {result_path}")
            continue

        try:
            data = load_json(result_path)
        except Exception as e:
            print(f"[WARN] failed to load {result_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] invalid format: {result_path}")
            continue

        for ev in data:
            try:
                label = int(ev.get("label"))
                pred = int(ev.get("pred"))
            except Exception:
                print(f"[WARN] skip invalid event in place={plc}: {ev.get('event_key')}")
                continue

            event_score = ev.get("event_score", None)
            try:
                event_score = float(event_score) if event_score is not None else math.nan
            except Exception:
                event_score = math.nan

            threshold = ev.get("threshold", None)
            try:
                threshold = float(threshold) if threshold is not None else math.nan
            except Exception:
                threshold = math.nan

            frame_scores_raw = ev.get("frame_scores", [])
            frame_scores = []
            for x in frame_scores_raw:
                try:
                    frame_scores.append(float(x))
                except Exception:
                    pass

            frame_flags_raw = ev.get("frame_change_flags", [])
            frame_change_flags = []
            for x in frame_flags_raw:
                try:
                    frame_change_flags.append(int(x))
                except Exception:
                    pass

            query_paths = ev.get("query_paths", [])
            if not isinstance(query_paths, list):
                query_paths = []

            rows.append({
                "place": plc,
                "event_key": str(ev.get("event_key", "")),
                "label": label,

                # 기존 Offline_eval의 최종 event-level pred.
                # current operating point 평가에 사용.
                "pred": pred,

                # 참고용. threshold sweep에는 사용하지 않음.
                "event_score": event_score,

                # frame-level ROC/PR 및 threshold sweep에 사용.
                "frame_scores": frame_scores,
                "frame_change_flags": frame_change_flags,

                # 기존 시스템 threshold.
                "threshold": threshold,

                "num_frames": len(frame_scores),
                "frame_score_mean": float(np.mean(frame_scores)) if frame_scores else math.nan,
                "frame_score_max": float(np.max(frame_scores)) if frame_scores else math.nan,
                "frame_score_min": float(np.min(frame_scores)) if frame_scores else math.nan,

                "summary": str(ev.get("summary", "")),
                "query_paths": query_paths,
            })

    return rows


# =========================================================
# Stats
# =========================================================

def build_place_stats(
    events: List[Dict],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
    target_precision: float,
) -> List[Dict]:
    by_place: Dict[str, List[Dict]] = {}
    for ev in events:
        by_place.setdefault(ev["place"], []).append(ev)

    rows = []

    for plc, items in sorted(by_place.items()):
        y_true_event = np.array([ev["label"] for ev in items], dtype=int)
        y_pred_current = np.array([ev["pred"] for ev in items], dtype=int)
        current = compute_metrics(y_true_event, y_pred_current)

        y_frame_true, frame_scores_flat = flatten_frame_samples(items)
        frame_sweep = sweep_frame_scores(y_frame_true, frame_scores_flat)

        best_frame_f1 = best_threshold_by_f1(frame_sweep)
        target_precision_frame = best_threshold_by_min_precision(
            frame_sweep,
            min_precision=target_precision,
        )

        best_event = (
            event_metrics_at_frame_threshold(
                items,
                float(best_frame_f1["threshold"]),
                vote_rule,
                vote_ratio,
                min_positive_frames,
            )
            if best_frame_f1 else None
        )

        target_event = (
            event_metrics_at_frame_threshold(
                items,
                float(target_precision_frame["threshold"]),
                vote_rule,
                vote_ratio,
                min_positive_frames,
            )
            if target_precision_frame else None
        )

        n_normal = int(np.sum(y_true_event == 0))
        n_abnormal = int(np.sum(y_true_event == 1))

        all_frame_scores = []
        normal_frame_scores = []
        abnormal_frame_scores = []

        for ev in items:
            fs = ev.get("frame_scores", [])
            all_frame_scores.extend(fs)
            if int(ev["label"]) == 0:
                normal_frame_scores.extend(fs)
            else:
                abnormal_frame_scores.extend(fs)

        vote_scores = np.array([ev["vote_score"] for ev in items], dtype=float)

        row = {
            "place": plc,
            "n": int(len(items)),
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,

            # current event-level
            "current_tp": current["tp"],
            "current_tn": current["tn"],
            "current_fp": current["fp"],
            "current_fn": current["fn"],
            "current_accuracy": current["accuracy"],
            "current_precision": current["precision"],
            "current_recall": current["recall"],
            "current_specificity": current["specificity"],
            "current_f1": current["f1"],

            # frame score stats
            "frame_score_mean": float(np.mean(all_frame_scores)) if all_frame_scores else math.nan,
            "frame_score_std": float(np.std(all_frame_scores)) if all_frame_scores else math.nan,
            "normal_frame_score_mean": float(np.mean(normal_frame_scores)) if normal_frame_scores else math.nan,
            "abnormal_frame_score_mean": float(np.mean(abnormal_frame_scores)) if abnormal_frame_scores else math.nan,

            "vote_score_mean": float(np.mean(vote_scores)) if len(vote_scores) else math.nan,
            "vote_score_std": float(np.std(vote_scores)) if len(vote_scores) else math.nan,

            # best frame-F1 threshold: frame-level metrics
            "best_frame_f1_threshold": float(best_frame_f1["threshold"]) if best_frame_f1 else math.nan,
            "best_frame_f1_frame_precision": float(best_frame_f1["precision"]) if best_frame_f1 else math.nan,
            "best_frame_f1_frame_recall": float(best_frame_f1["recall"]) if best_frame_f1 else math.nan,
            "best_frame_f1_frame_f1": float(best_frame_f1["f1"]) if best_frame_f1 else math.nan,

            # best frame-F1 threshold applied to event-level vote
            "best_frame_f1_event_tp": int(best_event["tp"]) if best_event else -1,
            "best_frame_f1_event_tn": int(best_event["tn"]) if best_event else -1,
            "best_frame_f1_event_fp": int(best_event["fp"]) if best_event else -1,
            "best_frame_f1_event_fn": int(best_event["fn"]) if best_event else -1,
            "best_frame_f1_event_accuracy": float(best_event["accuracy"]) if best_event else math.nan,
            "best_frame_f1_event_precision": float(best_event["precision"]) if best_event else math.nan,
            "best_frame_f1_event_recall": float(best_event["recall"]) if best_event else math.nan,
            "best_frame_f1_event_f1": float(best_event["f1"]) if best_event else math.nan,

            # target precision threshold: frame-level metrics
            "target_precision": float(target_precision),
            "target_precision_threshold": float(target_precision_frame["threshold"]) if target_precision_frame else math.nan,
            "target_precision_frame_precision": float(target_precision_frame["precision"]) if target_precision_frame else math.nan,
            "target_precision_frame_recall": float(target_precision_frame["recall"]) if target_precision_frame else math.nan,
            "target_precision_frame_f1": float(target_precision_frame["f1"]) if target_precision_frame else math.nan,

            # target precision threshold applied to event-level vote
            "target_precision_event_tp": int(target_event["tp"]) if target_event else -1,
            "target_precision_event_tn": int(target_event["tn"]) if target_event else -1,
            "target_precision_event_fp": int(target_event["fp"]) if target_event else -1,
            "target_precision_event_fn": int(target_event["fn"]) if target_event else -1,
            "target_precision_event_accuracy": float(target_event["accuracy"]) if target_event else math.nan,
            "target_precision_event_precision": float(target_event["precision"]) if target_event else math.nan,
            "target_precision_event_recall": float(target_event["recall"]) if target_event else math.nan,
            "target_precision_event_f1": float(target_event["f1"]) if target_event else math.nan,
        }

        rows.append(row)

    return rows


def build_overall_summary(
    events: List[Dict],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
    target_precision: float,
) -> Dict:
    if len(events) == 0:
        return {}

    y_true_event = np.array([ev["label"] for ev in events], dtype=int)
    y_pred_current = np.array([ev["pred"] for ev in events], dtype=int)
    current_event = compute_metrics(y_true_event, y_pred_current)

    y_frame_true, frame_scores_flat = flatten_frame_samples(events)
    frame_sweep = sweep_frame_scores(y_frame_true, frame_scores_flat)

    best_frame_f1 = best_threshold_by_f1(frame_sweep)
    target_precision_frame = best_threshold_by_min_precision(
        frame_sweep,
        min_precision=target_precision,
    )

    best_event_from_frame_f1 = (
        event_metrics_at_frame_threshold(
            events,
            float(best_frame_f1["threshold"]),
            vote_rule,
            vote_ratio,
            min_positive_frames,
        )
        if best_frame_f1 else None
    )

    target_precision_event = (
        event_metrics_at_frame_threshold(
            events,
            float(target_precision_frame["threshold"]),
            vote_rule,
            vote_ratio,
            min_positive_frames,
        )
        if target_precision_frame else None
    )

    has_pos_frame = np.any(y_frame_true == 1)
    has_neg_frame = np.any(y_frame_true == 0)

    roc_auc = None
    pr_auc = None
    ap = None

    if len(frame_scores_flat) > 0 and has_pos_frame and has_neg_frame:
        fpr, tpr, _ = roc_curve_points(y_frame_true, frame_scores_flat)
        roc_auc = safe_auc(fpr, tpr)

    if len(frame_scores_flat) > 0 and has_pos_frame:
        recall, precision, _ = pr_curve_points(y_frame_true, frame_scores_flat)
        pr_auc = safe_auc(recall, precision)
        ap = average_precision_score_manual(y_frame_true, frame_scores_flat)

    vote_scores = np.array([ev["vote_score"] for ev in events], dtype=float)
    num_frames_set = sorted(set(int(ev.get("num_frames", 0)) for ev in events))

    return {
        "num_events": int(len(events)),
        "num_places": int(len(set(ev["place"] for ev in events))),
        "num_normal_events": int(np.sum(y_true_event == 0)),
        "num_abnormal_events": int(np.sum(y_true_event == 1)),
        "num_frame_samples": int(len(frame_scores_flat)),
        "num_normal_frame_samples": int(np.sum(y_frame_true == 0)) if len(y_frame_true) else 0,
        "num_abnormal_frame_samples": int(np.sum(y_frame_true == 1)) if len(y_frame_true) else 0,

        "curve_level": "frame_level",
        "threshold_selection_level": "frame_level",
        "final_metric_level": "event_level_after_vote",
        "threshold_type": "frame_score_threshold",
        "decision_rule": "frame_scores -> frame_threshold -> frame_flags -> vote -> event_pred",
        "vote_rule": vote_rule,
        "vote_ratio": float(vote_ratio),
        "min_positive_frames": int(min_positive_frames),
        "target_precision": float(target_precision),
        "num_frames_per_event_set": num_frames_set,

        # event-level current result from offline_eval_results.json pred
        "current_event_operating_point": current_event,

        # threshold selected by frame-level F1
        "best_frame_f1_operating_point": best_frame_f1,
        "best_frame_f1_event_operating_point": best_event_from_frame_f1,

        # threshold selected by frame-level target precision
        "target_precision_frame_operating_point": target_precision_frame,
        "target_precision_event_operating_point": target_precision_event,

        # frame-level curves
        "frame_level_roc_auc": roc_auc,
        "frame_level_pr_auc_trapezoid": pr_auc,
        "frame_level_average_precision": ap,

        # score distributions
        "frame_score_mean": float(np.mean(frame_scores_flat)) if len(frame_scores_flat) else math.nan,
        "frame_score_std": float(np.std(frame_scores_flat)) if len(frame_scores_flat) else math.nan,
        "frame_score_min": float(np.min(frame_scores_flat)) if len(frame_scores_flat) else math.nan,
        "frame_score_max": float(np.max(frame_scores_flat)) if len(frame_scores_flat) else math.nan,

        "vote_score_mean": float(np.mean(vote_scores)) if len(vote_scores) else math.nan,
        "vote_score_std": float(np.std(vote_scores)) if len(vote_scores) else math.nan,
        "vote_score_min": float(np.min(vote_scores)) if len(vote_scores) else math.nan,
        "vote_score_max": float(np.max(vote_scores)) if len(vote_scores) else math.nan,
    }


# =========================================================
# Plotting
# =========================================================

def plot_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
):
    if len(scores) == 0 or not (np.any(y_true == 1) and np.any(y_true == 0)):
        print("[WARN] ROC curve skipped: positive/negative class가 모두 있어야 함")
        return

    fpr, tpr, _ = roc_curve_points(y_true, scores)
    auc = safe_auc(fpr, tpr)

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    points = list(zip(fpr, tpr))
    points.append((0.0, 0.0))
    points.append((1.0, 1.0))

    points = sorted(set(points), key=lambda x: (x[0], x[1]))
    fpr = np.array([p[0] for p in points], dtype=float)
    tpr = np.array([p[1] for p in points], dtype=float)

    plt.figure(figsize=(6, 6))
    plt.step(
        fpr,
        tpr,
        where="post",
        linewidth=2.5,
        label=f"Frame-level ROC-AUC = {auc:.3f}" if auc is not None else "Frame-level ROC",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Random classifier",
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Frame-level ROC Curve", fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
):
    if len(scores) == 0 or not np.any(y_true == 1):
        print("[WARN] PR curve skipped: positive class가 없음")
        return

    recall, precision, _ = pr_curve_points(y_true, scores)
    pr_auc = safe_auc(recall, precision)
    ap = average_precision_score_manual(y_true, scores)

    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)

    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    merged = {}
    for r, p in zip(recall, precision):
        r_key = round(float(r), 10)
        merged[r_key] = max(float(p), merged.get(r_key, 0.0))

    recall = np.array(sorted(merged.keys()), dtype=float)
    precision = np.array([merged[round(float(r), 10)] for r in recall], dtype=float)

    if len(recall) == 0 or recall[0] > 0.0:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)

    label = "Frame-level PR curve"
    if ap is not None:
        label += f" / AP = {ap:.3f}"
    if pr_auc is not None:
        label += f" / AUC = {pr_auc:.3f}"

    plt.figure(figsize=(6, 6))
    plt.step(
        recall,
        precision,
        where="post",
        linewidth=2.5,
        label=label,
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Frame-level Precision-Recall Curve", fontsize=14)
    plt.legend(fontsize=10, loc="lower left")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_place_current_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["current_f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("Event-level F1")
    plt.title("Event-level F1 by Place - Current Operating Point")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_place_best_event_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["best_frame_f1_event_f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("Event-level F1")
    plt.title("Event-level F1 by Place - Best Frame-F1 Threshold Applied")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_place_target_precision_event_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["target_precision_event_f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("Event-level F1")
    plt.title("Event-level F1 by Place - Target Precision Frame Threshold Applied")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_vote_score_box_by_place(events: List[Dict], out_path: Path):
    by_place: Dict[str, List[float]] = {}

    for ev in events:
        by_place.setdefault(ev["place"], []).append(float(ev["vote_score"]))

    if len(by_place) == 0:
        return

    places = sorted(by_place.keys())
    data = [by_place[p] for p in places]

    plt.figure(figsize=(max(7, len(places) * 0.8), 5))
    plt.boxplot(data, labels=places, showmeans=True)
    plt.xlabel("Place")
    plt.ylabel("Vote Score")
    plt.title("Vote Score Distribution by Place")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_frame_score_hist_by_label(events: List[Dict], out_path: Path):
    normal_scores = []
    abnormal_scores = []

    for ev in events:
        if int(ev["label"]) == 0:
            normal_scores.extend(ev.get("frame_scores", []))
        else:
            abnormal_scores.extend(ev.get("frame_scores", []))

    plt.figure(figsize=(7, 5))

    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=20, alpha=0.6, label="normal event frames")

    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=20, alpha=0.6, label="abnormal event frames")

    plt.xlabel("Frame Score")
    plt.ylabel("Count")
    plt.title("Frame Score Histogram by Event Label")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_vote_score_hist_by_label(events: List[Dict], out_path: Path):
    normal_scores = [float(ev["vote_score"]) for ev in events if int(ev["label"]) == 0]
    abnormal_scores = [float(ev["vote_score"]) for ev in events if int(ev["label"]) == 1]

    plt.figure(figsize=(7, 5))

    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=20, alpha=0.6, label="normal events")

    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=20, alpha=0.6, label="abnormal events")

    plt.xlabel("Vote Score")
    plt.ylabel("Count")
    plt.title("Vote Score Histogram by Event Label")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_per_place_roc_pr(events: List[Dict], report_root: Path):
    curve_dir = report_root / "per_place_curves"
    ensure_dir(curve_dir)

    by_place: Dict[str, List[Dict]] = {}
    for ev in events:
        by_place.setdefault(ev["place"], []).append(ev)

    for plc, items in sorted(by_place.items()):
        y_frame_true, frame_scores_flat = flatten_frame_samples(items)

        if len(frame_scores_flat) == 0:
            print(f"[WARN] place={plc} curves skipped: no frame scores")
            continue

        if np.any(y_frame_true == 1) and np.any(y_frame_true == 0):
            plot_roc(y_frame_true, frame_scores_flat, curve_dir / f"{plc}_frame_roc.png")
        else:
            print(f"[WARN] place={plc} ROC skipped: one-class frame labels")

        if np.any(y_frame_true == 1):
            plot_pr(y_frame_true, frame_scores_flat, curve_dir / f"{plc}_frame_pr.png")
        else:
            print(f"[WARN] place={plc} PR skipped: no positive frame labels")


# =========================================================
# Report
# =========================================================

def _metric_line(prefix: str, m: Optional[Dict]) -> List[str]:
    if not m:
        return [f"- {prefix}: NA"]

    return [
        f"- {prefix} threshold: {float(m['threshold']):.6f}",
        f"- {prefix} TP / TN / FP / FN: {int(m['tp'])} / {int(m['tn'])} / {int(m['fp'])} / {int(m['fn'])}",
        f"- {prefix} Accuracy: {float(m['accuracy']):.4f}",
        f"- {prefix} Precision: {float(m['precision']):.4f}",
        f"- {prefix} Recall: {float(m['recall']):.4f}",
        f"- {prefix} F1: {float(m['f1']):.4f}",
    ]


def make_markdown_report(summary: Dict, place_stats: List[Dict], out_path: Path):
    if not summary:
        out_path.write_text("# Offline Evaluation Report\n\nNo events found.\n", encoding="utf-8")
        return

    cur = summary["current_event_operating_point"]
    best_frame = summary.get("best_frame_f1_operating_point") or {}
    best_event = summary.get("best_frame_f1_event_operating_point") or {}
    target_frame = summary.get("target_precision_frame_operating_point") or {}
    target_event = summary.get("target_precision_event_operating_point") or {}

    lines = []

    lines.append("# Offline Evaluation Report")
    lines.append("")
    lines.append("## Evaluation Definition")
    lines.append("")
    lines.append("- ROC/PR curve level: frame-level")
    lines.append("- Best threshold selection level: frame-level")
    lines.append("- Final metric level: event-level after vote")
    lines.append("- Frame label rule: each frame inherits its event label")
    lines.append("- Decision rule: frame_scores → frame_threshold → frame_flags → vote → event_pred")
    lines.append(f"- Vote rule: {summary['vote_rule']}")
    lines.append(f"- Vote ratio: {summary['vote_ratio']}")
    lines.append(f"- Min positive frames: {summary['min_positive_frames']}")
    lines.append(f"- Target precision for false-positive control: {summary['target_precision']}")
    lines.append(f"- Num frames per event set: {summary['num_frames_per_event_set']}")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Events: {summary['num_events']}")
    lines.append(f"- Places: {summary['num_places']}")
    lines.append(f"- Normal events: {summary['num_normal_events']}")
    lines.append(f"- Abnormal events: {summary['num_abnormal_events']}")
    lines.append(f"- Frame samples: {summary['num_frame_samples']}")
    lines.append(f"- Normal frame samples: {summary['num_normal_frame_samples']}")
    lines.append(f"- Abnormal frame samples: {summary['num_abnormal_frame_samples']}")
    lines.append("")
    lines.append("## Event-level Current Operating Point")
    lines.append("")
    lines.append(f"- TP / TN / FP / FN: {cur['tp']} / {cur['tn']} / {cur['fp']} / {cur['fn']}")
    lines.append(f"- Accuracy: {cur['accuracy']:.4f}")
    lines.append(f"- Precision: {cur['precision']:.4f}")
    lines.append(f"- Recall: {cur['recall']:.4f}")
    lines.append(f"- Specificity: {cur['specificity']:.4f}")
    lines.append(f"- F1: {cur['f1']:.4f}")
    lines.append("")

    lines.append("## Best Threshold by Frame-level F1")
    lines.append("")
    lines.extend(_metric_line("Frame-level best-F1", best_frame))
    lines.append("")
    lines.extend(_metric_line("Event-level after vote using best-F1 frame threshold", best_event))
    lines.append("")

    lines.append("## False-positive-control Threshold")
    lines.append("")
    lines.extend(_metric_line(f"Frame-level precision≥{summary['target_precision']:.2f}", target_frame))
    lines.append("")
    lines.extend(_metric_line(f"Event-level after vote using precision≥{summary['target_precision']:.2f} frame threshold", target_event))
    lines.append("")

    lines.append("## Frame-level Curves")
    lines.append("")
    lines.append(f"- Frame-level ROC-AUC: {summary['frame_level_roc_auc'] if summary['frame_level_roc_auc'] is not None else 'NA'}")
    lines.append(f"- Frame-level PR-AUC trapezoid: {summary['frame_level_pr_auc_trapezoid'] if summary['frame_level_pr_auc_trapezoid'] is not None else 'NA'}")
    lines.append(f"- Frame-level Average Precision: {summary['frame_level_average_precision'] if summary['frame_level_average_precision'] is not None else 'NA'}")
    lines.append("")

    lines.append("## Per-place Statistics")
    lines.append("")
    lines.append(
        "| Place | N | Normal | Abnormal | Current F1 | "
        "Best Frame Thr | Best Event TP/TN/FP/FN | Best Event F1 | "
        "TargetP Frame Thr | TargetP Event TP/TN/FP/FN | TargetP Event F1 |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"
    )

    for r in place_stats:
        best_counts = (
            f"{r['best_frame_f1_event_tp']}/"
            f"{r['best_frame_f1_event_tn']}/"
            f"{r['best_frame_f1_event_fp']}/"
            f"{r['best_frame_f1_event_fn']}"
        )
        target_counts = (
            f"{r['target_precision_event_tp']}/"
            f"{r['target_precision_event_tn']}/"
            f"{r['target_precision_event_fp']}/"
            f"{r['target_precision_event_fn']}"
        )

        lines.append(
            f"| {r['place']} "
            f"| {r['n']} "
            f"| {r['n_normal']} "
            f"| {r['n_abnormal']} "
            f"| {r['current_f1']:.4f} "
            f"| {r['best_frame_f1_threshold']:.6f} "
            f"| {best_counts} "
            f"| {r['best_frame_f1_event_f1']:.4f} "
            f"| {r['target_precision_threshold']:.6f} "
            f"| {target_counts} "
            f"| {r['target_precision_event_f1']:.4f} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_root", type=str, default="./recv/_offline_out")
    parser.add_argument("--report_root", type=str, default="./recv/_offline_report")
    parser.add_argument("--places", nargs="*", default=None)

    parser.add_argument(
        "--vote_rule",
        type=str,
        default="majority",
        choices=["any", "majority", "ratio", "min_count"],
        help="frame flags를 event pred로 바꾸는 vote rule",
    )
    parser.add_argument(
        "--vote_ratio",
        type=float,
        default=0.5,
        help="vote_rule=ratio일 때 사용하는 positive frame ratio",
    )
    parser.add_argument(
        "--min_positive_frames",
        type=int,
        default=1,
        help="vote_rule=min_count일 때 사용하는 최소 positive frame 수",
    )
    parser.add_argument(
        "--target_precision",
        type=float,
        default=0.95,
        help="오탐 방지용 frame-level precision lower bound",
    )

    args = parser.parse_args()

    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    ensure_dir(report_root)

    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] report_root={report_root}")
    print(f"[INFO] places={args.places}")
    print(f"[INFO] vote_rule={args.vote_rule}")
    print(f"[INFO] vote_ratio={args.vote_ratio}")
    print(f"[INFO] min_positive_frames={args.min_positive_frames}")
    print(f"[INFO] target_precision={args.target_precision}")

    events = collect_events(out_root, places=args.places)

    if len(events) == 0:
        print("[ERROR] no events found")
        return

    events = add_vote_scores(
        events=events,
        vote_rule=args.vote_rule,
        vote_ratio=args.vote_ratio,
        min_positive_frames=args.min_positive_frames,
    )

    # -----------------------------
    # Frame-level samples
    # -----------------------------
    frame_rows = make_frame_sample_rows(events)
    write_csv(report_root / "frame_level_samples.csv", frame_rows)

    # -----------------------------
    # Event-level table
    # -----------------------------
    write_csv(
        report_root / "event_level_results.csv",
        make_csv_safe_events(events),
    )

    # -----------------------------
    # Overall / place stats
    # -----------------------------
    place_stats = build_place_stats(
        events=events,
        vote_rule=args.vote_rule,
        vote_ratio=args.vote_ratio,
        min_positive_frames=args.min_positive_frames,
        target_precision=args.target_precision,
    )

    summary = build_overall_summary(
        events=events,
        vote_rule=args.vote_rule,
        vote_ratio=args.vote_ratio,
        min_positive_frames=args.min_positive_frames,
        target_precision=args.target_precision,
    )

    write_csv(report_root / "place_stats.csv", place_stats)
    dump_json(report_root / "summary.json", summary)

    # -----------------------------
    # Frame-level threshold sweep
    # -----------------------------
    y_frame_true, frame_scores_flat = flatten_frame_samples(events)
    frame_sweep_rows = sweep_frame_scores(y_frame_true, frame_scores_flat)
    write_csv(report_root / "threshold_sweep_frame_level.csv", frame_sweep_rows)

    # -----------------------------
    # Event-level threshold application tables
    # -----------------------------
    selected_rows = []

    best_frame = summary.get("best_frame_f1_operating_point")
    best_event = summary.get("best_frame_f1_event_operating_point")
    if best_frame and best_event:
        selected_rows.append({
            "name": "best_frame_f1",
            "threshold": float(best_frame["threshold"]),
            "selection_level": "frame",
            "selection_metric": "f1",
            "frame_precision": float(best_frame["precision"]),
            "frame_recall": float(best_frame["recall"]),
            "frame_f1": float(best_frame["f1"]),
            "event_tp": int(best_event["tp"]),
            "event_tn": int(best_event["tn"]),
            "event_fp": int(best_event["fp"]),
            "event_fn": int(best_event["fn"]),
            "event_accuracy": float(best_event["accuracy"]),
            "event_precision": float(best_event["precision"]),
            "event_recall": float(best_event["recall"]),
            "event_f1": float(best_event["f1"]),
        })

    target_frame = summary.get("target_precision_frame_operating_point")
    target_event = summary.get("target_precision_event_operating_point")
    if target_frame and target_event:
        selected_rows.append({
            "name": f"frame_precision_ge_{args.target_precision:.2f}",
            "threshold": float(target_frame["threshold"]),
            "selection_level": "frame",
            "selection_metric": f"precision>={args.target_precision:.2f}",
            "frame_precision": float(target_frame["precision"]),
            "frame_recall": float(target_frame["recall"]),
            "frame_f1": float(target_frame["f1"]),
            "event_tp": int(target_event["tp"]),
            "event_tn": int(target_event["tn"]),
            "event_fp": int(target_event["fp"]),
            "event_fn": int(target_event["fn"]),
            "event_accuracy": float(target_event["accuracy"]),
            "event_precision": float(target_event["precision"]),
            "event_recall": float(target_event["recall"]),
            "event_f1": float(target_event["f1"]),
        })

    write_csv(report_root / "selected_threshold_event_metrics.csv", selected_rows)

    # -----------------------------
    # Plots: frame-level ROC/PR
    # -----------------------------
    plot_roc(y_frame_true, frame_scores_flat, report_root / "frame_roc_curve_overall.png")
    plot_pr(y_frame_true, frame_scores_flat, report_root / "frame_pr_curve_overall.png")

    plot_place_current_f1(place_stats, report_root / "place_event_f1_current.png")
    plot_place_best_event_f1(place_stats, report_root / "place_event_f1_best_frame_threshold.png")
    plot_place_target_precision_event_f1(place_stats, report_root / "place_event_f1_target_precision_threshold.png")

    plot_vote_score_box_by_place(events, report_root / "vote_score_box_by_place.png")
    plot_frame_score_hist_by_label(events, report_root / "frame_score_hist_by_label.png")
    plot_vote_score_hist_by_label(events, report_root / "vote_score_hist_by_label.png")

    plot_per_place_roc_pr(events, report_root)

    # -----------------------------
    # Markdown report
    # -----------------------------
    make_markdown_report(summary, place_stats, report_root / "report.md")

    # -----------------------------
    # Console summary
    # -----------------------------
    cur = summary["current_event_operating_point"]
    best_frame = summary.get("best_frame_f1_operating_point") or {}
    best_event = summary.get("best_frame_f1_event_operating_point") or {}
    target_frame = summary.get("target_precision_frame_operating_point") or {}
    target_event = summary.get("target_precision_event_operating_point") or {}

    print("\n" + "=" * 70)
    print("[EVALUATION DEFINITION]")
    print("=" * 70)
    print("ROC/PR curve level        = frame_level")
    print("best threshold level      = frame_level")
    print("final metric level        = event_level_after_vote")
    print("frame label rule          = frame inherits event label")
    print("prediction                = frame_scores -> threshold -> vote -> event_pred")
    print(f"vote_rule                 = {summary['vote_rule']}")
    print(f"num_frames_set            = {summary['num_frames_per_event_set']}")
    print(f"target_precision          = {summary['target_precision']}")

    print("\n" + "=" * 70)
    print("[EVENT-LEVEL CURRENT OPERATING POINT]")
    print("=" * 70)
    print(f"N={cur['n']}")
    print(f"TP={cur['tp']} TN={cur['tn']} FP={cur['fp']} FN={cur['fn']}")
    print(f"Accuracy ={cur['accuracy']:.4f}")
    print(f"Precision={cur['precision']:.4f}")
    print(f"Recall   ={cur['recall']:.4f}")
    print(f"F1       ={cur['f1']:.4f}")

    print("\n" + "=" * 70)
    print("[BEST THRESHOLD BY FRAME-LEVEL F1]")
    print("=" * 70)

    if best_frame:
        print(f"frame_threshold={best_frame['threshold']:.6f}")
        print("[Frame-level]")
        print(f"TP={best_frame['tp']} TN={best_frame['tn']} FP={best_frame['fp']} FN={best_frame['fn']}")
        print(f"Precision={best_frame['precision']:.4f}")
        print(f"Recall   ={best_frame['recall']:.4f}")
        print(f"F1       ={best_frame['f1']:.4f}")

    if best_event:
        print("[Event-level after vote]")
        print(f"TP={best_event['tp']} TN={best_event['tn']} FP={best_event['fp']} FN={best_event['fn']}")
        print(f"Accuracy ={best_event['accuracy']:.4f}")
        print(f"Precision={best_event['precision']:.4f}")
        print(f"Recall   ={best_event['recall']:.4f}")
        print(f"F1       ={best_event['f1']:.4f}")
    else:
        print("NA")

    print("\n" + "=" * 70)
    print(f"[FALSE-POSITIVE-CONTROL THRESHOLD BY FRAME-LEVEL PRECISION >= {args.target_precision:.2f}]")
    print("=" * 70)

    if target_frame:
        print(f"frame_threshold={target_frame['threshold']:.6f}")
        print("[Frame-level]")
        print(f"TP={target_frame['tp']} TN={target_frame['tn']} FP={target_frame['fp']} FN={target_frame['fn']}")
        print(f"Precision={target_frame['precision']:.4f}")
        print(f"Recall   ={target_frame['recall']:.4f}")
        print(f"F1       ={target_frame['f1']:.4f}")

    if target_event:
        print("[Event-level after vote]")
        print(f"TP={target_event['tp']} TN={target_event['tn']} FP={target_event['fp']} FN={target_event['fn']}")
        print(f"Accuracy ={target_event['accuracy']:.4f}")
        print(f"Precision={target_event['precision']:.4f}")
        print(f"Recall   ={target_event['recall']:.4f}")
        print(f"F1       ={target_event['f1']:.4f}")
    else:
        print("NA")

    print("\n" + "=" * 70)
    print("[FRAME-LEVEL CURVES]")
    print("=" * 70)
    print(f"ROC AUC={summary['frame_level_roc_auc']}")
    print(f"PR AUC ={summary['frame_level_pr_auc_trapezoid']}")
    print(f"AP     ={summary['frame_level_average_precision']}")

    print("\n" + "=" * 70)
    print("[OUTPUT FILES]")
    print("=" * 70)
    print(f"- {report_root / 'frame_level_samples.csv'}")
    print(f"- {report_root / 'event_level_results.csv'}")
    print(f"- {report_root / 'place_stats.csv'}")
    print(f"- {report_root / 'threshold_sweep_frame_level.csv'}")
    print(f"- {report_root / 'selected_threshold_event_metrics.csv'}")
    print(f"- {report_root / 'summary.json'}")
    print(f"- {report_root / 'report.md'}")
    print(f"- {report_root / 'frame_roc_curve_overall.png'}")
    print(f"- {report_root / 'frame_pr_curve_overall.png'}")
    print(f"- {report_root / 'place_event_f1_current.png'}")
    print(f"- {report_root / 'place_event_f1_best_frame_threshold.png'}")
    print(f"- {report_root / 'place_event_f1_target_precision_threshold.png'}")
    print(f"- {report_root / 'vote_score_box_by_place.png'}")
    print(f"- {report_root / 'frame_score_hist_by_label.png'}")
    print(f"- {report_root / 'vote_score_hist_by_label.png'}")
    print(f"- {report_root / 'per_place_curves/'}")

    print("\n✅ offline report done")


if __name__ == "__main__":
    main()