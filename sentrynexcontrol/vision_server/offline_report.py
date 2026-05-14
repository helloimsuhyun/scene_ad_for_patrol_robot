#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline evaluation 부산물 기반 통계/곡선 리포트 생성 스크립트

핵심:
- event_score를 threshold와 직접 비교하지 않음.
- 각 event의 frame_scores를 frame threshold와 비교.
- threshold를 넘은 frame 개수가 vote 조건을 만족하면 event abnormal.
- PR/ROC/best threshold는 frame-score threshold 기준으로 sweep.
- 최종 성능지표는 event label vs vote-based event prediction 기준.
- 데모용 오탐 관리 operating point로 Precision >= 0.95 조건도 함께 계산.

전제:
먼저 Offline_eval.py를 실행해서 아래 구조가 있어야 함.

recv/_offline_out/
  01/
    offline_eval_results.json
  06/
    offline_eval_results.json
  ...

offline_eval_results.json 내부 이벤트 예:
{
  "event_key": "...",
  "label": 0 or 1,
  "pred": 0 or 1,
  "threshold": float,
  "frame_scores": [...],
  "frame_change_flags": [...],
  "event_score": float
}

사용 예시:
cd sentrynexcontrol

python -m vision_server.offline_report \
  --out_root ./recv/_offline_out \
  --report_root ./recv/val_offline_report \
  --places 01 06 07 08 \
  --vote_rule majority


python -m vision_server.offline_report \
  --out_root ./recv/_offline_out \
  --report_root ./recv/test_offline_report \
  --places P001 P002 P003 \
  --vote_rule majority
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

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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
    return {
        **c,
        **m,
        "n": int(len(y_true)),
    }


def safe_auc(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    return float(np.trapz(y_sorted, x_sorted))


def threshold_candidates(scores: np.ndarray) -> List[float]:
    if len(scores) == 0:
        return []

    unique = sorted(set(float(x) for x in scores), reverse=True)
    eps = 1e-9

    return [max(unique) + eps] + unique + [min(unique) - eps]


def best_threshold_by_f1(sweep_rows: List[Dict]) -> Optional[Dict]:
    if len(sweep_rows) == 0:
        return None

    return max(
        sweep_rows,
        key=lambda r: (
            float(r["f1"]),
            float(r["recall"]),
            float(r["precision"]),
        )
    )


def best_threshold_by_min_recall(
    sweep_rows: List[Dict],
    min_recall: float = 0.90,
) -> Optional[Dict]:
    candidates = [
        r for r in sweep_rows
        if float(r["recall"]) >= float(min_recall)
    ]

    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: (
            float(r["precision"]),
            float(r["f1"]),
            float(r["recall"]),
        )
    )


def best_threshold_by_min_precision(
    sweep_rows: List[Dict],
    min_precision: float = 0.95,
) -> Optional[Dict]:
    candidates = [
        r for r in sweep_rows
        if float(r["precision"]) >= float(min_precision)
    ]

    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: (
            float(r["recall"]),
            float(r["f1"]),
            float(r["precision"]),
        )
    )


def best_threshold_by_max_fp(
    sweep_rows: List[Dict],
    max_fp: int = 2,
) -> Optional[Dict]:
    candidates = [
        r for r in sweep_rows
        if int(r["fp"]) <= int(max_fp)
    ]

    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: (
            float(r["recall"]),
            float(r["precision"]),
            float(r["f1"]),
        )
    )


def average_precision_score_manual(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    sklearn 없이 AP 계산.
    event-level score 기준 AP.

    여기서 score는 event_score가 아니라 vote_score임.
    vote_score는 frame threshold + vote와 동등한 event-level decision score.
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
    event가 abnormal이 되기 위해 threshold 이상이어야 하는 frame 개수.

    vote_rule:
    - any: 1장 이상
    - majority: 절반 이상. 5장이면 3장.
    - ratio: ceil(num_frames * vote_ratio)
    - min_count: min_positive_frames장 이상
    """
    if num_frames <= 0:
        return 1

    if vote_rule == "any":
        return 1

    if vote_rule == "majority":
        return int(math.ceil(num_frames * 0.5))

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
    실제 판정 방식:
    frame_scores >= threshold인 frame 개수를 세고,
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

    num_positive = int(np.sum(np.array(frame_scores, dtype=float) >= float(threshold)))

    return 1 if num_positive >= required else 0


def vote_score_from_frame_scores(
    frame_scores: List[float],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> float:
    """
    frame threshold + vote와 수학적으로 동일한 event-level score를 만든다.

    예:
    frame_scores = [10, 25, 30, 5, 40]

    any vote:
      required=1
      vote_score=max=40
      threshold <= 40이면 abnormal

    majority vote, 5장:
      required=3
      내림차순 [40, 30, 25, 10, 5]
      3번째 큰 값=25
      threshold <= 25이면 3장 이상 threshold를 넘으므로 abnormal

    즉:
    event_pred = vote_score >= threshold
    는 frame_scores를 threshold와 비교한 뒤 vote하는 것과 동일함.
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
# Threshold sweep based on frame threshold + vote
# =========================================================

def sweep_vote_scores(
    y_true: np.ndarray,
    vote_scores: np.ndarray,
) -> List[Dict]:
    """
    vote_score >= threshold 로 event pred 계산.
    이 threshold는 실제로 frame_score threshold와 동일한 의미를 가짐.
    """
    rows = []

    for thr in threshold_candidates(vote_scores):
        y_pred = (vote_scores >= float(thr)).astype(int)
        met = compute_metrics(y_true, y_pred)

        rows.append({
            "threshold": float(thr),
            **met,
        })

    return rows


def roc_curve_points(
    y_true: np.ndarray,
    vote_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rows = sweep_vote_scores(y_true, vote_scores)

    fpr = np.array([r["fpr"] for r in rows], dtype=np.float32)
    tpr = np.array([r["recall"] for r in rows], dtype=np.float32)
    thrs = [float(r["threshold"]) for r in rows]

    order = np.argsort(fpr)

    return fpr[order], tpr[order], [thrs[i] for i in order]


def pr_curve_points(
    y_true: np.ndarray,
    vote_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rows = sweep_vote_scores(y_true, vote_scores)

    precision = np.array([r["precision"] for r in rows], dtype=np.float32)
    recall = np.array([r["recall"] for r in rows], dtype=np.float32)
    thrs = [float(r["threshold"]) for r in rows]

    order = np.argsort(recall)

    return recall[order], precision[order], [thrs[i] for i in order]


# =========================================================
# Data loading
# =========================================================

def collect_events(out_root: Path, places: Optional[List[str]] = None) -> List[Dict]:
    """
    recv/_offline_out/<place>/offline_eval_results.json 전체 수집.
    frame_scores를 반드시 보존한다.
    """
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

                # 기존 Offline_eval의 최종 pred.
                # current operating point 평가에 사용.
                "pred": pred,

                # 참고용. threshold sweep에는 사용하지 않음.
                "event_score": event_score,

                # 실제 threshold sweep에 사용.
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
) -> List[Dict]:
    by_place: Dict[str, List[Dict]] = {}
    for ev in events:
        by_place.setdefault(ev["place"], []).append(ev)

    rows = []

    for plc, items in sorted(by_place.items()):
        y_true = np.array([ev["label"] for ev in items], dtype=int)
        y_pred_current = np.array([ev["pred"] for ev in items], dtype=int)
        vote_scores = np.array([ev["vote_score"] for ev in items], dtype=float)

        current = compute_metrics(y_true, y_pred_current)

        sweep = sweep_vote_scores(y_true, vote_scores)
        best = best_threshold_by_f1(sweep)
        recall90 = best_threshold_by_min_recall(sweep, min_recall=0.90)
        precision80 = best_threshold_by_min_precision(sweep, min_precision=0.80)
        precision95 = best_threshold_by_min_precision(sweep, min_precision=0.95)
        fp2 = best_threshold_by_max_fp(sweep, max_fp=2)

        n_normal = int(np.sum(y_true == 0))
        n_abnormal = int(np.sum(y_true == 1))

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

        row = {
            "place": plc,
            "n": int(len(items)),
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,

            "tp": current["tp"],
            "tn": current["tn"],
            "fp": current["fp"],
            "fn": current["fn"],
            "accuracy": current["accuracy"],
            "precision": current["precision"],
            "recall": current["recall"],
            "specificity": current["specificity"],
            "f1": current["f1"],

            "frame_score_mean": float(np.mean(all_frame_scores)) if all_frame_scores else math.nan,
            "frame_score_std": float(np.std(all_frame_scores)) if all_frame_scores else math.nan,
            "normal_frame_score_mean": float(np.mean(normal_frame_scores)) if normal_frame_scores else math.nan,
            "abnormal_frame_score_mean": float(np.mean(abnormal_frame_scores)) if abnormal_frame_scores else math.nan,

            "vote_score_mean": float(np.mean(vote_scores)) if len(vote_scores) else math.nan,
            "vote_score_std": float(np.std(vote_scores)) if len(vote_scores) else math.nan,

            "best_f1_threshold": float(best["threshold"]) if best else math.nan,
            "best_f1": float(best["f1"]) if best else math.nan,
            "best_precision": float(best["precision"]) if best else math.nan,
            "best_recall": float(best["recall"]) if best else math.nan,
            "best_accuracy": float(best["accuracy"]) if best else math.nan,

            "recall90_threshold": float(recall90["threshold"]) if recall90 else math.nan,
            "recall90_precision": float(recall90["precision"]) if recall90 else math.nan,
            "recall90_recall": float(recall90["recall"]) if recall90 else math.nan,
            "recall90_f1": float(recall90["f1"]) if recall90 else math.nan,

            "precision80_threshold": float(precision80["threshold"]) if precision80 else math.nan,
            "precision80_precision": float(precision80["precision"]) if precision80 else math.nan,
            "precision80_recall": float(precision80["recall"]) if precision80 else math.nan,
            "precision80_f1": float(precision80["f1"]) if precision80 else math.nan,

            "precision95_threshold": float(precision95["threshold"]) if precision95 else math.nan,
            "precision95_precision": float(precision95["precision"]) if precision95 else math.nan,
            "precision95_recall": float(precision95["recall"]) if precision95 else math.nan,
            "precision95_f1": float(precision95["f1"]) if precision95 else math.nan,
            "precision95_accuracy": float(precision95["accuracy"]) if precision95 else math.nan,
            "precision95_fp": int(precision95["fp"]) if precision95 else -1,
            "precision95_fn": int(precision95["fn"]) if precision95 else -1,

            "fp2_threshold": float(fp2["threshold"]) if fp2 else math.nan,
            "fp2_precision": float(fp2["precision"]) if fp2 else math.nan,
            "fp2_recall": float(fp2["recall"]) if fp2 else math.nan,
            "fp2_f1": float(fp2["f1"]) if fp2 else math.nan,
            "fp2_fp": int(fp2["fp"]) if fp2 else -1,
            "fp2_fn": int(fp2["fn"]) if fp2 else -1,
        }

        rows.append(row)

    return rows


def build_overall_summary(
    events: List[Dict],
    vote_rule: str,
    vote_ratio: float,
    min_positive_frames: int,
) -> Dict:
    if len(events) == 0:
        return {}

    y_true = np.array([ev["label"] for ev in events], dtype=int)
    y_pred_current = np.array([ev["pred"] for ev in events], dtype=int)
    vote_scores = np.array([ev["vote_score"] for ev in events], dtype=float)

    current = compute_metrics(y_true, y_pred_current)

    sweep = sweep_vote_scores(y_true, vote_scores)
    best = best_threshold_by_f1(sweep)
    recall90 = best_threshold_by_min_recall(sweep, min_recall=0.90)
    precision80 = best_threshold_by_min_precision(sweep, min_precision=0.80)
    precision95 = best_threshold_by_min_precision(sweep, min_precision=0.95)
    fp2 = best_threshold_by_max_fp(sweep, max_fp=2)

    has_pos = np.any(y_true == 1)
    has_neg = np.any(y_true == 0)

    roc_auc = None
    pr_auc = None
    ap = None

    if has_pos and has_neg:
        fpr, tpr, _ = roc_curve_points(y_true, vote_scores)
        roc_auc = safe_auc(fpr, tpr)

    if has_pos:
        recall, precision, _ = pr_curve_points(y_true, vote_scores)
        pr_auc = safe_auc(recall, precision)
        ap = average_precision_score_manual(y_true, vote_scores)

    all_frame_scores = []
    for ev in events:
        all_frame_scores.extend(ev.get("frame_scores", []))

    num_frames_set = sorted(set(int(ev.get("num_frames", 0)) for ev in events))

    return {
        "num_events": int(len(events)),
        "num_places": int(len(set(ev["place"] for ev in events))),
        "num_normal": int(np.sum(y_true == 0)),
        "num_abnormal": int(np.sum(y_true == 1)),

        "threshold_type": "frame_score_threshold",
        "decision_rule": "frame_scores -> threshold -> frame_flags -> vote -> event_pred",
        "metric_level": "event_level",
        "vote_rule": vote_rule,
        "vote_ratio": float(vote_ratio),
        "min_positive_frames": int(min_positive_frames),
        "num_frames_per_event_set": num_frames_set,

        "current_operating_point": current,
        "best_f1_operating_point": best,
        "recall90_operating_point": recall90,
        "precision80_operating_point": precision80,
        "precision95_operating_point": precision95,
        "fp2_operating_point": fp2,

        "roc_auc": roc_auc,
        "pr_auc_trapezoid": pr_auc,
        "average_precision": ap,

        "frame_score_mean": float(np.mean(all_frame_scores)) if all_frame_scores else math.nan,
        "frame_score_std": float(np.std(all_frame_scores)) if all_frame_scores else math.nan,
        "frame_score_min": float(np.min(all_frame_scores)) if all_frame_scores else math.nan,
        "frame_score_max": float(np.max(all_frame_scores)) if all_frame_scores else math.nan,

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
    vote_scores: np.ndarray,
    out_path: Path,
):
    if not (np.any(y_true == 1) and np.any(y_true == 0)):
        print("[WARN] ROC curve skipped: positive/negative class가 모두 있어야 함")
        return

    fpr, tpr, _ = roc_curve_points(y_true, vote_scores)
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
        label=f"ROC-AUC = {auc:.3f}" if auc is not None else "ROC curve",
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
    plt.title("ROC Curve", fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr(
    y_true: np.ndarray,
    vote_scores: np.ndarray,
    out_path: Path,
):
    if not np.any(y_true == 1):
        print("[WARN] PR curve skipped: positive class가 없음")
        return

    recall, precision, _ = pr_curve_points(y_true, vote_scores)
    pr_auc = safe_auc(recall, precision)
    ap = average_precision_score_manual(y_true, vote_scores)

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

    label = "PR curve"
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
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(fontsize=10, loc="lower left")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_place_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("F1")
    plt.title("F1 by Place - Current Operating Point")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_place_best_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["best_f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("Best F1")
    plt.title("Best F1 by Place - Frame Threshold Sweep + Vote")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_place_precision95_f1(place_stats: List[Dict], out_path: Path):
    if len(place_stats) == 0:
        return

    places = [r["place"] for r in place_stats]
    f1s = [r["precision95_f1"] for r in place_stats]

    plt.figure(figsize=(max(7, len(places) * 0.8), 4))
    plt.bar(places, f1s)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Place")
    plt.ylabel("F1")
    plt.title("F1 by Place - Precision >= 0.95 Operating Point")
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
    normal_scores = [
        float(ev["vote_score"])
        for ev in events
        if int(ev["label"]) == 0
    ]
    abnormal_scores = [
        float(ev["vote_score"])
        for ev in events
        if int(ev["label"]) == 1
    ]

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
        y_true = np.array([ev["label"] for ev in items], dtype=int)
        vote_scores = np.array([ev["vote_score"] for ev in items], dtype=float)

        if np.any(y_true == 1) and np.any(y_true == 0):
            plot_roc(y_true, vote_scores, curve_dir / f"{plc}_roc.png")
        else:
            print(f"[WARN] place={plc} ROC skipped: one-class labels")

        if np.any(y_true == 1):
            plot_pr(y_true, vote_scores, curve_dir / f"{plc}_pr.png")
        else:
            print(f"[WARN] place={plc} PR skipped: no positive labels")


# =========================================================
# Report
# =========================================================

def make_markdown_report(summary: Dict, place_stats: List[Dict], out_path: Path):
    if not summary:
        out_path.write_text("# Offline Evaluation Report\n\nNo events found.\n", encoding="utf-8")
        return

    cur = summary["current_operating_point"]
    best = summary.get("best_f1_operating_point") or {}
    recall90 = summary.get("recall90_operating_point") or {}
    precision80 = summary.get("precision80_operating_point") or {}
    precision95 = summary.get("precision95_operating_point") or {}
    fp2 = summary.get("fp2_operating_point") or {}

    lines = []

    lines.append("# Offline Evaluation Report")
    lines.append("")
    lines.append("## Evaluation Definition")
    lines.append("")
    lines.append("- Threshold type: frame-score threshold")
    lines.append("- Prediction rule: frame_scores → threshold → frame_flags → vote → event_pred")
    lines.append("- Metric level: event-level label vs event-level prediction")
    lines.append(f"- Vote rule: {summary['vote_rule']}")
    lines.append(f"- Vote ratio: {summary['vote_ratio']}")
    lines.append(f"- Min positive frames: {summary['min_positive_frames']}")
    lines.append(f"- Num frames per event set: {summary['num_frames_per_event_set']}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Events: {summary['num_events']}")
    lines.append(f"- Places: {summary['num_places']}")
    lines.append(f"- Normal events: {summary['num_normal']}")
    lines.append(f"- Abnormal events: {summary['num_abnormal']}")
    lines.append("")
    lines.append("### Current Operating Point")
    lines.append("")
    lines.append(f"- TP / TN / FP / FN: {cur['tp']} / {cur['tn']} / {cur['fp']} / {cur['fn']}")
    lines.append(f"- Accuracy: {cur['accuracy']:.4f}")
    lines.append(f"- Precision: {cur['precision']:.4f}")
    lines.append(f"- Recall: {cur['recall']:.4f}")
    lines.append(f"- Specificity: {cur['specificity']:.4f}")
    lines.append(f"- F1: {cur['f1']:.4f}")
    lines.append("")
    lines.append("### Best-F1 Frame Threshold")
    lines.append("")

    if best:
        lines.append(f"- Best threshold: {float(best['threshold']):.6f}")
        lines.append(f"- TP / TN / FP / FN: {best['tp']} / {best['tn']} / {best['fp']} / {best['fn']}")
        lines.append(f"- Accuracy: {float(best['accuracy']):.4f}")
        lines.append(f"- Precision: {float(best['precision']):.4f}")
        lines.append(f"- Recall: {float(best['recall']):.4f}")
        lines.append(f"- F1: {float(best['f1']):.4f}")

    lines.append("")
    lines.append("### Optional Operating Points")
    lines.append("")

    if recall90:
        lines.append(
            f"- Recall≥0.90 threshold: {float(recall90['threshold']):.6f}, "
            f"P={float(recall90['precision']):.4f}, "
            f"R={float(recall90['recall']):.4f}, "
            f"F1={float(recall90['f1']):.4f}, "
            f"FP={int(recall90['fp'])}, "
            f"FN={int(recall90['fn'])}"
        )
    else:
        lines.append("- Recall≥0.90 threshold: NA")

    if precision80:
        lines.append(
            f"- Precision≥0.80 threshold: {float(precision80['threshold']):.6f}, "
            f"P={float(precision80['precision']):.4f}, "
            f"R={float(precision80['recall']):.4f}, "
            f"F1={float(precision80['f1']):.4f}, "
            f"FP={int(precision80['fp'])}, "
            f"FN={int(precision80['fn'])}"
        )
    else:
        lines.append("- Precision≥0.80 threshold: NA")

    if precision95:
        lines.append(
            f"- Precision≥0.95 threshold: {float(precision95['threshold']):.6f}, "
            f"P={float(precision95['precision']):.4f}, "
            f"R={float(precision95['recall']):.4f}, "
            f"F1={float(precision95['f1']):.4f}, "
            f"FP={int(precision95['fp'])}, "
            f"FN={int(precision95['fn'])}"
        )
    else:
        lines.append("- Precision≥0.95 threshold: NA")

    if fp2:
        lines.append(
            f"- FP≤2 threshold: {float(fp2['threshold']):.6f}, "
            f"P={float(fp2['precision']):.4f}, "
            f"R={float(fp2['recall']):.4f}, "
            f"F1={float(fp2['f1']):.4f}, "
            f"FP={int(fp2['fp'])}, "
            f"FN={int(fp2['fn'])}"
        )
    else:
        lines.append("- FP≤2 threshold: NA")

    lines.append("")
    lines.append("### Curves")
    lines.append("")
    lines.append(f"- ROC-AUC: {summary['roc_auc'] if summary['roc_auc'] is not None else 'NA'}")
    lines.append(f"- PR-AUC trapezoid: {summary['pr_auc_trapezoid'] if summary['pr_auc_trapezoid'] is not None else 'NA'}")
    lines.append(f"- Average Precision: {summary['average_precision'] if summary['average_precision'] is not None else 'NA'}")
    lines.append("")
    lines.append("## Per-place Statistics")
    lines.append("")
    lines.append(
        "| Place | N | Normal | Abnormal | TP | TN | FP | FN | "
        "Precision | Recall | F1 | Best F1 | Best Thr | "
        "Recall90 Thr | Precision80 Thr | Precision95 Thr | FP≤2 Thr |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in place_stats:
        lines.append(
            f"| {r['place']} "
            f"| {r['n']} "
            f"| {r['n_normal']} "
            f"| {r['n_abnormal']} "
            f"| {r['tp']} "
            f"| {r['tn']} "
            f"| {r['fp']} "
            f"| {r['fn']} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['best_f1']:.4f} "
            f"| {r['best_f1_threshold']:.6f} "
            f"| {r['recall90_threshold']:.6f} "
            f"| {r['precision80_threshold']:.6f} "
            f"| {r['precision95_threshold']:.6f} "
            f"| {r['fp2_threshold']:.6f} |"
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
    # Event-level table 저장
    # -----------------------------
    write_csv(
        report_root / "event_level_results.csv",
        make_csv_safe_events(events),
    )

    # -----------------------------
    # Overall / place stats
    # -----------------------------
    place_stats = build_place_stats(events)

    summary = build_overall_summary(
        events=events,
        vote_rule=args.vote_rule,
        vote_ratio=args.vote_ratio,
        min_positive_frames=args.min_positive_frames,
    )

    write_csv(report_root / "place_stats.csv", place_stats)
    dump_json(report_root / "summary.json", summary)

    # -----------------------------
    # Threshold sweep 저장
    # -----------------------------
    y_true = np.array([ev["label"] for ev in events], dtype=int)
    vote_scores = np.array([ev["vote_score"] for ev in events], dtype=float)

    sweep_rows = sweep_vote_scores(y_true, vote_scores)
    write_csv(report_root / "threshold_sweep_overall.csv", sweep_rows)

    # -----------------------------
    # Plots
    # -----------------------------
    plot_roc(y_true, vote_scores, report_root / "roc_curve_overall.png")
    plot_pr(y_true, vote_scores, report_root / "pr_curve_overall.png")

    plot_place_f1(place_stats, report_root / "place_f1_current.png")
    plot_place_best_f1(place_stats, report_root / "place_f1_best_threshold.png")
    plot_place_precision95_f1(place_stats, report_root / "place_f1_precision95.png")

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
    cur = summary["current_operating_point"]
    best = summary.get("best_f1_operating_point") or {}
    recall90 = summary.get("recall90_operating_point") or {}
    precision80 = summary.get("precision80_operating_point") or {}
    precision95 = summary.get("precision95_operating_point") or {}
    fp2 = summary.get("fp2_operating_point") or {}

    print("\n" + "=" * 60)
    print("[EVALUATION DEFINITION]")
    print("=" * 60)
    print("threshold_type = frame_score_threshold")
    print("prediction     = frame_scores -> threshold -> vote -> event_pred")
    print("metric_level   = event_level")
    print(f"vote_rule      = {summary['vote_rule']}")
    print(f"num_frames_set = {summary['num_frames_per_event_set']}")

    print("\n" + "=" * 60)
    print("[OVERALL CURRENT]")
    print("=" * 60)
    print(f"N={cur['n']}")
    print(f"TP={cur['tp']} TN={cur['tn']} FP={cur['fp']} FN={cur['fn']}")
    print(f"Accuracy ={cur['accuracy']:.4f}")
    print(f"Precision={cur['precision']:.4f}")
    print(f"Recall   ={cur['recall']:.4f}")
    print(f"F1       ={cur['f1']:.4f}")

    print("\n" + "=" * 60)
    print("[OVERALL BEST FRAME THRESHOLD BY F1]")
    print("=" * 60)

    if best:
        print(f"threshold={best['threshold']:.6f}")
        print(f"TP={best['tp']} TN={best['tn']} FP={best['fp']} FN={best['fn']}")
        print(f"Accuracy ={best['accuracy']:.4f}")
        print(f"Precision={best['precision']:.4f}")
        print(f"Recall   ={best['recall']:.4f}")
        print(f"F1       ={best['f1']:.4f}")
    else:
        print("NA")

    print("\n" + "=" * 60)
    print("[OPTIONAL OPERATING POINTS]")
    print("=" * 60)

    if recall90:
        print(
            f"Recall>=0.90: threshold={recall90['threshold']:.6f}, "
            f"P={recall90['precision']:.4f}, "
            f"R={recall90['recall']:.4f}, "
            f"F1={recall90['f1']:.4f}, "
            f"FP={recall90['fp']}, "
            f"FN={recall90['fn']}"
        )
    else:
        print("Recall>=0.90: NA")

    if precision80:
        print(
            f"Precision>=0.80: threshold={precision80['threshold']:.6f}, "
            f"P={precision80['precision']:.4f}, "
            f"R={precision80['recall']:.4f}, "
            f"F1={precision80['f1']:.4f}, "
            f"FP={precision80['fp']}, "
            f"FN={precision80['fn']}"
        )
    else:
        print("Precision>=0.80: NA")

    if precision95:
        print(
            f"Precision>=0.95: threshold={precision95['threshold']:.6f}, "
            f"P={precision95['precision']:.4f}, "
            f"R={precision95['recall']:.4f}, "
            f"F1={precision95['f1']:.4f}, "
            f"FP={precision95['fp']}, "
            f"FN={precision95['fn']}"
        )
    else:
        print("Precision>=0.95: NA")

    if fp2:
        print(
            f"FP<=2: threshold={fp2['threshold']:.6f}, "
            f"P={fp2['precision']:.4f}, "
            f"R={fp2['recall']:.4f}, "
            f"F1={fp2['f1']:.4f}, "
            f"FP={fp2['fp']}, "
            f"FN={fp2['fn']}"
        )
    else:
        print("FP<=2: NA")

    print("\n" + "=" * 60)
    print("[CURVES]")
    print("=" * 60)
    print(f"ROC AUC={summary['roc_auc']}")
    print(f"PR AUC ={summary['pr_auc_trapezoid']}")
    print(f"AP     ={summary['average_precision']}")

    print("\n" + "=" * 60)
    print("[OUTPUT FILES]")
    print("=" * 60)
    print(f"- {report_root / 'event_level_results.csv'}")
    print(f"- {report_root / 'place_stats.csv'}")
    print(f"- {report_root / 'threshold_sweep_overall.csv'}")
    print(f"- {report_root / 'summary.json'}")
    print(f"- {report_root / 'report.md'}")
    print(f"- {report_root / 'roc_curve_overall.png'}")
    print(f"- {report_root / 'pr_curve_overall.png'}")
    print(f"- {report_root / 'place_f1_current.png'}")
    print(f"- {report_root / 'place_f1_best_threshold.png'}")
    print(f"- {report_root / 'place_f1_precision95.png'}")
    print(f"- {report_root / 'vote_score_box_by_place.png'}")
    print(f"- {report_root / 'frame_score_hist_by_label.png'}")
    print(f"- {report_root / 'vote_score_hist_by_label.png'}")
    print(f"- {report_root / 'per_place_curves/'}")

    print("\n✅ offline report done")


if __name__ == "__main__":
    main()