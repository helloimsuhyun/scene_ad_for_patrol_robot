# sweep_global_patch_with_aligned.py
# python sweep_global_patch_with_aligned.py \
#   --offline-eval /home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/vision_server/Offline_eval.py \
#   --bank-root /home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/vision_server/recv \
#   --places 06 07 \
#   --output-dir /home/choisuhyun/scene_ad_for_patrol_robot/sentrynexcontrol/vision_server/tune_runs \
#   --python-bin /home/choisuhyun/miniconda3/envs/dl/bin/python

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ------------------------------------------------------------
# 1) 고정 베이스 config
# ------------------------------------------------------------
BASE_CONFIG: Dict[str, Any] = {
    "embed": {
        "model_name": "dinov2_vits14",
        "img_size": 560,
        "global_mode": "patch_mean",
    },
    "repr": {
        "repr_mode": "global_patch_with_aligned",
    },
    "patchcore": {
        "top_p": 0.05,
        "preselect_m": 3,
        "radius": 1,
        "alpha": 0.6,
        "min_cut": 0.2,
        "singleton_weight": 0.25,
        "component_min_area": 2
    },
    "calib": {
        "k": 3,
        "method": "robust",
        "gaussian_k": 2.5,
        "robust_k": 2.0,
        "percentile": 97,
    },
    "infer": {
        "event_rule": "median",
        "use_two_stage_vlm": False,
    },
    "superglue": {
        "resize_long_side": 640,
        "weights": "indoor",
        "max_keypoints": 1024,
        "keypoint_threshold": 0.003,
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
        "min_matches": 10,
        "min_inliers": 15,
        "min_inlier_ratio": 0.2,
        "max_reproj_error": 5.0,
        "valid_patch_thr": 1.0,
        "mask_erode_kernel": 3,
        "mask_erode_iter": 1,
    },
}

# ------------------------------------------------------------
# 2) 탐색 공간
#    처음엔 너무 넓게 잡지 마라.
# ------------------------------------------------------------
SEARCH_SPACE = {
    "patchcore.top_p": [0.03, 0.05, 0.07],
    "patchcore.alpha": [0.5, 0.6, 0.7],
    "calib.robust_k": [1.5, 2.0, 2.5],

    "patchcore.min_cut": [0.15, 0.20, 0.25],
    "patchcore.singleton_weight": [0.10, 0.25],
}

# 원하면 2차 미세 탐색 때 추가:
# "calib.k": [1, 3, 5]


# ------------------------------------------------------------
# 3) 유틸
# ------------------------------------------------------------
def set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for k in dotted_key.split("."):
        cur = cur[k]
    return cur


def deep_copy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)


def stable_trial_name(params: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        parts.append(f"{k.replace('.', '_')}={v}")
    name = "__".join(parts)
    name = name.replace("/", "_")
    name = name.replace(" ", "")
    return name[:220]


def iter_grid(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# 4) metric summary
# ------------------------------------------------------------
@dataclass
class TrialMetrics:
    macro_f1: float
    macro_accuracy: float
    macro_precision: float
    macro_recall: float
    num_places: int
    valid: bool
    raw_path: str

    @staticmethod
    def invalid(raw_path: str) -> "TrialMetrics":
        return TrialMetrics(
            macro_f1=-1.0,
            macro_accuracy=-1.0,
            macro_precision=-1.0,
            macro_recall=-1.0,
            num_places=0,
            valid=False,
            raw_path=raw_path,
        )


def try_extract_place_metrics(place_obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    offline_eval_results.json 구조가 약간 달라도 버티도록 여러 키를 시도.
    기대값:
      accuracy, precision, recall, f1
    """
    candidates = [
        ("accuracy", "precision", "recall", "f1"),
        ("acc", "precision", "recall", "f1"),
        ("Accuracy", "Precision", "Recall", "F1"),
    ]

    for a, p, r, f in candidates:
        if all(k in place_obj for k in (a, p, r, f)):
            return (
                float(place_obj[a]),
                float(place_obj[p]),
                float(place_obj[r]),
                float(place_obj[f]),
            )

    # nested metrics
    if "metrics" in place_obj and isinstance(place_obj["metrics"], dict):
        m = place_obj["metrics"]
        for a, p, r, f in candidates:
            if all(k in m for k in (a, p, r, f)):
                return (
                    float(m[a]),
                    float(m[p]),
                    float(m[r]),
                    float(m[f]),
                )

    return None


def summarize_offline_eval_results(results_json_path: Path, target_places: Optional[List[str]] = None) -> TrialMetrics:
    if not results_json_path.exists():
        return TrialMetrics.invalid(str(results_json_path))

    data = load_json(results_json_path)

    # 가능한 구조들 대응
    place_records: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        if "places" in data and isinstance(data["places"], dict):
            for plc, obj in data["places"].items():
                if isinstance(obj, dict):
                    rec = {"place_id": plc, **obj}
                    place_records.append(rec)

        elif "results" in data and isinstance(data["results"], list):
            for obj in data["results"]:
                if isinstance(obj, dict):
                    place_records.append(obj)

        else:
            # 최상위 dict 자체가 place-id 맵일 가능성
            for k, v in data.items():
                if isinstance(v, dict):
                    rec = {"place_id": k, **v}
                    place_records.append(rec)

    elif isinstance(data, list):
        place_records = [x for x in data if isinstance(x, dict)]

    if target_places:
        target_set = set(target_places)
        place_records = [x for x in place_records if str(x.get("place_id", "")) in target_set]

    accs, precs, recs, f1s = [], [], [], []
    for obj in place_records:
        tup = try_extract_place_metrics(obj)
        if tup is None:
            continue
        acc, prec, rec, f1 = tup
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    if not f1s:
        return TrialMetrics.invalid(str(results_json_path))

    return TrialMetrics(
        macro_f1=sum(f1s) / len(f1s),
        macro_accuracy=sum(accs) / len(accs),
        macro_precision=sum(precs) / len(precs),
        macro_recall=sum(recs) / len(recs),
        num_places=len(f1s),
        valid=True,
        raw_path=str(results_json_path),
    )

def find_all_offline_eval_results(run_dir: Path) -> List[Path]:
    return sorted(run_dir.rglob("offline_eval_results.json"))


def summarize_multiple_offline_eval_results(result_paths: List[Path]) -> TrialMetrics:
    accs, precs, recs, f1s = [], [], [], []

    for p in result_paths:
        if not p.exists():
            continue
        data = load_json(p)
        tup = try_extract_place_metrics(data) if isinstance(data, dict) else None

        if tup is None and isinstance(data, dict):
            # 혹시 nested 구조면 기존 로직 재사용
            one = summarize_offline_eval_results(p, target_places=None)
            if one.valid:
                accs.append(one.macro_accuracy)
                precs.append(one.macro_precision)
                recs.append(one.macro_recall)
                f1s.append(one.macro_f1)
            continue

        if tup is not None:
            acc, prec, rec, f1 = tup
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

    if not f1s:
        return TrialMetrics.invalid("")

    return TrialMetrics(
        macro_f1=sum(f1s) / len(f1s),
        macro_accuracy=sum(accs) / len(accs),
        macro_precision=sum(precs) / len(precs),
        macro_recall=sum(recs) / len(recs),
        num_places=len(f1s),
        valid=True,
        raw_path="multiple_files",
    )

def objective_tuple(m: TrialMetrics) -> Tuple[float, float, float, float]:
    """
    큰 값이 좋다.
    1순위 F1, 이후 accuracy, precision, recall.
    """
    return (m.macro_f1, m.macro_accuracy, m.macro_precision, m.macro_recall)


# ------------------------------------------------------------
# 5) Offline_eval 실행
# ------------------------------------------------------------
def find_latest_offline_eval_results(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("offline_eval_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def run_one_trial(
    python_bin: str,
    offline_eval_py: str,
    bank_root: str,
    config: Dict[str, Any],
    places: Optional[List[str]],
    output_root: Path,
    trial_name: str,
    extra_args: Optional[List[str]] = None,
) -> Tuple[int, str, str, Optional[Path], Path]:
    trial_dir = output_root / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    config_path = trial_dir / "config.json"
    stdout_path = trial_dir / "stdout.txt"
    stderr_path = trial_dir / "stderr.txt"

    write_json(config_path, config)

    cmd = [
        python_bin,
        "-m", "sentrynexcontrol.vision_server.Offline_eval",
        "--bank_root", bank_root,
        "--config", str(config_path),
    ]

    if places:
        cmd += ["--places", *places]

    cmd += ["--output_dir", str(trial_dir)]

    if extra_args:
        cmd += extra_args

    with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(
            cmd,
            stdout=out_f,
            stderr=err_f,
            text=True,
            cwd="/home/choisuhyun/scene_ad_for_patrol_robot",
        )

    result_json = find_latest_offline_eval_results(trial_dir)
    return proc.returncode, str(stdout_path), str(stderr_path), result_json, trial_dir


# ------------------------------------------------------------
# 6) sweep
# ------------------------------------------------------------
def save_rows_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # 모든 key 합집합
    fields = set()
    for r in rows:
        fields.update(r.keys())
    fieldnames = sorted(fields)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def sweep(
    python_bin: str,
    offline_eval_py: str,
    bank_root: str,
    places: Optional[List[str]],
    output_dir: Path,
    search_space: Dict[str, List[Any]],
    base_config: Dict[str, Any],
    resume: bool = True,
    max_trials: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    best_row: Optional[Dict[str, Any]] = None
    trial_iter = iter_grid(search_space)

    for idx, params in enumerate(trial_iter, start=1):
        if max_trials is not None and idx > max_trials:
            break

        trial_name = f"{idx:04d}__{stable_trial_name(params)}"
        trial_dir = output_dir / trial_name
        result_json = trial_dir / "result_summary.json"

        if resume and result_json.exists():
            row = load_json(result_json)

            # 이미 만들어진 trial 폴더 안의 place별 결과 파일들 다시 검사
            all_result_jsons = find_all_offline_eval_results(trial_dir)

            # 1) 예전 요약이 invalid였더라도, place별 결과 파일이 있으면 다시 요약해서 복구
            if all_result_jsons:
                metrics = summarize_multiple_offline_eval_results(all_result_jsons)

                row["raw_result_json"] = [str(p) for p in all_result_jsons]
                row["valid"] = metrics.valid
                row["macro_f1"] = metrics.macro_f1
                row["macro_accuracy"] = metrics.macro_accuracy
                row["macro_precision"] = metrics.macro_precision
                row["macro_recall"] = metrics.macro_recall
                row["num_places"] = metrics.num_places
                row["objective"] = list(objective_tuple(metrics))

                write_json(result_json, row)

                rows.append(row)
                if best_row is None or tuple(row["objective"]) > tuple(best_row["objective"]):
                    best_row = row

                print(f"[RESUME-RECHECK] {trial_name} | valid={metrics.valid} | n={metrics.num_places}")
                continue

            # 2) place 결과 파일이 하나도 없으면 기존 result_summary가 valid일 때만 resume
            if row.get("valid", False):
                rows.append(row)
                if best_row is None or tuple(row["objective"]) > tuple(best_row["objective"]):
                    best_row = row
                print(f"[RESUME] {trial_name}")
                continue

            # 3) result_summary는 있는데 복구할 원본도 없고 invalid면 재실행
            print(f"[RETRY] {trial_name} | stale invalid summary without place results")

        cfg = deep_copy_config(base_config)
        for k, v in params.items():
            set_nested(cfg, k, v)

        t0 = time.time()
        returncode, stdout_path, stderr_path, raw_result_json, run_dir = run_one_trial(
            python_bin=python_bin,
            offline_eval_py=offline_eval_py,
            bank_root=bank_root,
            config=cfg,
            places=places,
            output_root=output_dir,
            trial_name=trial_name,
        )
        dt = time.time() - t0

        all_result_jsons = find_all_offline_eval_results(run_dir)

        if returncode != 0 or not all_result_jsons:
            metrics = TrialMetrics.invalid("" if raw_result_json is None else str(raw_result_json))
        else:
            metrics = summarize_multiple_offline_eval_results(all_result_jsons)

        row: Dict[str, Any] = {
            "trial_index": idx,
            "trial_name": trial_name,
            "returncode": returncode,
            "elapsed_sec": round(dt, 3),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "raw_result_json": metrics.raw_path,
            "valid": metrics.valid,
            "macro_f1": metrics.macro_f1,
            "macro_accuracy": metrics.macro_accuracy,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "num_places": metrics.num_places,
            "objective": list(objective_tuple(metrics)),
            "params": params,
            "config": cfg,
        }
        rows.append(row)
        write_json(result_json, row)

        if best_row is None or tuple(row["objective"]) > tuple(best_row["objective"]):
            best_row = row
            write_json(output_dir / "best_so_far.json", best_row)

        save_rows_csv(rows, summary_csv)
        write_json(summary_json, rows)

        print(
            f"[{idx}] F1={metrics.macro_f1:.4f} "
            f"Acc={metrics.macro_accuracy:.4f} "
            f"P={metrics.macro_precision:.4f} "
            f"R={metrics.macro_recall:.4f} "
            f"| params={params}"
        )

    # 최종 best
    if best_row is not None:
        write_json(output_dir / "best_final.json", best_row)

        best_cfg = best_row["config"]
        write_json(output_dir / "best_config.json", best_cfg)

        print("\n========== BEST ==========")
        print(json.dumps({
            "trial_name": best_row["trial_name"],
            "macro_f1": best_row["macro_f1"],
            "macro_accuracy": best_row["macro_accuracy"],
            "macro_precision": best_row["macro_precision"],
            "macro_recall": best_row["macro_recall"],
            "params": best_row["params"],
        }, ensure_ascii=False, indent=2))
    else:
        print("No valid trial found.")


# ------------------------------------------------------------
# 7) 2-stage search (선택)
# ------------------------------------------------------------
def build_refined_search_space(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    1차 grid 후 best 근처만 좁혀서 재탐색하고 싶을 때.
    """
    refined: Dict[str, List[Any]] = {}

    # float 계열
    top_p = float(best_params.get("patchcore.top_p", 0.05))
    refined["patchcore.top_p"] = sorted(set([
        max(0.01, round(top_p - 0.02, 3)),
        round(top_p, 3),
        min(0.20, round(top_p + 0.02, 3)),
    ]))

    alpha = float(best_params.get("patchcore.alpha", 0.6))
    refined["patchcore.alpha"] = sorted(set([
        max(0.3, round(alpha - 0.1, 2)),
        round(alpha, 2),
        min(0.9, round(alpha + 0.1, 2)),
    ]))

    robust_k = float(best_params.get("calib.robust_k", 2.0))
    refined["calib.robust_k"] = sorted(set([
        max(0.5, round(robust_k - 0.5, 2)),
        round(robust_k, 2),
        round(robust_k + 0.5, 2),
    ]))

    # int / categorical 계열
    preselect_m = int(best_params.get("patchcore.preselect_m", 3))
    refined["patchcore.preselect_m"] = sorted(set([
        max(1, preselect_m - 1),
        preselect_m,
        preselect_m + 1,
    ]))

    radius = int(best_params.get("patchcore.radius", 1))
    refined["patchcore.radius"] = sorted(set([
        max(0, radius - 1),
        radius,
        radius + 1,
    ]))

    percentile = int(best_params.get("calib.percentile", 97))
    refined["calib.percentile"] = sorted(set([
        max(90, percentile - 2),
        percentile,
        min(99, percentile + 2),
    ]))

    event_rule = str(best_params.get("infer.event_rule", "median"))
    refined["infer.event_rule"] = [event_rule]

    return refined


# ------------------------------------------------------------
# 8) CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--offline-eval", required=True, help="Offline_eval.py path")
    p.add_argument("--bank-root", required=True, help="recv or bank root path")
    p.add_argument("--places", nargs="*", default=None, help="target places, e.g. 06 07")
    p.add_argument("--output-dir", required=True, help="directory to save tuning outputs")
    p.add_argument("--python-bin", default=sys.executable, help="python executable")
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--two-stage", action="store_true", help="run coarse search then refined search")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)

    # 1차 coarse
    coarse_dir = output_dir / "stage1_coarse"
    sweep(
        python_bin=args.python_bin,
        offline_eval_py=args.offline_eval,
        bank_root=args.bank_root,
        places=args.places,
        output_dir=coarse_dir,
        search_space=SEARCH_SPACE,
        base_config=BASE_CONFIG,
        resume=not args.no_resume,
        max_trials=args.max_trials,
    )
    

    if args.two_stage:
        best_path = coarse_dir / "best_final.json"
        if not best_path.exists():
            print("stage1 best_final.json not found; skip stage2")
            return

        best_row = load_json(best_path)
        best_params = best_row["params"]
        refined = build_refined_search_space(best_params)

        stage2_dir = output_dir / "stage2_refined"
        sweep(
            python_bin=args.python_bin,
            offline_eval_py=args.offline_eval,
            bank_root=args.bank_root,
            places=args.places,
            output_dir=stage2_dir,
            search_space=refined,
            base_config=BASE_CONFIG,
            resume=not args.no_resume,
            max_trials=None,
        )


if __name__ == "__main__":
    main()