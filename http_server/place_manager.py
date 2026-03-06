# place_manager.py
from __future__ import annotations

import shutil
from pathlib import Path
import sqlite_db

VALID_MODES = {"idle", "bank", "th_calib", "query"}

BANK_TARGET = 150
TH_CALIB_TARGET = 40

# place 폴더 안 특정 mode 이미지 개수 계산
def count_images(save_root: Path, place_id: str, mode: str) -> int:
    d = save_root / str(place_id) / mode
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file())


# 특정 place 현재 상태 + 수집 현황 요약 반환
def get_place_status(db, save_root: Path, place_id: str):
    row = sqlite_db.get_place(db, place_id)
    return {
        "place_id": str(place_id),
        "mode": row["mode"],
        "bank_target": BANK_TARGET,
        "th_calib_target": TH_CALIB_TARGET,
        "bank_count": count_images(save_root, place_id, "bank"),
        "th_calib_count": count_images(save_root, place_id, "th_calib"),
        "query_count": count_images(save_root, place_id, "query"),
        "threshold_ready": bool(row["threshold_ready"]),
        "calibrating": bool(row["calibrating"]),
        "updated_at": row["updated_at"],
    }

# 전체 place 상태 + 수집 현황 리스트 반환 (list로)
def list_place_status(db, save_root: Path):
    rows = sqlite_db.list_places(db)
    results = []
    for row in rows:
        place_id = row["place_id"]
        results.append({
            "place_id": place_id,
            "mode": row["mode"],
            "bank_target": BANK_TARGET,
            "th_calib_target": TH_CALIB_TARGET,
            "bank_count": count_images(save_root, place_id, "bank"),
            "th_calib_count": count_images(save_root, place_id, "th_calib"),
            "query_count": count_images(save_root, place_id, "query"),
            "threshold_ready": bool(row["threshold_ready"]),
            "calibrating": bool(row["calibrating"]),
            "updated_at": row["updated_at"],
        })
    return results

# place mode 변경
def set_place_mode(db, place_id: str, mode: str):
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode}")
    sqlite_db.set_place_mode(db, place_id, mode)


# threshold ready 상태 설정
def set_threshold_ready(db, place_id: str, ready: bool):
    sqlite_db.set_place_threshold_ready(db, place_id, ready)


# calibration 진행 상태 설정
def set_calibrating(db, place_id: str, calibrating: bool):
    sqlite_db.set_place_calibrating(db, place_id, calibrating)


# bank/th_calib 데이터가 calibration 조건을 만족하는지 확인
def is_ready_for_calibration(save_root: Path, place_id: str) -> bool:
    bank_count = count_images(save_root, place_id, "bank")
    th_calib_count = count_images(save_root, place_id, "th_calib")
    return bank_count >= BANK_TARGET and th_calib_count >= TH_CALIB_TARGET


# 특정 place threshold 삭제 후 상태 초기화
def delete_threshold(db, save_root: Path, place_id: str):
    th_path = save_root / str(place_id) / "threshold.json"
    if th_path.exists():
        th_path.unlink()
    sqlite_db.set_place_threshold_ready(db, place_id, False)
    sqlite_db.set_place_calibrating(db, place_id, False)


# 특정 place 데이터 + place row 삭제
def delete_place(db, save_root: Path, place_id: str):
    place_dir = save_root / str(place_id)
    if place_dir.exists():
        shutil.rmtree(place_dir)
    sqlite_db.delete_place_row(db, place_id)


# 전체 place 데이터 + place row 전체 삭제
def delete_all_places(db, save_root: Path):
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    sqlite_db.delete_all_place_rows(db)