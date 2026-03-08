# place.dict

from __future__ import annotations

import shutil
from pathlib import Path
import sqlite_db

VALID_MODES = {"idle", "bank", "th_calib", "query"}

BANK_TARGET = 70
TH_CALIB_TARGET = 30


# place 폴더 안 특정 mode 이미지 개수 계산
def count_images(save_root: Path, place_id: str, mode: str) -> int:
    d = save_root / str(place_id) / mode
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file())


# threshold.json 존재 여부
def has_threshold(save_root: Path, place_id: str) -> bool:
    th_path = save_root / str(place_id) / "threshold.json"
    return th_path.exists()

# place status dict 생성
def _build_place_status(row, save_root: Path):
    place_id = str(row["place_id"])
    bank_count = count_images(save_root, place_id, "bank")
    th_calib_count = count_images(save_root, place_id, "th_calib")

    return {
        "place_id": place_id,
        "mode": row["mode"],
        "need_calibration": bool(row["need_calibration"]),
        "bank_target": BANK_TARGET,
        "th_calib_target": TH_CALIB_TARGET,
        "bank_count": bank_count,
        "th_calib_count": th_calib_count,
        "threshold_ready": has_threshold(save_root, place_id),
        "ready_for_calibration": (
            bank_count >= BANK_TARGET and th_calib_count >= TH_CALIB_TARGET
        ),
        "updated_at": row["updated_at"],
    }


# 특정 place 현재 상태 + 수집 현황 요약 반환
def get_place_status(db, save_root: Path, place_id: str):
    row = sqlite_db.get_place(db, place_id)
    return _build_place_status(row, save_root)


# 전체 place 상태 + 수집 현황 리스트 반환
def list_place_status(db, save_root: Path):
    rows = sqlite_db.list_places(db)
    return [_build_place_status(row, save_root) for row in rows]


# place mode 변경
def set_place_mode(db, place_id: str, mode: str):
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode}")
    sqlite_db.set_place_mode(db, place_id, mode)


# need_calibration 상태 업데이트 함수
def set_need_calibration(db, place_id: str, need: bool):
    sqlite_db.set_place_need_calibration(db, place_id, need)


# 특정 place threshold 삭제
def delete_threshold(save_root: Path, place_id: str):
    th_path = save_root / str(place_id) / "threshold.json"
    if th_path.exists():
        th_path.unlink()


# 특정 place 데이터 + place row 삭제 - 파일도 다 지움
def delete_place(db, save_root: Path, place_id: str):
    place_dir = save_root / str(place_id)
    if place_dir.exists():
        shutil.rmtree(place_dir)
    sqlite_db.delete_place_row(db, place_id)


# 전체 place 데이터 + place row 전체 삭제 - 파일도 다 지움
def delete_all_places(db, save_root: Path):
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    sqlite_db.delete_all_place_rows(db)