from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

# 🔥 여기에 로봇 주소
ROBOT_URL = "http://192.168.0.10:8091"  


def compute_run_yolo(mode: int, has_active_regions: bool) -> bool:
    if mode == 0:
        return False
    if mode == 1:
        return True
    if mode == 2:
        return has_active_regions
    return False


async def get_enabled_regions_from_db(app, sqlite_db) -> List[Dict[str, Any]]:
    async with app.state.db_lock:
        rows = sqlite_db.list_yolo_regions(app.state.db, enabled_only=True)

    regions = []
    for r in rows:
        d = dict(r)
        regions.append({
            "region_id": d.get("region_id"),
            "name": d["name"],
            "x_min": min(float(d["x_min"]), float(d["x_max"])),
            "x_max": max(float(d["x_min"]), float(d["x_max"])),
            "y_min": min(float(d["y_min"]), float(d["y_max"])),
            "y_max": max(float(d["y_min"]), float(d["y_max"])),
            "is_enabled": bool(d.get("is_enabled", True)),
        })
    return regions


async def build_robot_yolo_config_payload(app, sqlite_db) -> Dict[str, Any]:
    yolo_mode = int(getattr(app.state, "yolo_mode", 0))

    regions = []
    has_active_regions = False

    if yolo_mode == 2:
        regions = await get_enabled_regions_from_db(app, sqlite_db)
        has_active_regions = len(regions) > 0

    return {
        "yolo_mode": yolo_mode,
        "run_yolo": compute_run_yolo(yolo_mode, has_active_regions),
        "regions": regions if yolo_mode == 2 else [],
        "updated_at": datetime.now().isoformat(),
    }


async def push_yolo_config_to_robot(app, sqlite_db) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    payload = await build_robot_yolo_config_payload(app, sqlite_db)

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(
                f"{ROBOT_URL.rstrip('/')}/robot/yolo_config",
                json=payload
            )
            resp.raise_for_status()
            return True, None, {
                "request_payload": payload,
                "robot_response": resp.json(),
            }
    except Exception as e:
        return False, str(e), {
            "request_payload": payload
        }