# http_server.py

""" 
>>>> events
------
1 row = 1개 query 이벤트

Columns
- event_id        : UUID (PRIMARY KEY)
- place_id        : place 고유 id
- captured_at     : 이벤트 캡처 시각
- anomaly_flag    : 모델 판정 결과 (0 = 정상, 1 = 비정상)

Optional fields
- anomaly_score   : 이벤트 이상 점수
- threshold_used  : 판정에 사용한 threshold
- ref_bank_id     : 사용된 reference bank 식별자
- ref_topk_json   : top-k reference 매칭 결과(JSON)
- summary_text    : 이벤트 요약 설명
- admin_checked   : 관리자 검토 여부 (0/1)
- admin_label     : 관리자 라벨
                    (예: false_positive, true_anomaly, NULL=미검토 또는 미지정)

Auto fields
- created_at      : DB 기록 시각


>>>> frames
------
1 row = 이벤트 내부 이미지 1장

Columns
- frame_id        : UUID (PRIMARY KEY)
- event_id        : 소속 이벤트 id (events.event_id와 연결)
- idx             : 이벤트 내부 프레임 순서
- image_path      : 저장된 이미지 경로
- frame_score     : 프레임별 이상 점수
- capture_time    : 해당 이미지 캡처 시각

Relationship
- events (1) -> frames (N)


>>>> places
------
1 row = 장소 1개

Columns
- place_id          : place 고유 id
- display_name      : GUI 표시용 장소 이름
- x                 : 지도/교시용 x 좌표
- y                 : 지도/교시용 y 좌표
- yaw               : 교시된 heading
- patrol_enabled    : 순찰 경로 포함 여부 (0/1)
- patrol_order      : 순찰 순서
- is_active         : 활성 장소 여부 (0/1)
- mode              : 현재 동작 모드
                      ('idle', 'bank', 'th_calib', 'query')
- need_calibration  : threshold 재계산 필요 여부 (0/1)
- updated_at        : 마지막 갱신 시각

Role
- place별 운영 상태를 관리
- mode로 로봇이 현재 어떤 데이터 수집/운영 모드인지 제어
- need_calibration으로 bank/th_calib 변경 이후 threshold 재계산 필요 여부 기록
- x, y, yaw / patrol_enabled / patrol_order로 순찰 경로와 지도 표시 제어

"""


import json
import asyncio
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np
import shutil
from uuid import uuid4

from pathlib import Path
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from . import sqlite_db
from . import dino_emb
from . import cnn_emb

from . import place_manager
from .distance import infer_event, calibrate_place
from .matcher import SuperGlueMatcher, SuperGlueMatchConfig
from .config import load_cfg

# path 
SAVE_ROOT = Path("./recv")  # 이미지저장 root
AUDIO_ROOT = Path("./recv_audio") # 오디오 저장 root
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("./recv/events.db") # db root

def sync_places_from_fs(db, save_root: Path):
    for p in save_root.iterdir():
        if not p.is_dir():
            continue

        place_id = p.name
        sqlite_db.ensure_place(db, place_id)

# 서버 시작 & 끝 관리 lifespan -------------------------------------------------------------
# ---------------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # 서버 시작시에 실행
    # db connect & init(없으면 만들어줌)
    app.state.db = sqlite_db.connect_db(DB_PATH)
    sqlite_db.init_db(app.state.db)
    app.state.db_lock = asyncio.Lock()

    sync_places_from_fs(app.state.db, SAVE_ROOT)

    # calibration state
    app.state.calib_lock = asyncio.Lock()
    app.state.global_calibrating = False
    app.state.calib_task = None

    #robot 현재 / 목표 position
    app.state.robot_pose = {
        "x": None,
        "y": None,
        "yaw": None,
        "status": "idle",
        "timestamp": None,
    }

    app.state.robot_goal = {
        "x": None,
        "y": None,
        "yaw": None,
        "next_place_id": None,
        "timestamp": None,
    }

    app.state.robot_command = {
        "command": "idle",
        "timestamp": None,
    }

    # GPU lock
    app.state.gpu_lock = asyncio.Lock()

    app.state.calib_progress = {
    "total": 0,
    "done": 0,
    "current_place_id": None,
    }

    #model load
    model, device = dino_emb.load_model()
    local_model, device = cnn_emb.load_model(
        model_name="resnet18",
        out_layer="layer3",
        device=device,
    )

    cfg = load_cfg(SAVE_ROOT)
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

    app.state.engine = {
        "global_model": model,
        "local_model": local_model,
        "device": device,
        "bank_root": SAVE_ROOT,
        "sg_matcher": sg_matcher,
    }

    print(" ------------Server startup complete")

    yield  # ------------- 

    # 서버 종료 시 (shutdown)
    print(" ------------ Server shutting down")
    app.state.db.close()
    print("DB closed")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(SAVE_ROOT)), name="images")
app.mount("/audio", StaticFiles(directory=str(AUDIO_ROOT)), name="audio")

# --------------------------------------------------------------------------------------------------------
# 추론함수 호출 ------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def run_inference_event(imgs_bgr, meta_obj, engine):
    place_id = str(meta_obj.get("place_id", "unknown"))
    return infer_event(
        imgs_bgr=imgs_bgr,
        bank_root=engine["bank_root"],
        plc_idx=place_id,
        global_model=engine["global_model"],
        local_model=engine["local_model"],
        device=engine["device"],
        sg_matcher=engine.get("sg_matcher"),
    )

#임계치 업데이트 함수 호출----------------------------------
def run_threshold_calibration(place_id, engine):
    thr, scores, _ = calibrate_place(
        bank_root=engine["bank_root"],
        plc_idx=place_id,
        global_model=engine["global_model"],
        local_model=engine["local_model"],
        device=engine["device"],
        sg_matcher=engine.get("sg_matcher"),
    )
    return thr

# ----------------------------------------------------------
# 이미지를 받은 경우 endpoint

@app.post("/place_imgs")
async def place_imgs(
    images: List[UploadFile] = File(...),  # 클라이언트에서 ("images", (...)) 반복
    meta: str = Form(...),                 # 클라이언트에서 ("meta", (None, json, "application/json"))
):
    # meta 파싱
    try:
        meta_obj = json.loads(meta)
    except Exception:
        raise HTTPException(status_code=400, detail="meta must be valid json string")
    
    #캘리브레이션 중이면 저장하지 않고 바로 RETURN
    if app.state.global_calibrating:
        return JSONResponse(
            {
                "ok": False,
                "status": "global_calibration_in_progress",
                "meta": meta_obj,
            },
            status_code=409,
        )

    place_id = str(meta_obj.get("place_id", "unknown")).strip()
    label = meta_obj.get("label", None)
    ts = meta_obj.get("timestamp") or datetime.now().isoformat()

    if not place_id:
        raise HTTPException(status_code=400, detail="place_id is required")

    async with app.state.db_lock:
        sqlite_db.ensure_place(app.state.db, place_id)
        place = sqlite_db.get_place(app.state.db, place_id)

    if place is None:
        raise HTTPException(status_code=404, detail="place not found")

    mode = str(place.get("mode", "idle")).strip()

    if mode == "idle":
        return JSONResponse(
            {
                "ok": False,
                "status": "place_idle",
                "place_id": place_id,
                "applied_mode": mode,
                "meta": meta_obj,
            },
            status_code=409,
        )

    if mode not in ("bank", "th_calib", "query"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid server-side mode for place {place_id}: {mode}"
        )

    # 저장 디렉토리: root / place / mode
    safe_ts = ts.replace(":", "-")
    out_dir = SAVE_ROOT / place_id / mode 
    out_dir.mkdir(parents=True, exist_ok=True)
    th_path = SAVE_ROOT / place_id / "threshold.json"


    if label is None:
        prefix = f"{place_id}_{mode}_{safe_ts}"
    else : 
        prefix = f"{label}_{place_id}_{mode}_{safe_ts}"

    #th가 존재하지 않는데 query면 바로 return
    if mode == "query" and not th_path.exists():
        return JSONResponse(
            {
                "ok": False,
                "status": "threshold_not_ready",
                "place_id": place_id,
                "meta": meta_obj,
            },
            status_code=409,
        )

    # 파일 저장
    saved = []
    imgs_bgr = [] 

    for i, uf in enumerate(images):
        ext = Path(uf.filename).suffix.lower() or ".jpg"
        out_path = out_dir / f"{prefix}_{i:03d}{ext}"

        data = await uf.read()
        out_path.write_bytes(data)
        saved.append(str(out_path))

        if mode == "query": #바로 메모리 끌고와서 추론할 준비
            rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
            bgr = rgb[:, :, ::-1].copy()  
            imgs_bgr.append(bgr)

    resp_status = "saved"

    if mode in ("bank", "th_calib"):
        # th.json 존재하는데, ref bank 업데이트하면 calibaration 필요로 db에 기록
        if th_path.exists():
            async with app.state.db_lock:
                place_manager.set_need_calibration(app.state.db,place_id, True)

    if mode == "query":
        #db에 event 생성 / frame 기록
        async with app.state.db_lock:
            #event
            event_id = sqlite_db.insert_event( 
                db=app.state.db,
                place_id=place_id,
                captured_at=ts,
            )
            #frame 1행 = 1개 frame
            sqlite_db.insert_frames(   
                db=app.state.db,
                event_id=event_id, 
                image_paths=saved,
                frame_scores=None,
                capture_times=ts, 
            )
        
        #추론 
        async with app.state.gpu_lock:
            result = await asyncio.to_thread(
                run_inference_event,
                imgs_bgr,
                meta_obj,
                app.state.engine
            )

        async with app.state.db_lock:
            sqlite_db.update_frame_scores(app.state.db, event_id, result["frame_scores"])
            sqlite_db.update_event_result(
                app.state.db,
                event_id=event_id,
                anomaly_flag=int(result["anomaly_flag"]),
                anomaly_score=float(result["event_score"]),
                threshold_used=float(result["threshold"]),
                ref_bank_id=result.get("ref_bank_id"),
                ref_topk_json=result.get("ref_topk_json"),
                summary_text=result.get("summary"),
            )

    return JSONResponse(
        {
            "ok": True,
            "status": resp_status,
            "place_id": place_id,
            "applied_mode": mode,
            "n_images": len(saved),
            "saved_dir": str(out_dir),
            "meta": meta_obj,
        }
    )


# -----------------------------------------------------------------------------------------------
# GUI place table 상태 조회 / 제어 API ( 주로 관리자 캘리브레이션 데이터 수집 사용 endpoint 모음)
# place 상태 조회 / mode 변경 

# 모든 place 상태 조회 (GUI 대시보드용)
@app.get("/places")
async def get_places():
    async with app.state.db_lock:
        places = place_manager.list_place_status(app.state.db, SAVE_ROOT)

    return {
        "ok": True,
        "global_calibrating": app.state.global_calibrating,
        "calib_progress": app.state.calib_progress,
        "places": places,
    }

# 특정 place 상태 조회
@app.get("/places/{place_id}")
async def get_place(place_id: str):
    async with app.state.db_lock:
        sqlite_db.ensure_place(app.state.db, place_id)
        place = place_manager.get_place_status(app.state.db, SAVE_ROOT, place_id)
    return {
        "ok": True,
        "global_calibrating": app.state.global_calibrating,
        "place": place,
    }

# GUI에서 place mode 변경 (bank / th_calib / query)
@app.post("/places/{place_id}/config")
async def set_place_config(
    place_id: str,
    mode: str = Form(...),
):
    try:
        async with app.state.db_lock:
            sqlite_db.ensure_place(app.state.db, place_id)
            place_manager.set_place_mode(app.state.db, place_id, mode)
            place = place_manager.get_place_status(app.state.db, SAVE_ROOT, place_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True, "place": place}



# 특정 place 전체 삭제
@app.delete("/places/{place_id}")
async def delete_place(place_id: str):
    async with app.state.db_lock:
        place_manager.delete_place(app.state.db, SAVE_ROOT, place_id)
    return {"ok": True, "place_id": place_id, "status": "place_deleted"}

# 전체 place 전체 삭제
@app.delete("/places")
async def delete_all_places():
    async with app.state.db_lock:
        place_manager.delete_all_places(app.state.db, SAVE_ROOT)
    return {"ok": True, "status": "all_places_deleted"}

# 특정 place threshold 삭제
@app.delete("/places/{place_id}/threshold")
async def delete_place_threshold(place_id: str):
    async with app.state.db_lock:
        place_manager.delete_threshold(SAVE_ROOT, place_id)
        place = place_manager.get_place_status(app.state.db, SAVE_ROOT, place_id)

    return {"ok": True, "place": place, "status": "threshold_deleted"}

# 캘리브레이션 start endpoint
@app.post("/places/recalibrate_all")
async def calibrate_all_places():
    async with app.state.calib_lock:
        if app.state.global_calibrating or app.state.calib_task is not None:
            return {"ok": False, "status": "already_calibrating_all"}

        app.state.global_calibrating = True
        app.state.calib_progress = {
            "total": 0,
            "done": 0,
            "current_place_id": None,
        }
        app.state.calib_task = asyncio.create_task(run_calibration_job(app))

    return {"ok": True, "status": "recalibration_started"}

# 현재 캘리브레이션 상태 endpoint
@app.get("/calibration_status")
async def get_calibration_status():
    return {
        "ok": True,
        "global_calibrating": app.state.global_calibrating,
        "calib_progress": app.state.calib_progress,
    }

async def run_calibration_job(app: FastAPI):

    try:
        async with app.state.db_lock:
            places = sqlite_db.list_places(app.state.db)

        print(f"[CALIB] n_places={len(places)}")
        print(f"[CALIB] places={places}")

        # progress reset
        app.state.calib_progress = {
            "total": len(places),
            "done": 0,
            "current_place_id": None,
        }

        # calibration
        for row in places:
            place_id = row["place_id"]
            app.state.calib_progress["current_place_id"] = place_id

            # 시작 상태 반영
            async with app.state.db_lock:
                place_manager.delete_threshold(SAVE_ROOT, place_id)

            try:
                async with app.state.gpu_lock:
                    await asyncio.to_thread(
                        run_threshold_calibration,
                        place_id,
                        app.state.engine,
                    )

                th_path = SAVE_ROOT / place_id / "threshold.json"
                ok = th_path.exists()

                if not ok:
                    print(f"[CALIB ERROR] place={place_id} threshold.json not created")

                async with app.state.db_lock:
                    place_manager.set_need_calibration(app.state.db, place_id, not ok)

            except Exception as e:
                async with app.state.db_lock:
                    place_manager.set_need_calibration(app.state.db, place_id, True)
                print(f"[CALIB ERROR] place={place_id} exc={e}")

            finally:
                app.state.calib_progress["done"] += 1

    finally:
        app.state.global_calibrating = False
        app.state.calib_progress["current_place_id"] = None
        app.state.calib_task = None


# ======================================================================= GUI Place table 제어 endpoint 모음 (이벤트 라벨링 / 순찰 순서)
# ( 관리자의 이벤트 라벨링 / display name 변경 / 특정 waypoint 순찰 여부 변경)

class UpdatePlaceNameReq(BaseModel):
    display_name: str

class UpdateEventLabelReq(BaseModel):
    admin_label: Optional[str] = None

class UpdatePatrolEnabledReq(BaseModel):
    patrol_enabled: bool

class ReorderPatrolReq(BaseModel):
    place_ids: List[str]

# event 관리자 라벨링 기능
@app.patch("/events/{event_id}/label")
async def update_event_label(event_id: str, req: UpdateEventLabelReq):
    async with app.state.db_lock:
        event = sqlite_db.get_event(app.state.db, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")

        sqlite_db.set_event_manual_label(app.state.db, event_id, req.admin_label)
        updated_event = sqlite_db.get_event(app.state.db, event_id)

    return {
        "ok": True,
        "event": updated_event,
    }

# display name을 변경
@app.patch("/places/{place_id}/display_name")
async def update_place_name(place_id: str, req: UpdatePlaceNameReq):
    async with app.state.db_lock:
        sqlite_db.set_place_display_name(app.state.db, place_id, req.display_name)
        place = sqlite_db.get_place(app.state.db, place_id)
    return {"ok": True, "place": place}


# place에 대해 순찰 여부를 변경 > 후에 로봇에게 전달할 db
@app.patch("/places/{place_id}/patrol_enabled")
async def update_patrol_enabled(place_id: str, req: UpdatePatrolEnabledReq):
    async with app.state.db_lock:
        sqlite_db.set_place_patrol_enabled(app.state.db, place_id, req.patrol_enabled)
        place = sqlite_db.get_place(app.state.db, place_id)
    return {"ok": True, "place": place}


# 순찰 순서를 reorder
@app.patch("/places/patrol_order")
async def update_patrol_order(req: ReorderPatrolReq):
    try:
        async with app.state.db_lock:
            sqlite_db.reorder_patrol_places(app.state.db, req.place_ids)
            places = sqlite_db.list_places(app.state.db, active_only=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ok": True,
        "places": places,
    }


# -----------------------------------------------------------------------------------------------
# GUI bank 이동 endpoint 

class MoveEventToBankReq(BaseModel):
    event_id: str
    place_id: Optional[str] = None

@app.post("/move_event")
async def move_event(req: MoveEventToBankReq):
    place_id = req.place_id
    event_id = req.event_id

    # event_id 와 place_id에 대응하는 event와 frames를 가져옴
    async with app.state.db_lock:
        event = sqlite_db.get_event(app.state.db, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")

        frames = sqlite_db.list_frames(app.state.db, event_id)
        if not frames:
            raise HTTPException(status_code=400, detail="event has no frames")

    event_place_id = str(event["place_id"])
    target_place_id = place_id or event_place_id

    # place 보장
    async with app.state.db_lock:
        sqlite_db.ensure_place(app.state.db, target_place_id)

    bank_dir = SAVE_ROOT / target_place_id / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)

    moved_files = []
    skipped_files = []

    for i, frame in enumerate(frames):
        src_path = Path(frame["image_path"])

        if not src_path.exists():
            skipped_files.append({
                "frame_id": frame["frame_id"],
                "reason": "source_not_found",
                "path": str(src_path),
            })
            continue

        ext = src_path.suffix.lower() or ".jpg"

        # bank용 새 파일명
        # 예: 00_bank_2026-03-11T12-30-00_xxxxx_000.jpg
        ts = (event["captured_at"] or datetime.now().isoformat()).replace(":", "-")
        new_name = f"{target_place_id}_bank_{ts}_{uuid4().hex[:8]}_{i:03d}{ext}"
        dst_path = bank_dir / new_name

        try:   
            shutil.copy2(str(src_path), str(dst_path))
            op = "copied"

            moved_files.append({
                "frame_id": frame["frame_id"],
                "src": str(src_path),
                "dst": str(dst_path),
                "op": op,
            })

        except Exception as e:
            skipped_files.append({
                "frame_id": frame["frame_id"],
                "reason": f"move_failed: {e}",
                "path": str(src_path),
            })

    # bank가 바뀌었으니 threshold 재계산 필요로 places db 변경
    async with app.state.db_lock:
        place_manager.set_need_calibration(app.state.db, target_place_id, True)
        sqlite_db.set_event_manual_label(app.state.db, event_id, "normal")
        place = place_manager.get_place_status(app.state.db, SAVE_ROOT, target_place_id)

    return {
        "ok": True,
        "event_id": event_id,
        "source_place_id": event_place_id,
        "target_place_id": target_place_id,
        "bank_dir": str(bank_dir),
        "n_total_frames": len(frames),
        "n_moved": len(moved_files),
        "n_skipped": len(skipped_files),
        "moved_files": moved_files,
        "skipped_files": skipped_files,
        "place": place,
    }


# ========================================================================================= vision 이상감지 이벤트 GUI 조회 / Pooling endpoint
# GUI event polling API

# 이벤트 polling 최근에 생긴 것
@app.get("/events")
async def get_events(since: Optional[str] = None, limit: int = 50):
    """
    Flutter polling용.
    - since가 없으면 최근 이벤트 목록 반환
    - since가 있으면 그 시각 이후의 이벤트만 반환
    응답 형식:
    {
        "ok": True,
        "events": [...]
    }
    """
    async with app.state.db_lock:
        cur = app.state.db.cursor()

        if since is None:
            cur.execute(
                """
                SELECT event_id, place_id, captured_at, anomaly_flag,
                    anomaly_score, threshold_used, ref_bank_id,
                    ref_topk_json, summary_text, admin_checked, admin_label, created_at
                FROM events
                ORDER BY captured_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cur.execute(
                """
                SELECT event_id, place_id, captured_at, anomaly_flag,
                    anomaly_score, threshold_used, ref_bank_id,
                    ref_topk_json, summary_text, admin_checked, admin_label, created_at
                FROM events
                WHERE captured_at > ?
                ORDER BY captured_at DESC
                LIMIT ?
                """,
                (since, limit),
            )

        rows = cur.fetchall()

    events = []
    for row in rows:
        event_id = row["event_id"]
        # Fetch frames for this event
        async with app.state.db_lock:
            frames = sqlite_db.list_frames(app.state.db, event_id)
            
        # 각 프레임의 경로를 /images/ 경로에서 접근 가능한 형태로 변환
        processed_frames = []
        for f in frames:
            f_dict = dict(f)
            raw_path = f_dict["image_path"]
            # 'recv/' 이후의 경로만 추출하여 상대 경로화
            if "recv/" in raw_path:
                f_dict["image_path"] = raw_path.split("recv/")[-1]
            else:
                f_dict["image_path"] = Path(raw_path).name
            processed_frames.append(f_dict)
            
        events.append({
            "event_id": event_id,
            "place_id": row["place_id"],
            "captured_at": row["captured_at"],
            "anomaly_flag": int(row["anomaly_flag"]) if row["anomaly_flag"] is not None else 0,
            "anomaly_score": float(row["anomaly_score"]) if row["anomaly_score"] is not None else None,
            "threshold_used": float(row["threshold_used"]) if row["threshold_used"] is not None else None,
            "ref_bank_id": row["ref_bank_id"],
            "ref_topk_json": row["ref_topk_json"],
            "summary_text": row["summary_text"],
            "admin_checked": int(row["admin_checked"]) if row["admin_checked"] is not None else 0,
            "admin_label": row["admin_label"],
            "created_at": row["created_at"],
            "frames": processed_frames,
        })

    return {
        "ok": True,
        "events": events,
    }

# 이벤트 조회 (db에 있는)
@app.get("/events/{event_id}")
async def get_event_detail(event_id: str):
    async with app.state.db_lock:
        event = sqlite_db.get_event(app.state.db, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")

        frames = sqlite_db.list_frames(app.state.db, event_id)

    return {
        "ok": True,
        "event": event,
        "frames": frames,
    }

#========================================================================================= robot 교시 endpoint
# --------------- 로봇 교시 및 waypoint 조회 -------------

class TeachPlaceReq(BaseModel):
    place_id: str
    x: float
    y: float
    yaw: float
    display_name: Optional[str] = None
    patrol_enabled: bool = True
    patrol_order: Optional[int] = None

#로봇 교시 유틸
@app.post("/robot/teach")
async def teach_place(req: TeachPlaceReq):
    async with app.state.db_lock:
        existing = sqlite_db.get_place(app.state.db, req.place_id)

        if req.patrol_order is not None and req.patrol_order < 0:
            raise HTTPException(status_code=400, detail="patrol_order must be >= 0")

        if req.patrol_order is not None:
            patrol_order = int(req.patrol_order)
        elif existing is not None and existing.get("patrol_order") is not None:
            # 이미 존재하는 place를 다시 teach하는 경우 기존 순서 유지
            patrol_order = int(existing["patrol_order"])
        else:
            # 새 place면 자동으로 맨 뒤에 append
            patrol_order = sqlite_db.get_next_patrol_order(app.state.db)

        sqlite_db.upsert_place_waypoint(
            db=app.state.db,
            place_id=req.place_id,
            x=req.x,
            y=req.y,
            yaw=req.yaw,
            display_name=req.display_name,
            patrol_enabled=req.patrol_enabled,
            patrol_order=patrol_order,
        )
        place = sqlite_db.get_place(app.state.db, req.place_id)

    return {
        "ok": True,
        "status": "place_taught",
        "place": place,
    }

# 로봇용 순찰 waypoint 가져가는 endpoint
@app.get("/robot/patrol_points")
async def get_patrol_points():
    async with app.state.db_lock:
        places = sqlite_db.list_patrol_places(app.state.db)

    return {
        "ok": True,
        "n_places": len(places),
        "places": places,
    } 

#========================================================================================= robot 실시간 위치 endpoint
# ----------------------------------------------------------------로봇 위치 표시 endpoint
# 로봇 현재 위치 표시용 endpoint

class RobotPoseReq(BaseModel):
    x: float
    y: float
    yaw: float
    status: Optional[str] = None   # 예: moving / idle / error
    timestamp: Optional[str] = None

class RobotGoalReq(BaseModel):
    x: float
    y: float
    yaw: float
    next_place_id: Optional[str] = None 
    timestamp: Optional[str] = None

# 로봇 -> 서버 : 현재 pose 업로드 endpoint ( 로봇쪽에서 주기적으로 송신 /robot_pose)
@app.post("/robot/pose")
async def update_robot_pose(req: RobotPoseReq):
    #print(
    #    f"[ROBOT POSE] x={req.x:.3f}, y={req.y:.3f}, yaw={req.yaw:.3f}, "
    #    f"status={req.status}",
    #    flush=True,
    #)
    app.state.robot_pose = {
        "x": req.x,
        "y": req.y,
        "yaw": req.yaw,
        "status": req.status or "moving",
        "timestamp": req.timestamp or datetime.now().isoformat(),
    }
    return {"ok": True}

# 로봇 -> 서버 : 목표 지점 업로드 endpoint ( 로봇에서 이벤트성으로 송신 ) : 로봇 nav2의 단기적인 목표 위치 /goal_pose_2d (x,y, yaw)와 로봇쪽에서 다음에 방문하고자 하는 place의 place_id를 송신
@app.post("/robot/goal")
async def update_robot_goal(req: RobotGoalReq):
    print(
        f"[GOAL POSE] x={req.x:.3f}, y={req.y:.3f}, yaw={req.yaw:.3f}, "
        f"[NEXT_PLACE_ID]={req.next_place_id}",
        flush=True,
    )
    app.state.robot_goal = {
        "x": req.x,
        "y": req.y,
        "yaw": req.yaw,
        "next_place_id": req.next_place_id,
        "timestamp": req.timestamp or datetime.now().isoformat(),
    }
    return {"ok": True}

# GUI -> 서버 : 현재 pose 조회
@app.get("/robot/pose")
async def get_robot_pose():
    return {
        "ok": True,
        "pose": app.state.robot_pose,
    }

# GUI -> 서버 : 로봇의 가장 최신 goal 조회
@app.get("/robot/goal")
async def get_robot_goal():
    return {
        "ok": True,
        "goal": app.state.robot_goal,
    }

# ------------------------------------ 서버 > 로봇 주행명령 endpoint
class RobotCommandReq(BaseModel):
    command: str
    timestamp: Optional[str] = None

# ------------------------------------ GUI > 서버 로봇 주행명령 endpoint
@app.post("/robot/command")
async def update_robot_command(req: RobotCommandReq):
    command = str(req.command).strip()
    if not command:
        raise HTTPException(status_code=400, detail="command must not be empty")

    app.state.robot_command = {
        "command": command,
        "timestamp": req.timestamp or datetime.now().isoformat(),
    }

    print(
        f"[ROBOT COMMAND] command={command}",
        flush=True,
    )

    return {
        "ok": True,
        "command": app.state.robot_command,
    }

#----------------------------------- 서버 > 로봇 주행명령 송신 endpoint
@app.get("/robot/command")
async def get_robot_command():
    return {
        "ok": True,
        "command": app.state.robot_command.get("command"),
        "timestamp": app.state.robot_command.get("timestamp"),
    }

#========================================================================================= audio endpoint
# ----------------------------------------------------------------오디오 이벤트 endpoint
# 오디오 이벤트를 받고, db에 넣는 endpoint
@app.post("/upload_audio")
async def upload_audio_event(
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    yaw: float = Form(...),
    timestamp: str = Form(...),
    doa: float = Form(...),
    model_label: Optional[str] = Form(None),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".wav", ".wave"]:
        raise HTTPException(status_code=400, detail="only wav file is allowed")

    audio_event_id = str(uuid4())
    safe_ts = timestamp.replace(":", "-")

    safe_label = "unlabeled" if model_label is None else str(model_label).strip()
    if not safe_label:
        safe_label = "unlabeled"
    safe_label = safe_label.replace("/", "_").replace(" ", "_")
    save_path = AUDIO_ROOT / f"{audio_event_id}_{safe_label}_{safe_ts}.wav"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")
    save_path.write_bytes(data)

    async with app.state.db_lock:
        sqlite_db.insert_audio_event(
            db=app.state.db,
            audio_event_id=audio_event_id,
            timestamp=timestamp,
            audio_path=str(save_path),
            x=x,
            y=y,
            yaw=yaw,
            doa=doa,
            model_label=model_label,
        )

    return {
        "ok": True,
        "audio_event_id": audio_event_id,
        "audio_path": str(save_path),
        "audio_url": f"/audio/{save_path.name}",
    }


# --------------------------------------------------------------------------- audio event GUI Pooling 및 조회 / 관리 endpoint
# audio event gui pooling

@app.get("/audio_events")
async def get_audio_events(
    since: Optional[str] = None,
    limit: int = 50,
    unchecked_only: bool = False,
):
    """
    since가 없으면 최근 오디오 이벤트 목록 반환
    since가 있으면 그 시각 이후의 오디오 이벤트만 반환
    unchecked_only=true 이면 admin_label이 없는 항목만 반환
    """
    async with app.state.db_lock:
        cur = app.state.db.cursor()

        if since is None:
            if unchecked_only:
                cur.execute(
                    """
                    SELECT *
                    FROM audio_events
                    WHERE admin_label IS NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM audio_events
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
        else:
            if unchecked_only:
                cur.execute(
                    """
                    SELECT *
                    FROM audio_events
                    WHERE created_at > ?
                      AND admin_label IS NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (since, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM audio_events
                    WHERE created_at > ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (since, limit),
                )

        rows = cur.fetchall()

    audio_events = []
    for row in rows:
        item = dict(row)
        item["audio_url"] = f"/audio/{Path(item['audio_path']).name}"
        audio_events.append(item)

    return {
        "ok": True,
        "audio_events": audio_events,
    }

#개별 event 조회
@app.get("/audio_events/{audio_event_id}")
async def get_audio_event_detail(audio_event_id: str):
    async with app.state.db_lock:
        row = sqlite_db.get_audio_event(app.state.db, audio_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="audio event not found")

    item = dict(row)
    item["audio_url"] = f"/audio/{Path(item['audio_path']).name}"

    return {
        "ok": True,
        "audio_event": item,
    }

# ----------------- 오디오 이벤트 라벨링
class UpdateAudioLabelReq(BaseModel):
    admin_label: Optional[str] = None

@app.patch("/audio_events/{audio_event_id}/label")
async def update_audio_event_label(audio_event_id: str, req: UpdateAudioLabelReq):
    async with app.state.db_lock:
        row = sqlite_db.get_audio_event(app.state.db, audio_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="audio event not found")

        sqlite_db.set_audio_event_admin_label(
            app.state.db,
            audio_event_id,
            req.admin_label,
        )
        updated = sqlite_db.get_audio_event(app.state.db, audio_event_id)

    return {
        "ok": True,
        "audio_event": updated,
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # 서버 - 로봇간 통신이면 ...