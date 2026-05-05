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
from .backbone_wrapper import build_local_backbone

from . import place_manager
from .distance import infer_event, calibrate_place
from .matcher import SuperGlueMatcher, SuperGlueMatchConfig
from .vpr_megaloc import MegaLocWrapper
from .yolo_server_util import (
    push_yolo_config_to_robot,
    push_audio_config_to_robot,
    push_all_region_configs_to_robot,
)


from .config import load_cfg

# path 
SAVE_ROOT = Path("./recv")  # 이미지저장 root
AUDIO_ROOT = Path("./recv_audio") # 오디오 저장 root
PERSON_EVENT_ROOT = Path("./recv_person") # person event 저장 root
AUTH_EVENT_ROOT = Path("./recv_auth") # 2차 인증 이벤트 저장 root

SAVE_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
PERSON_EVENT_ROOT.mkdir(parents=True, exist_ok=True)
AUTH_EVENT_ROOT.mkdir(parents=True, exist_ok=True)


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

    # GT
    app.state.query_capture_label = None

    # YOLO & audio mode
    app.state.yolo_mode = 2
    app.state.audio_mode = 2
    app.state.audio_allowed_labels = []

    # robot 현재 / 목표 position / 로봇 배터리
    app.state.robot_battery = None

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
    # inference queue
    app.state.inference_queue = asyncio.Queue()
    app.state.inference_worker_task = None

    app.state.calib_progress = {
    "total": 0,
    "done": 0,
    "current_place_id": None,
    }

    # global preselect model load - dino / megaloc
    model, device = dino_emb.load_model()
    vpr_model = MegaLocWrapper(device=device)

    cfg = load_cfg(SAVE_ROOT)
    if "superglue" not in cfg:
        raise RuntimeError("config에 'superglue' 섹션이 없습니다.")
    

    # local focus model load ---------------------
    cc_backbone = build_local_backbone("resnet18_layer3", img_size=560)
    verifier_backbone = build_local_backbone("resnet18_layer3", img_size=224)

    # superpoint & superglue load ----------------
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
        "cc_backbone": cc_backbone,
        "verifier_backbone": verifier_backbone,
        "device": device,
        "bank_root": SAVE_ROOT,
        "sg_matcher": sg_matcher,
        "vpr_model": vpr_model,
    }

    print(" ------------Server startup complete")

    asyncio.create_task(push_yolo_config_to_robot(app, sqlite_db))
    asyncio.create_task(push_audio_config_to_robot(app, sqlite_db))
    app.state.scheduler_task = asyncio.create_task(run_scheduler_daemon(app))
    
    app.state.inference_worker_task = asyncio.create_task(
        inference_worker_loop(app)
    )

    yield  # -------------

    # 서버 종료 시 (shutdown)
    print(" ------------ Server shutting down")

    # 1) background task 정리
    try:
        if getattr(app.state, "scheduler_task", None):
            app.state.scheduler_task.cancel()
            try:
                await app.state.scheduler_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"[SHUTDOWN WARN] scheduler_task cleanup failed: {e}")
    
    # 2) 이미지 추론 큐 정리
    try:
        if getattr(app.state, "inference_worker_task", None):
            app.state.inference_worker_task.cancel()
            try:
                await app.state.inference_worker_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"[SHUTDOWN WARN] inference_worker_task cleanup failed: {e}")

    # 3) calibration task 정리
    try:
        if getattr(app.state, "calib_task", None):
            app.state.calib_task.cancel()
            try:
                await app.state.calib_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        print(f"[SHUTDOWN WARN] calib_task cleanup failed: {e}")

    # 4) GPU 모델 참조 제거
    try:
        engine = getattr(app.state, "engine", None)

        if engine is not None:
            for key in [
                "global_model",
                "cc_backbone",
                "verifier_backbone",
                "sg_matcher",
                "vpr_model",
            ]:
                if key in engine:
                    engine[key] = None

            engine.clear()
            app.state.engine = None

    except Exception as e:
        print(f"[SHUTDOWN WARN] engine cleanup failed: {e}")

    # 5) Python object / CUDA cache 정리
    try:
        import gc
        gc.collect()

        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    except Exception as e:
        print(f"[SHUTDOWN WARN] cuda cleanup failed: {e}")

    # 6) DB close
    try:
        app.state.db.close()
        print("DB closed")
    except Exception as e:
        print(f"[SHUTDOWN WARN] DB close failed: {e}")

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
app.mount("/person_images", StaticFiles(directory=str(PERSON_EVENT_ROOT)), name="person_images")
app.mount("/auth_images", StaticFiles(directory=str(AUTH_EVENT_ROOT)), name="auth_images")

# --------------------------------------------------------------------------------------------------------
# 추론함수 호출 ------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def run_inference_event(imgs_bgr, meta_obj, engine):
    place_id = str(meta_obj.get("place_id", "unknown"))
    cfg = load_cfg(engine["bank_root"])

    return infer_event(
        imgs_bgr=imgs_bgr,
        bank_root=engine["bank_root"],
        plc_idx=place_id,
        global_model=engine["global_model"],
        cc_backbone=engine["cc_backbone"],
        verifier_backbone = engine["verifier_backbone"],
        device=engine["device"],
        sg_matcher=engine.get("sg_matcher"),
        cfg=cfg,
        vpr_model=engine.get("vpr_model"),
    )

def load_event_images_bgr(image_paths: List[str]):
    imgs_bgr = []

    for p in image_paths:
        path = Path(p)

        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")

        rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        imgs_bgr.append(bgr)

    return imgs_bgr

def is_cuda_fatal_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    fatal_keywords = [
        "cuda error",
        "unspecified launch failure",
        "illegal memory access",
        "device-side assert",
        "cublas",
        "cudnn",
    ]
    return any(k in msg for k in fatal_keywords)

async def inference_worker_loop(app: FastAPI):
    print("[INFERENCE WORKER] started", flush=True)

    while True:
        event_id = None

        try:
            event_id = await app.state.inference_queue.get()

            async with app.state.db_lock:
                event = sqlite_db.get_event(app.state.db, event_id)

                if event is None:
                    print(f"[INFERENCE WORKER] event not found: {event_id}", flush=True)
                    continue

                frames = sqlite_db.list_frames(app.state.db, event_id)
                image_paths = [f["image_path"] for f in frames]

                meta_obj = {
                    "place_id": event["place_id"],
                    "timestamp": event["captured_at"],
                    "event_id": event_id,
                }

            imgs_bgr = load_event_images_bgr(image_paths)

            async with app.state.gpu_lock:
                result = await asyncio.to_thread(
                    run_inference_event,
                    imgs_bgr,
                    meta_obj,
                    app.state.engine,
                )

            async with app.state.db_lock:
                sqlite_db.update_frame_scores(
                    app.state.db,
                    event_id,
                    result["frame_scores"],
                )

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

            print(f"[INFERENCE WORKER] done event_id={event_id}", flush=True)

        except asyncio.CancelledError:
            print("[INFERENCE WORKER] cancelled", flush=True)
            break

        except Exception as e:
            print(f"[INFERENCE WORKER ERROR] event_id={event_id}, exc={e}", flush=True)

            if event_id is not None:
                try:
                    async with app.state.db_lock:
                        sqlite_db.update_event_result(
                            app.state.db,
                            event_id=event_id,
                            anomaly_flag=-1,
                            anomaly_score=0.0,
                            threshold_used=0.0,
                            ref_bank_id=None,
                            ref_topk_json=None,
                            summary_text=f"inference_failed: {e}",
                        )
                except Exception as db_e:
                    print(f"[INFERENCE WORKER DB ERROR] {db_e}", flush=True)

        finally:
            if event_id is not None:
                try:
                    app.state.inference_queue.task_done()
                except ValueError:
                    pass


# 임계치 업데이트 함수 호출----------------------------------
def run_threshold_calibration(place_id, engine):
    cfg = load_cfg(engine["bank_root"])
                                  
    thr, scores, _ = calibrate_place(
        bank_root=engine["bank_root"],
        plc_idx=place_id,
        global_model=engine["global_model"],
        cc_backbone=engine["cc_backbone"],
        verifier_backbone = engine["verifier_backbone"],
        device=engine["device"],
        sg_matcher=engine.get("sg_matcher"),
        cfg=cfg,
        vpr_model=engine.get("vpr_model"),
    )
    return thr


# ======== label 관련 gui 함수
class UpdateQueryCaptureLabelReq(BaseModel):
    label: str

@app.post("/query_capture_label")
async def update_query_capture_label(req: UpdateQueryCaptureLabelReq):
    label = str(req.label).strip().lower()

    if label not in ("normal", "abnormal"):
        raise HTTPException(status_code=400, detail="label must be normal or abnormal")

    app.state.query_capture_label = label

    return {
        "ok": True,
        "query_capture_label": app.state.query_capture_label,
    }

@app.get("/query_capture_label")
async def get_query_capture_label():
    return {
        "ok": True,
        "query_capture_label": app.state.query_capture_label,
    }

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
    
    # 캘리브레이션 중이면 저장하지 않고 바로 RETURN
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

    if mode in ("bank", "th_calib"):
        label = "normal"
    elif mode == "query":
        label = app.state.query_capture_label
    else:
        label = None

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

    for i, uf in enumerate(images):
        ext = Path(uf.filename).suffix.lower() or ".jpg"
        out_path = out_dir / f"{prefix}_{i:03d}{ext}"

        data = await uf.read()
        out_path.write_bytes(data)
        saved.append(str(out_path))

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
        await app.state.inference_queue.put(event_id)

    return JSONResponse(
        {
            "ok": True,
            "status": "accepted",
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
async def get_events(since: Optional[str] = None, limit: int = 30):
    """
    Flutter polling용.
    - 프론트가 since를 보내도 서버는 since를 사용하지 않음
    - 항상 최근 이벤트 목록을 반환
    - 비동기 추론 결과 업데이트를 GUI에 반영하기 위한 구조
    응답 형식:
    {
        "ok": True,
        "events": [...]
    }
    """

    async with app.state.db_lock:
        cur = app.state.db.cursor()

        # 기존 since 기반 polling 로직
        # 비동기 추론 결과가 나중에 업데이트되는 구조에서는,
        # captured_at 기준 since 필터를 사용하면 이미 받은 이벤트의 업데이트를 놓칠 수 있음.
        #
        # if since is None:
        #     cur.execute(
        #         """
        #         SELECT event_id, place_id, captured_at, anomaly_flag,
        #             anomaly_score, threshold_used, ref_bank_id,
        #             ref_topk_json, summary_text, admin_checked, admin_label, created_at
        #         FROM events
        #         ORDER BY captured_at DESC
        #         LIMIT ?
        #         """,
        #         (limit,),
        #     )
        # else:
        #     cur.execute(
        #         """
        #         SELECT event_id, place_id, captured_at, anomaly_flag,
        #             anomaly_score, threshold_used, ref_bank_id,
        #             ref_topk_json, summary_text, admin_checked, admin_label, created_at
        #         FROM events
        #         WHERE captured_at > ?
        #         ORDER BY captured_at DESC
        #         LIMIT ?
        #         """,
        #         (since, limit),
        #     )

        # 현재 데모용 안정화 로직:
        # since를 무시하고 항상 최신 20개 이벤트를 반환한다.
        # 프론트는 event_id 기준으로 기존 이벤트를 교체하므로,
        # 추론 완료 후 업데이트된 이벤트도 GUI에 반영될 수 있다.
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

# ========================================================================================= robot 교시 endpoint
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
    """
    print(
        f"[ROBOT POSE] x={req.x:.3f}, y={req.y:.3f}, yaw={req.yaw:.3f}, "
        f"status={req.status}",
        flush=True,
    )
    """
    app.state.robot_pose = {
        "x": req.x,
        "y": req.y,
        "yaw": req.yaw,
        "status": req.status or "moving",
        "timestamp": req.timestamp or datetime.now().isoformat(),
    }
    return {"ok": True}

# 로봇 -> 서버 : 목표 지점 업로드 endpoint ( 로봇에서 이벤트성으로 송신 ) : 
# 로봇 nav2의 단기적인 목표 위치 /goal_pose_2d (x,y, yaw)와 로봇쪽에서 다음에 방문하고자 하는 place의 place_id를 송신
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
        source_region_id = None
        source_region_name = None

        if x is not None and y is not None:
            source_region_id, source_region_name = find_region_from_pose(
                app.state.db, x, y
            )

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
            source_region_id=source_region_id,
            source_region_name=source_region_name,
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

# ================================================= 순찰 프리셋 및 스케줄링 엔드포인트

class CreatePresetReq(BaseModel):
    name: str
    routes: List[str]

@app.get("/patrol/presets")
async def get_presets():
    async with app.state.db_lock:
        presets = sqlite_db.list_presets(app.state.db)
    return {"ok": True, "presets": presets}

@app.post("/patrol/presets")
async def add_preset(req: CreatePresetReq):
    async with app.state.db_lock:
        pid = sqlite_db.create_preset(app.state.db, req.name, json.dumps(req.routes))
    return {"ok": True, "preset_id": pid}

@app.delete("/patrol/presets/{preset_id}")
async def del_preset(preset_id: int):
    async with app.state.db_lock:
        sqlite_db.delete_preset(app.state.db, preset_id)
    return {"ok": True}

class CreateScheduleReq(BaseModel):
    preset_id: int
    time_str: str # "HH:MM"
    is_active: int = 1

@app.get("/patrol/schedules")
async def get_schedules():
    async with app.state.db_lock:
        schedules = sqlite_db.list_schedules(app.state.db)
    return {"ok": True, "schedules": schedules}

@app.post("/patrol/schedules")
async def add_schedule(req: CreateScheduleReq):
    async with app.state.db_lock:
        sid = sqlite_db.create_schedule(app.state.db, req.preset_id, req.time_str, req.is_active)
    return {"ok": True, "schedule_id": sid}

@app.delete("/patrol/schedules/{schedule_id}")
async def del_schedule(schedule_id: int):
    async with app.state.db_lock:
        sqlite_db.delete_schedule(app.state.db, schedule_id)
    return {"ok": True}

@app.patch("/patrol/schedules/{schedule_id}/toggle")
async def toggle_schedule(schedule_id: int):
    async with app.state.db_lock:
        sched = sqlite_db.get_schedule(app.state.db, schedule_id)
        if sched:
            new_val = 0 if sched['is_active'] == 1 else 1
            sqlite_db.update_schedule(app.state.db, schedule_id, sched['preset_id'], sched['time_str'], new_val)
    return {"ok": True}

# ----------------- Timeline Scheduler Daemon

async def _apply_preset_and_start(app: FastAPI, routes: List[str]):
    all_places = sqlite_db.list_places(app.state.db, active_only=True)
    for p in all_places:
        pid = p['place_id']
        if pid in routes:
            sqlite_db.set_place_patrol_enabled(app.state.db, pid, True)
            sqlite_db.set_place_patrol_order(app.state.db, pid, routes.index(pid))
        else:
            sqlite_db.set_place_patrol_enabled(app.state.db, pid, False)
    
    app.state.robot_command = {
        "command": "start",
        "timestamp": datetime.now().isoformat()
    }

async def run_scheduler_daemon(app: FastAPI):
    while True:
        try:
            now = datetime.now()
            now_str = now.strftime("%H:%M")
            async with app.state.db_lock:
                schedules = sqlite_db.list_schedules(app.state.db)
                matched = [s for s in schedules if s['is_active'] == 1 and s['time_str'] == now_str]
                if matched:
                    sched = matched[0]
                    preset = sqlite_db.get_preset(app.state.db, sched['preset_id'])
                    if preset:
                        routes = json.loads(preset['routes'])
                        await _apply_preset_and_start(app, routes)
                        print(f"[SCHEDULER] Auto-triggered patrol preset {preset['name']} at {now_str}")
            
            now = datetime.now()
            sleep_sec = 60.1 - now.second 
            await asyncio.sleep(sleep_sec)
        except asyncio.CancelledError:
            break

        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")
            await asyncio.sleep(60)


# =============================================================== YOLO 관련 이벤트 ( 이벤트 수신 / 풀링 / 라벨링 )
def find_region_from_pose(db, x, y):
    cur = db.cursor()
    rows = cur.execute(
        """
        SELECT * FROM yolo_regions
        WHERE is_enabled=1
        """
    ).fetchall()

    for r in rows:
        if (
            x >= r["x_min"] and x <= r["x_max"] and
            y >= r["y_min"] and y <= r["y_max"]
        ):
            return r["region_id"], r["name"]

    return None, None

@app.post("/person_event")
async def upload_person_event(
    image: UploadFile = File(...),
    event_json: str = Form(...),
):
    try:
        event_obj = json.loads(event_json)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid event_json")

    yolo_event_id = str(uuid4())
    ts = event_obj.get("event_time") or datetime.now().isoformat()
    safe_ts = ts.replace(":", "-")
    tracking_person_id = event_obj.get("person_id")   

    ext = Path(image.filename or "").suffix.lower() or ".jpg"
    save_path = PERSON_EVENT_ROOT / f"{safe_ts}_{yolo_event_id}{ext}"

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")

    save_path.write_bytes(data)

    pose = event_obj.get("robot_pose") or {}

    x = pose.get("x")
    y = pose.get("y")

    region_id = None
    region_name = None

    async with app.state.db_lock:
        if x is not None and y is not None:
            region_id, region_name = find_region_from_pose(app.state.db, x, y)

        sqlite_db.insert_yolo_event(
            db=app.state.db,
            yolo_event_id=yolo_event_id,
            timestamp=ts,
            image_path=str(save_path),
            tracking_person_id=str(tracking_person_id) if tracking_person_id is not None else None,  
            x=pose.get("x"),
            y=pose.get("y"),
            yaw=pose.get("yaw"),
            person_count=int(event_obj.get("num_persons", 0)),
            event_type=str(event_obj.get("event_type", "person_present")),
            source_region_id=region_id,
            source_region_name=region_name,
            dwell_time_sec=event_obj.get("dwell_time_sec"),
        )

    return {
        "ok": True,
        "yolo_event_id": yolo_event_id,
        "image_url": f"/person_images/{save_path.name}",
    }
    
@app.get("/yolo_events")
async def get_yolo_events(
    since: Optional[str] = None,
    limit: int = 50,
    unchecked_only: bool = False,
):
    async with app.state.db_lock:
        rows = sqlite_db.list_yolo_events(
            app.state.db,
            since=since,
            limit=limit,
            unchecked_only=unchecked_only,
        )

    out = []
    for r in rows:
        item = dict(r)
        if item.get("image_path"):
            item["image_url"] = f"/person_images/{Path(item['image_path']).name}"
        else:
            item["image_url"] = None
        out.append(item)

    return {"ok": True, "yolo_events": out}

# 이벤트 상세 조회
@app.get("/yolo_events/{yolo_event_id}")
async def get_yolo_event_detail(yolo_event_id: str):
    async with app.state.db_lock:
        row = sqlite_db.get_yolo_event(app.state.db, yolo_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")

    item = dict(row)
    if item.get("image_path"):
        item["image_url"] = f"/person_images/{Path(item['image_path']).name}"

    return {
        "ok": True,
        "yolo_event": item,
    }
    
    
class UpdateYoloEventLabelReq(BaseModel):
    admin_label: Optional[str] = None


@app.patch("/yolo_events/{yolo_event_id}/label")
async def update_yolo_event_label(yolo_event_id: str, req: UpdateYoloEventLabelReq):
    async with app.state.db_lock:
        row = sqlite_db.get_yolo_event(app.state.db, yolo_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")

        sqlite_db.set_yolo_event_admin_label(
            app.state.db,
            yolo_event_id,
            req.admin_label,
        )
        updated = sqlite_db.get_yolo_event(app.state.db, yolo_event_id)

    return {"ok": True, "yolo_event": updated}


# ======================================================= YOLO GUI 관련 구역제어, 모드 관리

# =========================
# Pydantic models
# =========================

class CreateYoloRegionReq(BaseModel):
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    is_enabled: bool = True


class UpdateYoloRegionReq(BaseModel):
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class UpdateYoloRegionEnabledReq(BaseModel):
    is_enabled: bool


# 현재 YOLO 모드 조회
@app.get("/robot/yolo_mode")
async def get_yolo_mode():
    return {"ok": True, "yolo_mode": int(getattr(app.state, "yolo_mode", 0))}


# 전체 구역 조회
@app.get("/robot/yolo_regions")
async def get_yolo_regions():
    async with app.state.db_lock:
        rows = sqlite_db.list_yolo_regions(app.state.db)
    return {"ok": True, "regions": [dict(r) for r in rows]}


# 활성 구역만 조회
@app.get("/robot/yolo_regions_enabled")
async def get_yolo_regions_enabled():
    async with app.state.db_lock:
        rows = sqlite_db.list_yolo_regions(app.state.db, enabled_only=True)
    return {"ok": True, "regions": [dict(r) for r in rows]}


# 구역 생성
@app.post("/robot/yolo_regions")
async def create_yolo_region(req: CreateYoloRegionReq):
    try:
        async with app.state.db_lock:
            rid = sqlite_db.insert_yolo_region(
                app.state.db,
                req.name,
                req.x_min,
                req.x_max,
                req.y_min,
                req.y_max,
                req.is_enabled,
            )
            row = sqlite_db.get_yolo_region(app.state.db, rid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await push_all_region_configs_to_robot(app, sqlite_db)

    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "region": dict(row)}


# 구역 수정
@app.patch("/robot/yolo_regions/{region_id}")
async def update_yolo_region(region_id: int, req: UpdateYoloRegionReq):
    try:
        async with app.state.db_lock:
            row = sqlite_db.get_yolo_region(app.state.db, region_id)
            if row is None:
                raise HTTPException(status_code=404, detail="not found")

            sqlite_db.update_yolo_region(
                app.state.db,
                region_id,
                req.name,
                req.x_min,
                req.x_max,
                req.y_min,
                req.y_max,
            )

            row = sqlite_db.get_yolo_region(app.state.db, region_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await push_all_region_configs_to_robot(app, sqlite_db)
    
    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "region": dict(row)}


# 개별 구역 활성/비활성
@app.patch("/robot/yolo_regions/{region_id}/enabled")
async def set_yolo_region_enabled(region_id: int, req: UpdateYoloRegionEnabledReq):
    async with app.state.db_lock:
        row = sqlite_db.get_yolo_region(app.state.db, region_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")

        sqlite_db.set_yolo_region_enabled(
            app.state.db,
            region_id,
            req.is_enabled,
        )

        row = sqlite_db.get_yolo_region(app.state.db, region_id)

    result = await push_all_region_configs_to_robot(app, sqlite_db)
    
    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "region": dict(row)}


# 전체 구역 활성/비활성
@app.patch("/robot/yolo_regions/enabled")
async def set_all_yolo_regions_enabled(req: UpdateYoloRegionEnabledReq):
    async with app.state.db_lock:
        affected = sqlite_db.set_all_yolo_regions_enabled(
            app.state.db,
            req.is_enabled,
        )

    result = await push_all_region_configs_to_robot(app, sqlite_db)
    
    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "affected": affected}


# 개별 구역 삭제
@app.delete("/robot/yolo_regions/{region_id}")
async def delete_yolo_region(region_id: int):
    async with app.state.db_lock:
        row = sqlite_db.get_yolo_region(app.state.db, region_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")

        sqlite_db.delete_yolo_region(app.state.db, region_id)

    result = await push_all_region_configs_to_robot(app, sqlite_db)
    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "region_id": region_id}


# 전체 구역 삭제
@app.delete("/robot/yolo_regions")
async def delete_all_yolo_regions():
    async with app.state.db_lock:
        affected = sqlite_db.delete_all_yolo_regions(app.state.db)

    result = await push_all_region_configs_to_robot(app, sqlite_db)
    
    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {"ok": True, "deleted_count": affected}


# YOLO MODE 변경 ===================================================
class UpdateYoloModeReq(BaseModel):
    yolo_mode: int

@app.patch("/robot/yolo_mode")
async def set_yolo_mode(req: UpdateYoloModeReq):
    mode = int(req.yolo_mode)
    if mode not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="invalid mode")

    app.state.yolo_mode = mode
    app.state.audio_mode = mode

    result = await push_all_region_configs_to_robot(app, sqlite_db)

    if not result["yolo_ok"]:
        print("[YOLO PUSH FAIL]", result["yolo_err"])

    if not result["audio_ok"]:
        print("[AUDIO PUSH FAIL]", result["audio_err"])

    return {
        "ok": True,
        "yolo_mode": app.state.yolo_mode,
    }

# 오디오 모드 변경 / 라벨 변경 =========================================
"""
0 → 오디오 업로드 OFF
1 → 항상 업로드 (기존 동작)
2 → region 조건 있을 때만 업로드
"""

class UpdateAudioModeReq(BaseModel):
    audio_mode: int


# GUI > 서버 ( 현재 오디오 모드 조회 )
@app.get("/robot/audio_mode")
async def get_audio_mode():
    return {
        "ok": True,
        "audio_mode": int(getattr(app.state, "audio_mode", 1)),
    }

# 서버 > 로봇 오디오 모드 변경
@app.patch("/robot/audio_mode")
async def set_audio_mode(req: UpdateAudioModeReq):
    mode = int(req.audio_mode)
    if mode not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="invalid mode")

    app.state.audio_mode = mode

    ok, err, _ = await push_audio_config_to_robot(app, sqlite_db)
    if not ok:
        print("[AUDIO PUSH FAIL]", err)

    return {
        "ok": True,
        "audio_mode": app.state.audio_mode,
    }


class UpdateAudioAllowedLabelsReq(BaseModel):
    allowed_labels: List[str] = []

"""
현재는 아래의 라벨 모두를 보냄 초기값 비어잇는 LIST 인 경우는 아래 라벨을 다 보냄
- speech
- impact
- alarm
- scream
"""

# 현재 허용된 소리 이벤트 종류 조회
@app.get("/robot/audio_allowed_labels")
async def get_audio_allowed_labels():
    return {
        "ok": True,
        "allowed_labels": list(getattr(app.state, "audio_allowed_labels", [])),
    }

# 라벨 필터 설정 API
@app.patch("/robot/audio_allowed_labels")
async def set_audio_allowed_labels(req: UpdateAudioAllowedLabelsReq):
    labels = [
        str(x).strip()
        for x in req.allowed_labels
        if str(x).strip()
    ]
    labels = sorted(set(labels))

    app.state.audio_allowed_labels = labels

    ok, err, _ = await push_audio_config_to_robot(app, sqlite_db)
    if not ok:
        print("[AUDIO PUSH FAIL]", err)

    return {
        "ok": True,
        "allowed_labels": app.state.audio_allowed_labels,
    }

# ============================================================== 


# ========================= [TEST/SIMULATOR ENDPOINTS] =========================

@app.post("/test/create_event")
async def create_mock_vision_event(payload: dict):
    from datetime import datetime
    eid = sqlite_db._uuid()
    place_id = payload.get("place_id", "00_test")
    now = datetime.now().isoformat()
    async with app.state.db_lock:
        sqlite_db.insert_event(
            app.state.db,
            eid,
            place_id,
            now,
            payload.get("anomaly_flag", 1),
            summary_text=payload.get("summary_text", "[테스트] 카메라 강제 진동 감지")
        )
        # 더미 프레임 추가 (프론트엔드에서 프레임 목록이 비어있으면 표시 안 될 수 있음)
        sqlite_db.insert_frames(
            app.state.db,
            eid,
            ["recv/00_test/query/dummy.jpg"], # placeholder
            capture_times=now
        )
    return {"ok": True, "event_id": eid}

@app.post("/test/create_audio_event")
async def create_mock_audio_event():
    from datetime import datetime
    now = datetime.now()
    async with app.state.db_lock:
        sqlite_db.insert_audio_event(
            app.state.db,
            now.isoformat(),
            "", # audio_path
            model_label="glass_breaking", # 유리 깨지는 소리
            x=0.5, y=0.5 # mock coordinates
        )
    return {"ok": True}

@app.post("/test/create_yolo_event")
async def create_mock_yolo_event():
    from datetime import datetime
    eid = sqlite_db._uuid()
    now = datetime.now().isoformat()
    async with app.state.db_lock:
        sqlite_db.insert_yolo_event(
            app.state.db,
            now, # timestamp
            yolo_event_id=eid,
            person_count=1,
            event_type="person_dwelling",
            source_region_name="보안 구역 A"
        )
    return {"ok": True, "event_id": eid}


# ======================================================================================================= 2차 인증 관련 엔드포인트 추가
# ====================================================================

class StartAuthReq(BaseModel):
    tracking_person_id: Optional[str] = None
    yolo_event_id: Optional[str] = None
    timestamp: Optional[str] = None


# ======================================================= 2차 인증 관련 endpoint
# =========== 로봇 > 서버 엔드포인트 
# 1. POST /auth/start : 2차 인증 시작시에 로봇이 알리는 endpoint
@app.post("/auth/start")
async def start_auth(req: StartAuthReq):
    ts = req.timestamp or datetime.now().isoformat()

    async with app.state.db_lock:
        pose = app.state.robot_pose or {}

        linked_yolo_event_id = req.yolo_event_id

        if linked_yolo_event_id is None:
            latest = sqlite_db.get_latest_yolo_event(
                app.state.db,
                tracking_person_id=req.tracking_person_id,
            )
            if latest is not None:
                linked_yolo_event_id = latest["yolo_event_id"]

        x = pose.get("x")
        y = pose.get("y")
        yaw = pose.get("yaw")

        source_region_id = None
        source_region_name = None

        if x is not None and y is not None:
            source_region_id, source_region_name = find_region_from_pose(
                app.state.db, x, y
            )

        auth_event_id = sqlite_db.insert_auth_event(
            db=app.state.db,
            timestamp=ts,
            tracking_person_id=req.tracking_person_id,
            yolo_event_id=linked_yolo_event_id,
            status="waiting_rfid",
            source_region_id=source_region_id,
            source_region_name=source_region_name,
            x=x,
            y=y,
            yaw=yaw,
        )

        auth_event = sqlite_db.get_auth_event(app.state.db, auth_event_id)

    return {
        "ok": True,
        "auth_event_id": auth_event_id,
        "auth_event": auth_event,
    }

# 2차 인증 ID 판정 엔드포인트
@app.post("/auth/rfid")
async def verify_rfid(
    auth_event_id: str = Form(...),
    rfid_uid: str = Form(...),
    timestamp: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    ts = timestamp or datetime.now().isoformat()
    uid = str(rfid_uid).strip().upper()

    image_path = None
    image_url = None

    if image is not None:
        ext = Path(image.filename or "").suffix.lower() or ".jpg"
        safe_ts = ts.replace(":", "-")
        save_path = AUTH_EVENT_ROOT / f"{safe_ts}_{auth_event_id}{ext}"

        data = await image.read()
        if data:
            save_path.write_bytes(data)
            image_path = str(save_path)
            image_url = f"/auth_images/{save_path.name}"

    async with app.state.db_lock:
        auth_event = sqlite_db.get_auth_event(app.state.db, auth_event_id)
        if auth_event is None:
            raise HTTPException(status_code=404, detail="auth event not found")

        emp = sqlite_db.get_employee_by_rfid(app.state.db, uid)

        if emp is None:
            sqlite_db.update_auth_event_result(
                db=app.state.db,
                auth_event_id=auth_event_id,
                status="fail",
                employee_id=None,
                rfid_uid=uid,
                employee_name=None,
                result_message="unregistered or inactive card",
                image_path=image_path,
            )
        else:
            sqlite_db.update_auth_event_result(
                db=app.state.db,
                auth_event_id=auth_event_id,
                status="success",
                employee_id=emp["employee_id"],
                rfid_uid=uid,
                employee_name=emp["name"],
                result_message="authorized employee",
                image_path=image_path,
            )

        updated = sqlite_db.get_auth_event(app.state.db, auth_event_id)

    return {
        "ok": True,
        "auth_event": updated,
        "image_url": image_url,
    }

# 시간 내 인증 실패시 timeout 알리는 엔드포인트
@app.post("/auth/timeout")
async def auth_timeout(
    auth_event_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    ts = timestamp or datetime.now().isoformat()

    image_path = None
    image_url = None

    if image is not None:
        ext = Path(image.filename or "").suffix.lower() or ".jpg"
        safe_ts = ts.replace(":", "-")
        save_path = AUTH_EVENT_ROOT / f"{safe_ts}_{auth_event_id}{ext}"

        data = await image.read()
        if data:
            save_path.write_bytes(data)
            image_path = str(save_path)
            image_url = f"/auth_images/{save_path.name}"

    async with app.state.db_lock:
        row = sqlite_db.get_auth_event(app.state.db, auth_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="auth event not found")

        sqlite_db.set_auth_event_timeout(
            app.state.db,
            auth_event_id,
            image_path=image_path,
        )
        updated = sqlite_db.get_auth_event(app.state.db, auth_event_id)

    return {
        "ok": True,
        "auth_event": updated,
        "image_url": image_url,
    }

# ================= [2차인증] GUI > 서버 조회 엔드포인트

@app.get("/auth_events")
async def get_auth_events(
    since: Optional[str] = None,
    limit: int = 50,
    status: Optional[str] = None,
):
    """
    인증 이벤트 polling용.
    - since가 없으면 최근 인증 이벤트 목록 반환
    - since가 있으면 그 시각 이후의 인증 이벤트만 반환
    - status가 있으면 waiting_rfid / success / fail / timeout 필터 가능
    """
    async with app.state.db_lock:
        rows = sqlite_db.list_auth_events(
            app.state.db,
            since=since,
            limit=limit,
            status=status,
        )

    out = []
    for r in rows:
        item = dict(r)
        if item.get("image_path"):
            item["image_url"] = f"/auth_images/{Path(item['image_path']).name}"
        else:
            item["image_url"] = None
        out.append(item)

    return {
        "ok": True,
        "auth_events": out,
    }


@app.get("/auth_events/{auth_event_id}")
async def get_auth_event_detail(auth_event_id: str):
    async with app.state.db_lock:
        row = sqlite_db.get_auth_event(app.state.db, auth_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="auth event not found")

    item = dict(row)
    if item.get("image_path"):
        item["image_url"] = f"/auth_images/{Path(item['image_path']).name}"
    else:
        item["image_url"] = None

    return {
        "ok": True,
        "auth_event": item,
    }

class UpdateAuthEventLabelReq(BaseModel):
    admin_label: Optional[str] = None


@app.patch("/auth_events/{auth_event_id}/label")
async def update_auth_event_label(auth_event_id: str, req: UpdateAuthEventLabelReq):
    async with app.state.db_lock:
        row = sqlite_db.get_auth_event(app.state.db, auth_event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="auth event not found")

        sqlite_db.set_auth_event_admin_label(
            app.state.db,
            auth_event_id,
            req.admin_label,
        )

        updated = sqlite_db.get_auth_event(app.state.db, auth_event_id)

    return {
        "ok": True,
        "auth_event": updated,
    }


# ======================================================= 경유점 위한 엔드포인트 추가

class CreateGuiWaypointReq(BaseModel):
    place_id: Optional[str] = None
    x: float
    y: float
    yaw: float = 0.0
    display_name: Optional[str] = None
    patrol_enabled: bool = True

@app.post("/gui/waypoints")
async def create_gui_waypoint(req: CreateGuiWaypointReq):
    async with app.state.db_lock:
        if req.place_id is None or not req.place_id.strip():
            place_id = sqlite_db.generate_unique_waypoint_id(app.state.db)
        else:
            place_id = req.place_id.strip()

        existing = sqlite_db.get_place(app.state.db, place_id)
        if existing is not None:
            raise HTTPException(status_code=409, detail="place_id already exists")

        sqlite_db.insert_gui_waypoint(
            db=app.state.db,
            place_id=place_id,
            x=req.x,
            y=req.y,
            yaw=req.yaw,
            display_name=req.display_name or place_id,
            patrol_enabled=req.patrol_enabled,
        )

        place = sqlite_db.get_place(app.state.db, place_id)

    return {
        "ok": True,
        "status": "gui_waypoint_created",
        "place": place,
    }

# ============================================================= 로봇 배터리 잔량 확인

class RobotBatteryReq(BaseModel):
    percentage: int

@app.post("/robot/battery")
async def update_robot_battery(req: RobotBatteryReq):
    percentage = int(req.percentage)

    if percentage < 0 or percentage > 100:
        raise HTTPException(
            status_code=400,
            detail="percentage must be between 0 and 100",
        )

    app.state.robot_battery = percentage

    print(
        f"[ROBOT BATTERY] {percentage}%",
        flush=True,
    )

    return {
        "ok": True,
        "battery": app.state.robot_battery,
    }

@app.get("/robot/battery")
async def get_robot_battery():
    return {
        "ok": True,
        "battery": app.state.robot_battery,
    }