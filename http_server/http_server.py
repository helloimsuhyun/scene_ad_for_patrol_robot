# http_server.py

""" 
    로봇쪽에서 보내는 json meta

    meta = {
        "place_id": place_id, #str 각 장소별 고유한 id str
        "timestamp": datetime.now().isoformat(),
        "n_frames": len(frames), 
        "mode" : mode, # bank, query, th_calib로 제한
        "label" : gt # 평상시 None, 정답이 주어진 경우에는 normal, unnomal
    }

    서버 db >> sqlite_db
        event table / frame table / place table



    >>>> events
    ------
    1 row = 1 개 query event

    Columns
    - event_id        : UUID (PRIMARY KEY)
    - place_id        : place 고유 id (파일명)
    - captured_at     : event capture timestamp
    - anomaly_flag    : 0 = 정상, 1 = 비정상

    Optional fields
    - anomaly_score   : 이상 점수
    - threshold_used  : 감지에 사용한 임계치
    - ref_bank_id     : reference bank identifier
    - ref_topk_json   : top-k reference matches (JSON)
    - summary_text    : optional description

    Auto fields
    - created_at      : db에 기록된 시간


    >>>>frames
    ------
    1 row = 1개 캡쳐된 이미지

    Columns
    - frame_id        : UUID (PRIMARY KEY)
    - event_id        : 위에 event uuid를 그대로 받아옴
    - idx             : event안에서 1개 frame의 idx
    - image_path      : 해당 이미지 저장된 경로

    - frame_score     : 해당 frame의 이상 점수
    - capture_time    : capture timestamp

    Relationship
    - events (1) → frames (N)

    >>>>places
    ------

    Columns
    - place_id          : place 고유 id (파일명)
    - mode              : current operation mode
                        ('idle', 'bank', 'th_calib', 'query')
    - need_calibration  : 임계치 재계산이 필요한 경우 True(1)
    - updated_at        : last updated time

    Role
    - mode로 로봇에 어떤 모드로 보낼지 컨트롤함
    - need_calibration 변수로 

        




"""


import json
import asyncio
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import sqlite_db
import dino_emb
import place_manager
from distance import infer_event, calibrate_place

# path 
SAVE_ROOT = Path("./recv")  # 이미지저장 root
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = Path("./recv/events.db") # db root

def sync_places_from_fs(db, save_root: Path):
    for p in save_root.iterdir():
        if not p.is_dir():
            continue

        place_id = p.name
        sqlite_db.ensure_place(db, place_id)

# 서버 시작 & 끝 관리 lifespan
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

    # GPU lock
    app.state.gpu_lock = asyncio.Lock()

    app.state.calib_progress = {
    "total": 0,
    "done": 0,
    "current_place_id": None,
    }

    #model load
    model, device = dino_emb.load_model()
    app.state.engine = {
        "model": model,
        "device": device,
        "bank_root":SAVE_ROOT,
    }

    print(" ------------Server startup complete")

    yield  # ------------- 

    # 서버 종료 시 (shutdown)
    print(" ------------ Server shutting down")
    app.state.db.close()
    print("DB closed")

app = FastAPI(lifespan=lifespan)

#추론함수 호출 -------------------------------------------
def run_inference_event(imgs_bgr, meta_obj, engine):
    place_id = str(meta_obj.get("place_id", "unknown"))
    return infer_event(
        imgs_bgr,
        place_id,
        engine["bank_root"],
        engine["model"],
        engine["device"],
    )

#임계치 업데이트 함수 호출----------------------------------
def run_threshold_calibration(place_id, engine):
    thr, scores, _ = calibrate_place(
        engine["bank_root"],
        place_id,
        engine["model"],
        engine["device"],
    )
    return thr


# -----------------------------------------------------------------------------------------------
# place table 상태 조회 / 제어 API

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
        place = place_manager.get_place_status(app.state.db, SAVE_ROOT, place_id)
    return {
        "ok": True,
        "global_calibrating": app.state.global_calibrating,
        "place": place,
    }


# GUI에서 place mode 변경
@app.post("/places/{place_id}/config")
async def set_place_config(
    place_id: str,
    mode: str = Form(...),
):
    try:
        async with app.state.db_lock:
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


# ----------------------------------------------------------------------------------------------------
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

    place_id = str(meta_obj.get("place_id", "unknown"))
    mode = str(meta_obj.get("mode", "unknown"))
    label = meta_obj.get("label", None)  
    ts = meta_obj.get("timestamp") or datetime.now().isoformat()

    async with app.state.db_lock:
        sqlite_db.ensure_place(app.state.db, place_id)

    if mode not in ("bank", "th_calib", "query"):
        raise HTTPException(status_code=400, detail="mode must be one of bank/th_calib/query")

    # 저장 디렉토리: root / place / mode
    safe_ts = ts.replace(":", "-")
    out_dir = SAVE_ROOT / place_id / mode 
    out_dir.mkdir(parents=True, exist_ok=True)

    if label is None:
        prefix = f"{place_id}_{mode}_{safe_ts}"
    else : 
        prefix = f"{label}_{place_id}_{mode}_{safe_ts}"

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
    th_path = SAVE_ROOT / place_id / "threshold.json"


    if mode in ("bank", "th_calib"):
        # th.json 존재하는데, ref bank 업데이트하면 calibaration 필요로 db에 기록
        if th_path.exists():
            async with app.state.db_lock:
                place_manager.set_need_calibration(app.state.db,place_id, True)

    if mode == "query":

        if not th_path.exists(): #fail-safe
            return JSONResponse({
                "ok": False,
                "status": "threshold_not_ready",
                "place_id": place_id,
                "n_images": len(saved),
                "saved_dir": str(out_dir),
                "meta": meta_obj,
            })

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
            "n_images": len(saved),
            "saved_dir": str(out_dir),
            "meta": meta_obj,
        }
    )



if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000) #로봇, 서버 통신이면 
    uvicorn.run(app, host="127.0.0.1", port=8000) #한 개 pc안에서 test이면