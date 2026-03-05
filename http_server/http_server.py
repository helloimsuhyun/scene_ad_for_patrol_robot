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

"""


import json
import asyncio
from contextlib import asynccontextmanager
from PIL import Image
import io
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import sqlite_db
import dino_emb
from distance import infer_event, calibrate_place
from config import load_cfg

# path 
SAVE_ROOT = Path("./recv")  # 이미지저장 root
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = Path("./recv/events.db") # db root

# 서버 시작 & 끝 관리 lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):

    # 서버 시작시에 실행
    # db connect & init(없으면 만들어줌)
    app.state.db = sqlite_db.connect_db(DB_PATH)
    sqlite_db.init_db(app.state.db)
    app.state.db_lock = asyncio.Lock()

    #calibration lock
    app.state.calib_running = set()
    app.state.calib_lock = asyncio.Lock()

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

#추론함수 호출
def run_inference_event(imgs_bgr, meta_obj, engine):
    cfg = load_cfg(engine["bank_root"])
    i = cfg["infer"]
    place_id = str(meta_obj.get("place_id", "unknown"))
    return infer_event(
        imgs_bgr, place_id,
        engine["bank_root"],
        engine["model"],
        engine["device"],
        event_rule=i["event_rule"],
        use_two_stage_vlm=i["use_two_stage_vlm"],
    )

#임계치 업데이트 함수 호출
def run_threshold_calibration(place_id, engine):
    cfg = load_cfg(engine["bank_root"])
    c = cfg["calib"]
    thr, scores, _ = calibrate_place(
        engine["bank_root"], place_id,
        engine["model"], engine["device"],
        k=c["k"],
        percentile=c["percentile"],
        repr_mode=c["repr_mode"],
        grid_size=c["grid_size"],
        alpha=c["alpha"],
        method=c["method"],
    )
    return thr


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

    place_id = str(meta_obj.get("place_id", "unknown"))
    mode = str(meta_obj.get("mode", "unknown"))
    label = meta_obj.get("label", None)  
    ts = meta_obj.get("timestamp") or datetime.now().isoformat()

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
    
    th_path = SAVE_ROOT / place_id / "threshold.json"

    #mode가 bank / th_calib이면 기존에 있던 thresh hold는 무효화
    if mode in ("bank", "th_calib") and th_path.exists():
        th_path.unlink()
        print(f"[INFO] threshold invalidated (deleted): {th_path}")

    #--- 추론
    if mode == "query":
        #event 생성하고 , frame일단 집어넣기
        async with app.state.db_lock:
            event_id = sqlite_db.insert_event( #event 생성
                db=app.state.db,
                place_id=place_id,
                captured_at=ts,
            )

            sqlite_db.insert_frames(   #frame 넣기 - 1행 - 1개 frame
                db=app.state.db,
                event_id=event_id, 
                image_paths=saved,
                frame_scores=None,
                capture_times=ts, 
            )
        
        #추론
        if not th_path.exists():

            # place별 중복 캘리브 방지
            async with app.state.calib_lock:
                if place_id not in app.state.calib_running:
                    print(f"[INFO] threshold.json not found for place {place_id} → calibrating (background)")
                    app.state.calib_running.add(place_id)

                    task = asyncio.create_task(
                        asyncio.to_thread(
                            run_threshold_calibration,
                            place_id,
                            app.state.engine
                        )
                    )
                    def _calib_done_callback(t, pid=place_id):
                        try:
                            exc = t.exception()
                            if exc:
                                print(f"[CALIB ERROR] place={pid} exc={exc}")
                        finally:
                            app.state.calib_running.discard(pid)

                    task.add_done_callback(_calib_done_callback)
                else:
                    print(f"[INFO] threshold place {place_id} → calibrating wait plz")


            return JSONResponse({
                "ok": True,
                "status": "calibrating",
                "place_id": place_id,
                "retry_after_ms": 2000,
                "n_images": len(saved),
                "saved_dir": str(out_dir),
                "meta": meta_obj,
            })

        # threshold가 있으면 정상 추론
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
        print("add q")


    return JSONResponse(
        {
            "ok": True,
            "n_images": len(saved),
            "saved_dir": str(out_dir),
            "meta": meta_obj,
        }
    )

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000) 로봇, 서버 통신이면 
    uvicorn.run(app, host="127.0.0.1", port=8000)