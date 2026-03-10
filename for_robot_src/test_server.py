# server.py
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

SAVE_ROOT = Path("./recv")  # 저장 root
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

""" 
    로봇쪽 json meta

    meta = {
        "place_id": place_id, #str 각 장소별 고유한 id str
        "timestamp": datetime.now().isoformat(),
        "n_frames": len(frames), 
        "mode" : mode, # bank, query, th_calib로 제한
        "label" : gt # 평상시 None, 정답이 주어진 경우에는 normal, unnomal
    }

"""

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
    label = str(meta_obj.get("label", "unknown"))

    ts = meta_obj.get("timestamp") or datetime.now().isoformat()

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
    for i, uf in enumerate(images):
        ext = Path(uf.filename).suffix.lower() or ".jpg"
        out_path = out_dir / f"{prefix}_{i:03d}{ext}"

        data = await uf.read()
        out_path.write_bytes(data)
        saved.append(str(out_path))

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