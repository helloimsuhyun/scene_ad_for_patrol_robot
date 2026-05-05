# signaling_server.py
from typing import Optional
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

viewer_offer_store: Optional[dict] = None
viewer_answer_store: Optional[dict] = None

offer_ready_event: Optional[asyncio.Event] = None
answer_ready_event: Optional[asyncio.Event] = None

# 현재 viewer_offer 요청이 answer를 기다리는 중인지 표시
viewer_busy: bool = False

# 디버깅용 카운터
offer_seq: int = 0
delivered_offer_seq: Optional[int] = None


class SDP(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def startup():
    global offer_ready_event, answer_ready_event
    offer_ready_event = asyncio.Event()
    answer_ready_event = asyncio.Event()
    print("[startup] signaling server ready")


def clear_state():
    """현재 signaling 상태 초기화."""
    global viewer_offer_store, viewer_answer_store
    global offer_ready_event, answer_ready_event
    global viewer_busy, delivered_offer_seq

    viewer_offer_store = None
    viewer_answer_store = None
    viewer_busy = False
    delivered_offer_seq = None

    if offer_ready_event is not None:
        offer_ready_event.clear()
    if answer_ready_event is not None:
        answer_ready_event.clear()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/debug_state")
async def debug_state():
    return {
        "ok": True,
        "viewer_busy": viewer_busy,
        "has_offer": viewer_offer_store is not None,
        "has_answer": viewer_answer_store is not None,
        "offer_event_set": offer_ready_event.is_set() if offer_ready_event else None,
        "answer_event_set": answer_ready_event.is_set() if answer_ready_event else None,
        "offer_seq": offer_seq,
        "delivered_offer_seq": delivered_offer_seq,
    }


@app.post("/reset")
async def reset():
    clear_state()
    print("[reset] signaling state cleared")
    return {"ok": True}


@app.post("/viewer_offer")
async def viewer_offer(req: SDP):
    global viewer_offer_store, viewer_answer_store
    global offer_ready_event, answer_ready_event
    global viewer_busy, offer_seq, delivered_offer_seq

    # 중복 viewer_offer 방어
    # 프론트가 새로고침/재시도 등으로 offer를 또 보내면 기존 상태가 꼬일 수 있으므로 차단
    if viewer_busy:
        print("[viewer_offer] rejected: another viewer offer is already waiting")
        raise HTTPException(status_code=409, detail="viewer offer already pending")

    viewer_busy = True
    offer_seq += 1
    current_seq = offer_seq
    delivered_offer_seq = None

    print("[viewer_offer] received from viewer")
    print(f"[viewer_offer] seq={current_seq}, type={req.type}, sdp_len={len(req.sdp)}")

    try:
        viewer_offer_store = req.dict()
        viewer_answer_store = None

        # 새 offer가 들어왔으므로 answer 대기 상태 초기화
        answer_ready_event.clear()
        offer_ready_event.set()

        print(f"[viewer_offer] seq={current_seq} stored offer, waiting for sender answer")

        try:
            await asyncio.wait_for(answer_ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            print(f"[viewer_offer] seq={current_seq} timeout waiting for sender answer")

            # timeout이 나면 잔여 offer/answer를 정리해서 다음 연결에 영향 없게 함
            clear_state()
            raise HTTPException(status_code=504, detail="timeout waiting for sender answer")

        if viewer_answer_store is None:
            print(f"[viewer_offer] seq={current_seq} answer event set but answer store is empty")
            clear_state()
            raise HTTPException(status_code=500, detail="answer missing")

        print(f"[viewer_offer] seq={current_seq} sender answer received, returning to viewer")

        answer = viewer_answer_store

        # 정상 반환 직후에도 상태를 정리해 다음 연결이 깨끗하게 시작되도록 함
        clear_state()

        return answer

    except HTTPException:
        raise

    except Exception as e:
        print(f"[viewer_offer] seq={current_seq} unexpected error: {repr(e)}")
        clear_state()
        raise HTTPException(status_code=500, detail="viewer_offer internal error")


@app.get("/sender_poll")
async def sender_poll():
    global viewer_offer_store
    global offer_ready_event
    global delivered_offer_seq

    print("[sender_poll] robot polling started")

    try:
        await asyncio.wait_for(offer_ready_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("[sender_poll] no viewer offer within polling window")
        return JSONResponse(status_code=404, content={"detail": "no viewer offer"})

    if viewer_offer_store is None:
        print("[sender_poll] offer event set but store is empty")
        offer_ready_event.clear()
        return JSONResponse(status_code=404, content={"detail": "no viewer offer"})

    offer = viewer_offer_store

    # offer는 한 sender에게만 전달
    viewer_offer_store = None
    offer_ready_event.clear()
    delivered_offer_seq = offer_seq

    print(
        f"[sender_poll] delivering offer to robot, "
        f"seq={delivered_offer_seq}, sdp_len={len(offer['sdp'])}, type={offer['type']}"
    )

    return offer


@app.post("/sender_answer")
async def sender_answer(req: SDP):
    global viewer_answer_store
    global answer_ready_event

    print("[sender_answer] received from robot")
    print(f"[sender_answer] type={req.type}, sdp_len={len(req.sdp)}")

    # viewer가 answer를 기다리는 상태가 아니면 늦게 도착한 answer로 판단
    if not viewer_busy:
        print("[sender_answer] rejected: no viewer is waiting for answer")
        raise HTTPException(status_code=409, detail="no viewer waiting for answer")

    if answer_ready_event is None:
        print("[sender_answer] rejected: answer_ready_event is not initialized")
        raise HTTPException(status_code=500, detail="server not ready")

    viewer_answer_store = req.dict()
    answer_ready_event.set()

    print("[sender_answer] answer stored and event set")
    return {"ok": True}