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


class SDP(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def startup():
    global offer_ready_event, answer_ready_event
    offer_ready_event = asyncio.Event()
    answer_ready_event = asyncio.Event()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/viewer_offer")
async def viewer_offer(req: SDP):
    global viewer_offer_store, viewer_answer_store
    global offer_ready_event, answer_ready_event

    print("[viewer_offer] received from viewer")
    print(f"[viewer_offer] type={req.type}, sdp_len={len(req.sdp)}")

    viewer_offer_store = req.dict()
    viewer_answer_store = None

    # 새 offer가 들어왔으니 answer 대기는 다시 시작
    answer_ready_event.clear()
    offer_ready_event.set()

    print("[viewer_offer] stored offer, waiting for sender answer")

    try:
        await asyncio.wait_for(answer_ready_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("[viewer_offer] timeout waiting for sender answer")
        raise HTTPException(status_code=504, detail="timeout waiting for sender answer")

    if viewer_answer_store is None:
        print("[viewer_offer] answer event set but answer store is empty")
        raise HTTPException(status_code=500, detail="answer missing")

    print("[viewer_offer] sender answer received, returning to viewer")
    return viewer_answer_store


@app.get("/sender_poll")
async def sender_poll():
    global viewer_offer_store
    global offer_ready_event

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
    viewer_offer_store = None
    offer_ready_event.clear()

    print(
        f"[sender_poll] delivering offer to robot, "
        f"sdp_len={len(offer['sdp'])}, type={offer['type']}"
    )
    return offer


@app.post("/sender_answer")
async def sender_answer(req: SDP):
    global viewer_answer_store
    global answer_ready_event

    print("[sender_answer] received from robot")
    print(f"[sender_answer] type={req.type}, sdp_len={len(req.sdp)}")

    viewer_answer_store = req.dict()
    answer_ready_event.set()

    return {"ok": True}