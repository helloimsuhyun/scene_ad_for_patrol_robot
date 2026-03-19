# signaling_server.py
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio


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

sender_offer_store: Optional[dict] = None
sender_answer_store: Optional[dict] = None

viewer_waiter: Optional[asyncio.Event] = None
sender_waiter: Optional[asyncio.Event] = None


class SDP(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def startup():
    global viewer_waiter, sender_waiter
    viewer_waiter = asyncio.Event()
    sender_waiter = asyncio.Event()


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/viewer_offer")
async def viewer_offer(req: SDP):
    global viewer_offer_store, viewer_answer_store
    global sender_offer_store, sender_answer_store
    global viewer_waiter, sender_waiter

    viewer_offer_store = req.dict()
    viewer_answer_store = None

    sender_offer_store = None
    sender_answer_store = None

    sender_waiter.set()

    for _ in range(300):
        await asyncio.sleep(0.1)
        if viewer_answer_store is not None:
            return viewer_answer_store

    raise HTTPException(status_code=504, detail="timeout waiting for sender answer")


@app.post("/sender_offer")
async def sender_offer(req: SDP):
    global viewer_offer_store, viewer_answer_store
    global sender_offer_store, sender_answer_store
    global viewer_waiter, sender_waiter

    sender_offer_store = req.dict()

    if viewer_offer_store is None:
        raise HTTPException(status_code=409, detail="no viewer waiting")

    # sender가 viewer offer를 받아야 하므로 여기서는 직접 answer가 아니라
    # 구조를 바꾸는 게 낫다.
    raise HTTPException(status_code=500, detail="Use /sender_poll and /sender_answer flow")


@app.get("/sender_poll")
async def sender_poll():
    global viewer_offer_store
    print("[Signaling] Robot is polling for offers...") # 로봇이 접속하면 이 로그가 찍힙니다.
    for _ in range(300):
        await asyncio.sleep(0.1)
        if viewer_offer_store is not None:
            print("[Signaling] Offer found! Sending to robot.")
            # Offer를 가져가면 서버에서는 비워줍니다 (중복 방지)
            offer = viewer_offer_store
            viewer_offer_store = None 
            return offer
    return JSONResponse(status_code=404, content={"detail": "no viewer offer"})


@app.post("/sender_answer")
async def sender_answer(req: SDP):
    global viewer_answer_store
    viewer_answer_store = req.dict()
    return {"ok": True}