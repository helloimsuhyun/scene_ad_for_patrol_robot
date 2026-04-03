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

    print("[viewer_offer] received from viewer")
    print(f"[viewer_offer] type={req.type}, sdp_len={len(req.sdp)}")

    viewer_offer_store = req.dict()
    viewer_answer_store = None

    sender_offer_store = None
    sender_answer_store = None

    print("[viewer_offer] stored offer, notifying sender pollers")
    sender_waiter.set()

    for i in range(300):
        await asyncio.sleep(0.1)
        if viewer_answer_store is not None:
            print(f"[viewer_offer] got answer after {0.1*(i+1):.1f}s")
            return viewer_answer_store

    print("[viewer_offer] timeout waiting for sender answer")
    raise HTTPException(status_code=504, detail="timeout waiting for sender answer")


@app.get("/sender_poll")
async def sender_poll():
    global viewer_offer_store
    print("[sender_poll] robot polling started")

    for i in range(300):
        await asyncio.sleep(0.1)
        if viewer_offer_store is not None:
            print(f"[sender_poll] found viewer offer after {0.1*(i+1):.1f}s")
            offer = viewer_offer_store
            viewer_offer_store = None
            print(f"[sender_poll] delivering offer to robot, sdp_len={len(offer['sdp'])}, type={offer['type']}")
            return offer

    print("[sender_poll] no viewer offer within polling window")
    return JSONResponse(status_code=404, content={"detail": "no viewer offer"})


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