import asyncio
import threading
from typing import Optional
import cv2
import numpy as np
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import requests

class BufferVideoTrack(VideoStreamTrack):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self._last_frame: Optional[np.ndarray] = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        try:
            # wait_new(1.0) 처럼 버퍼에서 타임아웃 1초를 두고 새 프레임을 대기합니다
            rgb = await asyncio.to_thread(self.buffer.wait_new, 1.0)
            self._last_frame = rgb
        except Exception:
            rgb = self._last_frame

        if rgb is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

class WebRTCSender:
    def __init__(self, buffer, signaling_base_url: str):
        self.buffer = buffer
        self.signaling_base_url = signaling_base_url.rstrip("/")
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.pc: Optional[RTCPeerConnection] = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _thread_main(self):
        # 새로운 이벤트 루프를 생성해 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()

    async def _run(self):
        while self.running:
            try:
                # viewer offer 대기
                resp = requests.get(f"{self.signaling_base_url}/sender_poll", timeout=35.0)
                if resp.status_code != 200:
                    continue

                offer_data = resp.json()

                pc = RTCPeerConnection()
                self.pc = pc

                @pc.on("connectionstatechange")
                async def on_connectionstatechange():
                    print("[WebRTC] sender state:", pc.connectionState)

                pc.addTrack(BufferVideoTrack(self.buffer))

                offer = RTCSessionDescription(
                    sdp=offer_data["sdp"],
                    type=offer_data["type"],
                )
                await pc.setRemoteDescription(offer)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                requests.post(
                    f"{self.signaling_base_url}/sender_answer",
                    json={
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    },
                    timeout=10.0,
                )

                while self.running and pc.connectionState not in ("failed", "closed", "disconnected"):
                    await asyncio.sleep(1.0)

            except Exception as e:
                print(f"[WebRTC] sender error: {e}")
            finally:
                if self.pc is not None:
                    try:
                        await self.pc.close()
                    except Exception:
                        pass
                    self.pc = None

            await asyncio.sleep(1.0)
