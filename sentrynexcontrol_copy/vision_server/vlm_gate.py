import os
import base64
import io
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
from PIL import Image
from openai import OpenAI

client = OpenAI()  # OPENAI_API_KEY 환경변수 사용
PathLike = Union[str, os.PathLike]


def bgr_numpy_to_pil_rgb(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) BGR image, got {img_bgr.shape}")
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
    img_rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(img_rgb)


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt in ("JPEG", "JPG"):
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        mime = "image/jpeg"
    elif fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def vlm_gate(
    q_img_bgr: np.ndarray,
    ref_img_path: PathLike,
    model: str = "gpt-4.1-mini",
    timeout_s: int = 30,
    max_tokens: int = 200,
) -> Dict[str, Any]:
    ref_img_path = Path(ref_img_path)
    if not ref_img_path.exists():
        raise FileNotFoundError(f"ref_img_path not found: {ref_img_path}")

    q_pil = bgr_numpy_to_pil_rgb(q_img_bgr)
    ref_pil = Image.open(ref_img_path).convert("RGB")

    q_url = pil_to_data_url(q_pil, fmt="JPEG", quality=85)
    ref_url = pil_to_data_url(ref_pil, fmt="JPEG", quality=85)

    instruction = (
        "역할: 시각 변화 검증 모듈.\n"
        "REFERENCE(정상)와 QUERY(현재)를 비교하라.\n"
        "조명/그림자/노출/화이트밸런스 변화는 물리 변화가 아니다.\n"
        "물리 변화는 물체 추가/제거/이동, 문 열림/닫힘, 파손 등 실제 상태 변화만.\n"
        "아래 JSON 스키마로만 답하라.\n"
        "- physical_change: true/false\n"
        "- description: 한국어 한 문장으로 '무엇이 어떻게 바뀜 + 근거'를 같이.\n"
        "물리 변화가 없으면 description에 '조명/그림자/노이즈로 판단'과 근거 포함.\n"
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "VLMGateResult",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "physical_change": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["physical_change", "description"],
            },
        },
    }

    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                response_format=response_format,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "text", "text": "REFERENCE (normal) image:"},
                            {"type": "image_url", "image_url": {"url": ref_url}},
                            {"type": "text", "text": "QUERY (current) image:"},
                            {"type": "image_url", "image_url": {"url": q_url}},
                        ],
                    }
                ],
                timeout=timeout_s,
            )

            txt = resp.choices[0].message.content
            data = json.loads(txt)
            data["physical_change"] = bool(data["physical_change"])
            data["description"] = str(data["description"]).strip()
            return data

        except Exception as e:
            last_err = e

    raise RuntimeError(f"vlm_gate failed: {last_err}")