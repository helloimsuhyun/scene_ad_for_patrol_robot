import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
import torch
from diffusers import StableDiffusionInpaintPipeline


# =========================================================
# ✅ 여기만 바꿔서 컨트롤
# =========================================================
CONFIG = {
    "bank_root": "/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank",
    "out_root": "/home/choisuhyun/scene_ad_for_patrol_robot/data/anom_dataset",
    "plc_idx": "00",

    "n_per_image": 2,
    "mask_mode": "mix",  # "box" | "blob" | "mix"

    "prompt_type": "damage",  # "remove" | "damage"
    "prompt": None,           # None이면 내부 pool에서 랜덤 선택. 직접 지정하려면 문자열

    "negative_prompt": "low quality, blurry, artifacts, distorted",
    "strength": 0.95,
    "guidance_scale": 7.5,
    "steps": 30,
    "seed": 0,

    # ✅ 최종: 존재하는 repo_id로 수정 (404 해결)
    # - torch<2.6 환경에서 pickle(.bin) 로딩이 막혀있으므로 use_safetensors=True 유지
    "model_id": "sd2-community/stable-diffusion-2-inpainting",

    "device": "cuda",
    "fp16": True,
}
# =========================================================


# -----------------------------
# Mask generators
# -----------------------------
def make_random_box_mask(h, w, min_frac=0.10, max_frac=0.35):
    mask = np.zeros((h, w), dtype=np.uint8)
    bw = int(random.uniform(min_frac, max_frac) * w)
    bh = int(random.uniform(min_frac, max_frac) * h)
    x0 = random.randint(0, max(0, w - bw))
    y0 = random.randint(0, max(0, h - bh))
    mask[y0:y0 + bh, x0:x0 + bw] = 255
    return Image.fromarray(mask, mode="L")


def make_random_blob_mask(h, w, n_blobs=3, radius_frac=(0.05, 0.18)):
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n_blobs):
        r = int(random.uniform(*radius_frac) * min(h, w))
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        cv2.circle(mask, (cx, cy), r, 255, -1)

    k = random.choice([11, 15, 21])
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    thr = random.randint(80, 140)
    mask = (mask > thr).astype(np.uint8) * 255

    if random.random() < 0.6:
        dk = random.choice([7, 9, 11])
        kernel = np.ones((dk, dk), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return Image.fromarray(mask, mode="L")


def build_mask(img_pil, mask_mode="box"):
    w, h = img_pil.size
    if mask_mode == "box":
        return make_random_box_mask(h, w)
    elif mask_mode == "blob":
        return make_random_blob_mask(h, w)
    elif mask_mode == "mix":
        return make_random_box_mask(h, w) if random.random() < 0.5 else make_random_blob_mask(h, w)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")


# -----------------------------
# Prompts
# -----------------------------
DEFAULT_PROMPTS = {
    "remove": [
        "empty floor, clean surface, realistic",
        "clean wall, empty area, realistic",
        "nothing on the table, clean table, realistic",
    ],
    "damage": [
        "damaged surface, scratch, realistic",
        "stain and dirt on surface, realistic",
        "burn mark on surface, realistic",
        "crack on wall texture, realistic",
    ],
}


def pick_prompt(prompt_type="remove", prompt=None):
    if prompt is not None and len(str(prompt).strip()) > 0:
        return str(prompt).strip()
    return random.choice(DEFAULT_PROMPTS.get(prompt_type, DEFAULT_PROMPTS["remove"]))


# -----------------------------
# Pipeline
# -----------------------------
def load_pipeline(model_id, device="cuda", fp16=True):
    dtype = torch.float16 if (device.startswith("cuda") and fp16) else torch.float32

    # ✅ 핵심: safetensors만 사용 (pickle/.bin 로드 차단)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def iter_bank_images(bank_root: Path, plc_idx: str):
    bank_dir = bank_root / str(plc_idx) / "bank"
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    paths = []
    for ext in exts:
        paths += list(bank_dir.rglob(f"*{ext}"))
    return sorted(paths)


def gen_anomaly_for_place(cfg: dict):
    bank_root = Path(cfg["bank_root"])
    out_root = Path(cfg["out_root"])
    plc_idx = str(cfg["plc_idx"])

    img_paths = iter_bank_images(bank_root, plc_idx)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found under {bank_root}/{plc_idx}/bank")

    out_dir = out_root / plc_idx / "anom"
    mask_dir = out_root / plc_idx / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # meta 저장
    (out_root / plc_idx).mkdir(parents=True, exist_ok=True)
    with open(out_root / plc_idx / "anom_gen_meta.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    pipe = load_pipeline(cfg["model_id"], device=cfg["device"], fp16=cfg.get("fp16", True))

    rng = random.Random(int(cfg.get("seed", 0)))

    for p in tqdm(img_paths, desc=f"[{plc_idx}] inpainting"):
        img = Image.open(p).convert("RGB")

        for j in range(int(cfg["n_per_image"])):
            local_seed = rng.randint(0, 2**31 - 1)
            generator = torch.Generator(device=cfg["device"]).manual_seed(local_seed)

            mask = build_mask(img, mask_mode=cfg["mask_mode"])
            pr = pick_prompt(prompt_type=cfg["prompt_type"], prompt=cfg.get("prompt", None))

            res = pipe(
                prompt=pr,
                negative_prompt=cfg["negative_prompt"],
                image=img,
                mask_image=mask,
                strength=float(cfg["strength"]),
                guidance_scale=float(cfg["guidance_scale"]),
                num_inference_steps=int(cfg["steps"]),
                generator=generator,
            ).images[0]

            stem = p.stem
            res.save(out_dir / f"{stem}_anom_{j:02d}.png")
            mask.save(mask_dir / f"{stem}_mask_{j:02d}.png")

    print(f"✅ done: saved to {out_dir}")


if __name__ == "__main__":
    gen_anomaly_for_place(CONFIG)
