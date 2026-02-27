# 1. 들어온 img에 대해서 이미지를 저장하고 장소 npz에 임베딩을 추가  def save_one_to_bank(img, plc_idx : str, save_root):
# 2. 장소 단위로 npz를 load해서 e npy 배열들과 이미지 경로 모음 list를 return  def load_bank_by_place(save_root, plc_idx):
# 3. 장소 단위로 npz를 전부 탐색해서 해당 폴더에 존재하는 img에 대해 npz를 다시 만듬  def rebuild_bank(save_root, plc_idx):

# save_root : ref_bank 폴더
# plc_idx : 장소에 대한 고유한 idx str
# img : numpy uint8

# - suhyun

"""
def save_one_to_bank(img, plc_idx : str, save_root):
def load_bank_by_place(save_root, plc_idx):
def rebuild_bank(save_root, plc_idx):
"""

import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from dino_emb import make_embed , make_transform

def BGR_to_RGB(img_bgr_uint8):
    img_rgb = img_bgr_uint8[:, :, ::-1]          # BGR->RGB
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    return img_pil

#npz에 한개 추가
def append_to_bank_npz(npz_path, emb, img_path):
    """
    npz_path: Path or str (예: bank_03.npz)
    emb: torch.Tensor [D]
    img_path: str (이미지 경로)
    """

    npz_path = Path(npz_path)
    emb = emb.detach().cpu().numpy().astype(np.float32)

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        embs = data["embs"]
        paths = data["paths"].tolist()

        embs = np.vstack([embs, emb])
        paths.append(str(img_path))
    else:
        embs = emb.reshape(1, -1) # (1 D)차원으로 변환
        paths = [str(img_path)]

    np.savez_compressed(
        npz_path,
        embs=embs,
        paths=np.array(paths, dtype=object),
    )


def save_one_to_bank(img, plc_idx : str, save_root , model , device, mode = "bank"):
    # input img : npy uint8 이미지
    # input plc_idx : 촬영 장소에 대한 고유한 인덱스
    # save_root : ref_bank 저장 상위폴더

    save_root = Path(save_root)
    plc_idx = str(plc_idx)

    save_path = save_root / plc_idx / mode
    save_path.mkdir(parents=True,exist_ok= True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms 단위

    #이미지 저장
    img_path = save_path  / f"{timestamp}.png"
    img = BGR_to_RGB(img)
    img.save(img_path)

    #임베딩 저장
    e_path = img_path.with_suffix(".npy")
    tfm = make_transform()
    img_tensor = tfm(img)
    e = make_embed(model,device,img_tensor)
    #np.save(e_path, e.detach().cpu().numpy().astype(np.float32))

    #save npz
    npz_path = save_path / f"{plc_idx}.npz"
    append_to_bank_npz(npz_path,e,img_path)

    return img_path , e_path , npz_path

def load_bank_npz(npz_path):
    """
    npz_path: Path or str
    return:
        ref_embs: numpy array [K, D]
        paths: list[str]
    """

    npz_path = Path(npz_path)

    if not npz_path.exists():
        return None, None

    data = np.load(npz_path, allow_pickle=True)

    ref_embs = data["embs"]       
    paths = data["paths"].tolist()     

    return ref_embs, paths

# 장소 폴더 단위로 npz를 load함
def load_bank_by_place(save_root, plc_idx, mode = "bank"):
    """
    save_root: ref_bank 상위폴더
    plc_idx: 장소 index (str)
    """

    save_root = Path(save_root)
    plc_idx = str(plc_idx)

    npz_path = save_root / plc_idx / mode / f"{plc_idx}.npz"

    return load_bank_npz(npz_path)

#폴더 안 기준으로 npz 재생성
def rebuild_bank(save_root, plc_idx , model, device, mode = "bank"):

    save_root = Path(save_root)
    plc_idx = str(plc_idx)

    save_path = save_root / plc_idx / mode
    npz_path = save_path / f"{plc_idx}.npz"

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([p for p in save_path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS])


    if len(img_paths) == 0:
        print("no images found")
        return None

    ref_embs = []
    paths = []
    tfm = make_transform()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        
        x = tfm(img)
        e = make_embed(model , device, x)

        ref_embs.append(
            e.detach().cpu().numpy().astype(np.float32)
        )
        paths.append(str(img_path))

    ref_embs = np.vstack(ref_embs)

    np.savez_compressed(
        npz_path,
        embs=ref_embs,
        paths=np.array(paths, dtype=object),
    )

    return ref_embs, paths