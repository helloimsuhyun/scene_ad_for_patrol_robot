from pathlib import Path
import torch

from distance import calibrate_place
from dino_emb import load_model


def calibrate_one_place(
    bank_root: str,
    plc_idx: str,
    k: int = 3,
    percentile: int = 95,
):
    """
    특정 place(plc_idx) 하나에 대해:

    1. bank npz 재생성
    2. th_calib npz 재생성
    3. threshold.json 생성

    return:
        threshold (float)
    """

    bank_root = Path(bank_root)
    plc_idx = str(plc_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model ,device = load_model(device=device)

    thr, scores, thr_path = calibrate_place(
        bank_root=bank_root,
        plc_idx=plc_idx,
        model=model,
        device=device,
        k=k,
        percentile=percentile,
    )

    print("--------------------------------------------------")
    print(f"[OK] plc_idx      : {plc_idx}")
    print(f"[OK] threshold    : {thr:.6f}")
    print(f"[OK] saved path   : {thr_path}")
    print(f"[OK] #calib imgs  : {len(scores)}")

    return float(thr)


if __name__ == "__main__":
    calibrate_one_place(
        bank_root="./recv",   # 너 경로로 수정
        plc_idx="01",          # 원하는 place id
        k=3,
        percentile=95,
    )