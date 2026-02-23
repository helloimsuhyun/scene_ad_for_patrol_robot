#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import shutil
import uuid
from pathlib import Path
from typing import List, Dict

# =========================
# ✅ 여기만 수정
# =========================
ROOT = Path("/home/choisuhyun/scene_ad_for_patrol_robot/dataset")
APPLY = False  # 먼저 False로 로그 확인 후 True
SEED = 42

REF_RATIO = 0.6
TH_RATIO  = 0.2
QUERY_NORM_RATIO = 0.2  # (nomal_for_query에서) query 정상 후보를 얼마나 쓸지(잔여는 unused로)

BALANCE_QUERY_1TO1 = True
PAD_WIDTH = 6

IN_REF_SRC_DIR = "nomal"            # ref/th는 여기서만 뽑음
IN_QUERY_N_DIR = "nomal_for_query"  # query 정상은 여기서만 뽑음
IN_QUERY_A_DIR = "unnormal"         # query 비정상은 여기서만 뽑음

OUT_N_PREFIX = "nomal"
OUT_A_PREFIX = "unnormal"

ONLY_NUMERIC_PLACE_DIR = True

# ✅ 출력 폴더가 이미 있으면 그 안에 쌓을지 여부
ALLOW_APPEND_OUTPUT = True
# =========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NPY_EXTS = {".npy"}
SUPPORTED_EXTS = IMG_EXTS | NPY_EXTS

def is_supported_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS

def list_files_flat(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if is_supported_file(p)])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst_dir: Path) -> Path:
    """원본 유지: dst_dir로 복제. 이름 충돌 시 _dupN 붙임."""
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = dst.stem, dst.suffix
        k = 1
        while True:
            cand = dst_dir / f"{stem}_dup{k}{ext}"
            if not cand.exists():
                dst = cand
                break
            k += 1
    print(f"[COPY] {src} -> {dst}")
    if APPLY:
        shutil.copy2(str(src), str(dst))
    return dst

def format_name(prefix: str, idx: int, width: int, ext: str) -> str:
    return f"{prefix}_{idx:0{width}d}{ext}"

def safe_rename_in_one_dir_with_start(files: List[Path], prefix: str, start_idx: int) -> int:
    """
    같은 폴더 내 files를 충돌 없이 rename.
    index는 start_idx부터 시작해서 증가.
    반환: 다음에 쓸 index
    """
    if not files:
        return start_idx

    parents = {f.parent for f in files}
    if len(parents) != 1:
        raise ValueError("rename expects files from ONE directory only.")
    parent = next(iter(parents))
    fs = sorted(files)

    # 1) temp rename
    temp_map: Dict[Path, Path] = {}
    for src in fs:
        ext = src.suffix.lower()
        tmp = parent / f"__tmp__{uuid.uuid4().hex}{ext}"
        temp_map[src] = tmp
        print(f"[TMP] {src.name} -> {tmp.name}")
        if APPLY:
            if tmp.exists():
                raise FileExistsError(tmp)
            src.rename(tmp)

    # 2) final rename (start_idx부터)
    tmp_files = [temp_map[s] for s in fs]
    tmp_files.sort()
    cur = start_idx
    for tmp in tmp_files:
        ext = tmp.suffix.lower()
        final = parent / format_name(prefix, cur, PAD_WIDTH, ext)
        print(f"[RENAME] {tmp.name} -> {final.name}")
        if APPLY:
            if final.exists():
                raise FileExistsError(final)
            tmp.rename(final)
        cur += 1

    return cur

def iter_place_dirs(root: Path) -> List[Path]:
    places = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if ONLY_NUMERIC_PLACE_DIR and not p.name.isdigit():
            continue
        places.append(p)
    return places

def rebalance_one_place(place_dir: Path):
    s = REF_RATIO + TH_RATIO + QUERY_NORM_RATIO
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"REF_RATIO+TH_RATIO+QUERY_NORM_RATIO must be 1.0, got {s}")

    src_ref = place_dir / IN_REF_SRC_DIR
    src_qn  = place_dir / IN_QUERY_N_DIR
    src_qa  = place_dir / IN_QUERY_A_DIR

    nomal_ref_pool = list_files_flat(src_ref)
    nomal_q_pool   = list_files_flat(src_qn)
    abn_q_pool     = list_files_flat(src_qa)

    print(f"\n===== PLACE {place_dir.name} =====")
    print(f"[INPUT] ref_pool(nomal)={len(nomal_ref_pool)}, query_pool(nomal_for_query)={len(nomal_q_pool)}, query_pool(unnormal)={len(abn_q_pool)}")

    # 출력 폴더
    ref_dir   = place_dir / "ref_bank"
    th_dir    = place_dir / "th_calib"
    query_dir = place_dir / "query"
    unused_dir = query_dir / "_unused"

    if not ALLOW_APPEND_OUTPUT and any(d.exists() and any(d.iterdir()) for d in [ref_dir, th_dir, query_dir]):
        raise RuntimeError(f"Output not empty for place {place_dir.name}. Set ALLOW_APPEND_OUTPUT=True or clear outputs.")

    rng = random.Random(SEED + int(place_dir.name) if place_dir.name.isdigit() else SEED)
    rng.shuffle(nomal_ref_pool)
    rng.shuffle(nomal_q_pool)
    rng.shuffle(abn_q_pool)

    # A) ref/th split은 nomal_ref_pool에서만
    N = len(nomal_ref_pool)
    n_ref = int(N * REF_RATIO)
    n_th  = int(N * TH_RATIO)
    nomal_ref = nomal_ref_pool[:n_ref]
    nomal_th  = nomal_ref_pool[n_ref:n_ref + n_th]
    nomal_ref_leftover = nomal_ref_pool[n_ref + n_th:]

    # B) query 정상 후보는 nomal_for_query에서 QUERY_NORM_RATIO만큼 사용(나머지 unused)
    QN = len(nomal_q_pool)
    n_qn_use = int(QN * QUERY_NORM_RATIO)
    nomal_q = nomal_q_pool[:n_qn_use]
    nomal_q_leftover = nomal_q_pool[n_qn_use:]

    # C) query 비정상 후보는 전부
    abn_q = abn_q_pool[:]

    print(f"[SPLIT] ref={len(nomal_ref)}, th={len(nomal_th)}, query_nomal_pre={len(nomal_q)}, query_abn_pre={len(abn_q)}")

    # D) query 1:1 밸런싱
    extras: List[Path] = []
    if BALANCE_QUERY_1TO1 and nomal_q and abn_q:
        m = min(len(nomal_q), len(abn_q))
        extras += nomal_q[m:] + abn_q[m:]
        nomal_q = nomal_q[:m]
        abn_q   = abn_q[:m]
        print(f"[BALANCE query] keep nomal:abn = {m}:{m}, extras_to_unused(additional)={len(extras)}")
    else:
        print("[BALANCE query] skipped (one side empty or BALANCE_QUERY_1TO1=False)")

    # E) 복제(COPY)
    copied_ref = [safe_copy(p, ref_dir)   for p in nomal_ref]
    copied_th  = [safe_copy(p, th_dir)    for p in nomal_th]
    copied_qn  = [safe_copy(p, query_dir) for p in nomal_q]
    copied_qa  = [safe_copy(p, query_dir) for p in abn_q]

    leftovers = nomal_ref_leftover + nomal_q_leftover + extras
    for p in leftovers:
        safe_copy(p, unused_dir)

    # F) 인덱스 “place 전역 유니크” 리네임
    # 순서: ref(nomal) -> th(nomal) -> query(nomal) -> query(unnormal)
    idx = 0

    # ⚠️ append 허용 시 기존 파일도 같이 rename되며 번호가 재정렬됨.
    #     (원하면 "이번에 복제한 것만 rename" 방식으로 바꿀 수 있음)
    idx = safe_rename_in_one_dir_with_start(list_files_flat(ref_dir),   prefix=OUT_N_PREFIX, start_idx=idx)
    idx = safe_rename_in_one_dir_with_start(list_files_flat(th_dir),    prefix=OUT_N_PREFIX, start_idx=idx)

    # query는 이번에 복제한 그룹 기준으로 rename (섞임 방지)
    qn_after = [p for p in copied_qn if (not APPLY) or p.parent == query_dir]
    qa_after = [p for p in copied_qa if (not APPLY) or p.parent == query_dir]

    idx = safe_rename_in_one_dir_with_start(qn_after, prefix=OUT_N_PREFIX, start_idx=idx)
    idx = safe_rename_in_one_dir_with_start(qa_after, prefix=OUT_A_PREFIX, start_idx=idx)

    print(f"[DONE place {place_dir.name}] final_next_index={idx}")

def main():
    root = ROOT.resolve()
    print(f"ROOT: {root}")
    print(f"MODE: {'APPLY' if APPLY else 'DRY-RUN'}")
    print(f"Ratios: ref={REF_RATIO}, th={TH_RATIO}, query_n_use={QUERY_NORM_RATIO}")
    print(f"BALANCE_QUERY_1TO1: {BALANCE_QUERY_1TO1}")

    places = iter_place_dirs(root)
    if not places:
        raise RuntimeError(f"No place directories found under {root}")

    for place_dir in places:
        rebalance_one_place(place_dir)

    print("\nALL DONE.")

if __name__ == "__main__":
    main()