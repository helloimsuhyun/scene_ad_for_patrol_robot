import random
import shutil
from pathlib import Path


# =========================
# 설정 (여기만 수정)
# =========================
SRC_ROOT = Path("/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank/ㅇㅇ")
DST_ROOT = Path("/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank")

CALIB_RATIO = 0.30
SEED = 0
COPY_MODE = "copy"   # "copy" or "symlink"
CLEAN_PLACE = True   # 기존 place 폴더 삭제 후 재생성

# query에서 정상:이상 비율 (반반이면 1.0)
QUERY_POS_NEG_RATIO = 1.0   # 이상/정상 비율 (1.0 = 1:1)
# =========================


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(p: Path):
    return p.suffix.lower() in IMG_EXTS


def safe_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)


def collect_images(dir_path: Path):
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and is_image(p)])


def convert_one_place(cls_dir: Path):
    place = cls_dir.name
    out_place = DST_ROOT / place

    bank_dir = out_place / "bank"
    calib_dir = out_place / "th_calib"
    query_dir = out_place / "query"

    do_copy = (COPY_MODE == "copy")
    rng = random.Random(SEED)

    if CLEAN_PLACE and out_place.exists():
        shutil.rmtree(out_place)

    rows = []

    # ----------------------
    # 1) bank = train/good
    # ----------------------
    ref_imgs = collect_images(cls_dir / "train" / "good")
    for idx, p in enumerate(ref_imgs):
        dst = bank_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((place, "bank", "0", str(p), str(dst)))

    # ----------------------
    # 2) th_calib + query(normal)
    # ----------------------
    test_good = collect_images(cls_dir / "test" / "good")
    rng.shuffle(test_good)

    n_calib = int(round(len(test_good) * CALIB_RATIO))
    calib_imgs = test_good[:n_calib]
    query_good_imgs = test_good[n_calib:]  # 이게 query 정상 후보

    for idx, p in enumerate(sorted(calib_imgs)):
        dst = calib_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((place, "th_calib", "0", str(p), str(dst)))

    for idx, p in enumerate(sorted(query_good_imgs)):
        dst = query_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((place, "query", "0", str(p), str(dst)))

    # ----------------------
    # 3) defect → query(unnormal)  (정상 개수에 맞춰 샘플링)
    # ----------------------
    defect_all = []
    test_dir = cls_dir / "test"
    if test_dir.exists():
        for defect_dir in sorted([d for d in test_dir.iterdir()
                                  if d.is_dir() and d.name != "good"]):
            defect_all += collect_images(defect_dir)

    # 목표 이상 개수 = 정상 개수 * 비율
    target_abn = int(round(len(query_good_imgs) * QUERY_POS_NEG_RATIO))

    # 이상이 너무 많으면 자르고, 너무 적으면 있는 만큼만 사용
    rng.shuffle(defect_all)
    defect_sel = defect_all[:min(target_abn, len(defect_all))]

    for idx_abn, p in enumerate(sorted(defect_sel)):
        dst = query_dir / f"unnormal_{idx_abn:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((place, "query", "1", str(p), str(dst)))

    # ----------------------
    # symlink 모드
    # ----------------------
    if COPY_MODE == "symlink":
        for place_, split, label, src, dst in rows:
            src_p = Path(src)
            dst_p = Path(dst)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if not dst_p.exists():
                dst_p.symlink_to(src_p.resolve())

    return rows


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    classes = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()])

    for cls in classes:
        print(f"[+] converting {cls.name}")
        convert_one_place(cls)

    print("[DONE]")


if __name__ == "__main__":
    main()
