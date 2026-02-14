import random
import shutil
from pathlib import Path


# =========================
# 설정 (여기만 수정)
# =========================
# 이 폴더 바로 아래에 candle/, capsules/, ... 가 있어야 함
SRC_ROOT = Path("/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank/12/VisA_20220922")

DST_ROOT = Path("/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank")

SEED = 0
COPY_MODE = "copy"   # "copy" or "symlink"
CLEAN_PLACE = True   # 기존 place 폴더 삭제 후 재생성

# Normal split 비율
BANK_RATIO = 0.60    # normal 중 bank로 갈 비율
CALIB_RATIO = 0.30   # (bank로 빠지고 남은 normal) 중 th_calib 비율

# query에서 이상/정상 비율 (1.0 = 1:1)
QUERY_POS_NEG_RATIO = 1.0

# VisA raw 내부 경로
NORMAL_SUBDIR = Path("Data/Images/Normal")
ANOM_SUBDIR   = Path("Data/Images/Anomaly")

# candle 말고 다른 잡폴더가 섞여있으면 여기에 추가해서 제외
SKIP_DIRS = {"split_csv", "utils", "figures", ".git"}
# =========================


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(p: Path):
    return p.is_file() and (p.suffix.lower() in IMG_EXTS)


def collect_images(dir_path: Path):
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if is_image(p)])


def safe_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)


def convert_one_place(place_dir: Path):
    place = place_dir.name
    out_place = DST_ROOT / place

    bank_dir = out_place / "bank"
    calib_dir = out_place / "th_calib"
    query_dir = out_place / "query"

    do_copy = (COPY_MODE == "copy")
    rng = random.Random(SEED)

    if CLEAN_PLACE and out_place.exists():
        shutil.rmtree(out_place)

    # ---- VisA raw paths ----
    normal_dir = place_dir / NORMAL_SUBDIR
    anom_dir = place_dir / ANOM_SUBDIR

    normal_imgs = collect_images(normal_dir)
    anom_imgs = collect_images(anom_dir)

    # 방어: 폴더 구조가 기대와 다르면 바로 표시
    if len(normal_imgs) == 0 and len(anom_imgs) == 0:
        return {
            "place": place,
            "n_bank": 0,
            "n_calib": 0,
            "n_query_good": 0,
            "n_query_abn": 0,
            "warn": f"no images found under {normal_dir} / {anom_dir}",
        }

    rows = []

    # ----------------------
    # 1) normal split: bank / (rest)
    # ----------------------
    rng.shuffle(normal_imgs)
    n_bank = int(round(len(normal_imgs) * BANK_RATIO))
    bank_imgs = normal_imgs[:n_bank]
    rest_normals = normal_imgs[n_bank:]

    # ----------------------
    # 2) th_calib + query(normal) from remaining normals
    # ----------------------
    rng.shuffle(rest_normals)
    n_calib = int(round(len(rest_normals) * CALIB_RATIO))
    calib_imgs = rest_normals[:n_calib]
    query_good_imgs = rest_normals[n_calib:]

    for idx, p in enumerate(sorted(bank_imgs)):
        dst = bank_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((str(p), str(dst)))

    for idx, p in enumerate(sorted(calib_imgs)):
        dst = calib_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((str(p), str(dst)))

    for idx, p in enumerate(sorted(query_good_imgs)):
        dst = query_dir / f"nomal_{idx:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((str(p), str(dst)))

    # ----------------------
    # 3) anomaly -> query(unnormal) (정상 개수에 맞춰 샘플링)
    # ----------------------
    target_abn = int(round(len(query_good_imgs) * QUERY_POS_NEG_RATIO))
    rng.shuffle(anom_imgs)
    defect_sel = anom_imgs[:min(target_abn, len(anom_imgs))]

    for idx_abn, p in enumerate(sorted(defect_sel)):
        dst = query_dir / f"unnormal_{idx_abn:06d}{p.suffix.lower()}"
        safe_copy(p, dst, do_copy)
        rows.append((str(p), str(dst)))

    # ----------------------
    # symlink mode
    # ----------------------
    if COPY_MODE == "symlink":
        for src, dst in rows:
            src_p = Path(src)
            dst_p = Path(dst)
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if not dst_p.exists():
                dst_p.symlink_to(src_p.resolve())

    return {
        "place": place,
        "n_bank": len(bank_imgs),
        "n_calib": len(calib_imgs),
        "n_query_good": len(query_good_imgs),
        "n_query_abn": len(defect_sel),
        "warn": None,
    }


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    places = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir() and p.name not in SKIP_DIRS])
    if not places:
        raise RuntimeError(f"No object dirs found under: {SRC_ROOT}")

    total_bank = total_calib = total_qgood = total_qabn = 0
    n_warn = 0

    for place_dir in places:
        st = convert_one_place(place_dir)
        print(f"[+] converting {place_dir.name}")
        if st["warn"]:
            n_warn += 1
            print(f"    [WARN] {st['warn']}")
        print(
            f"    -> bank={st['n_bank']} calib={st['n_calib']} "
            f"query_good={st['n_query_good']} query_abn={st['n_query_abn']} | {DST_ROOT / st['place']}"
        )
        total_bank += st["n_bank"]
        total_calib += st["n_calib"]
        total_qgood += st["n_query_good"]
        total_qabn += st["n_query_abn"]

    print("[DONE]")
    if n_warn:
        print(f"[WARN] {n_warn} places had missing structure/images.")
    print(f"[SUM] bank={total_bank} calib={total_calib} query_good={total_qgood} query_abn={total_qabn}")


if __name__ == "__main__":
    main()
