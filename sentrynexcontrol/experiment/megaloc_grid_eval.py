import os
import re
import cv2
import uuid
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm


# =========================================================
# MegaLoc Wrapper
# =========================================================
class MegaLocWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.model = torch.hub.load(
            "gmberton/MegaLoc",
            "get_trained_model",
            trust_repo=True
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.tfm = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode_image(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tfm(img).unsqueeze(0).to(self.device)

        feat = self.model(x)

        if isinstance(feat, dict):
            feat = feat.get("global", feat.get("descriptor", feat))

        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0)


# =========================================================
# Utils
# =========================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FNAME_PATTERN = re.compile(
    r"(?P<mode>BANK|QUERY)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_id(?P<uid>[A-Za-z0-9_-]+)\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE
)


def is_image_file(p: Path):
    return p.suffix.lower() in IMG_EXTS


def parse_filename(name: str):
    m = FNAME_PATTERN.match(name)
    if m is None:
        return None
    return {
        "mode": m.group("mode").upper(),
        "grid_x": int(m.group("x")),
        "grid_y": int(m.group("y")),
        "uid": m.group("uid"),
    }


def cosine_search(query_feats, db_feats, topk=5):
    sims = query_feats @ db_feats.T
    topk_idx = np.argsort(-sims, axis=1)[:, :topk]
    topk_sims = np.take_along_axis(sims, topk_idx, axis=1)
    return topk_idx, topk_sims


# =========================================================
# Collect mode
# =========================================================
def run_collect(args):
    save_root = Path(args.save_root)
    mode = args.mode.upper()
    if mode not in {"BANK", "QUERY"}:
        raise ValueError("--mode must be BANK or QUERY")

    out_dir = save_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera: {args.camera_id}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    current_x = args.start_x
    current_y = args.start_y

    print("=" * 60)
    print(f"[COLLECT MODE] {mode}")
    print("조작법:")
    print("  c : 현재 프레임 저장")
    print("  g : 좌표 다시 입력")
    print("  w/s : y -1 / +1")
    print("  a/d : x -1 / +1")
    print("  q : 종료")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] camera frame read failed")
            continue

        vis = frame.copy()
        text1 = f"MODE: {mode}"
        text2 = f"GRID: x={current_x}, y={current_y}"
        text3 = "c=capture | g=input grid | w/a/s/d=move | q=quit"

        cv2.putText(vis, text1, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(vis, text2, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(vis, text3, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("collect", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            current_x -= 1

        elif key == ord("d"):
            current_x += 1

        elif key == ord("w"):
            current_y -= 1

        elif key == ord("s"):
            current_y += 1

        elif key == ord("g"):
            try:
                x_in = input("\ninput grid x: ").strip()
                y_in = input("input grid y: ").strip()
                current_x = int(x_in)
                current_y = int(y_in)
            except Exception:
                print("[WARN] invalid input, keep previous grid")

        elif key == ord("c"):
            uid = uuid.uuid4().hex[:8]
            fname = f"{mode}_x{current_x}_y{current_y}_id{uid}.jpg"
            out_path = out_dir / fname
            ok = cv2.imwrite(str(out_path), frame)
            if ok:
                print(f"[SAVED] {out_path}")
            else:
                print(f"[WARN] failed to save: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# Build manifest from saved images
# =========================================================
def run_build_manifest(args):
    save_root = Path(args.save_root)
    rows = []

    for mode in ["BANK", "QUERY"]:
        mode_dir = save_root / mode
        if not mode_dir.exists():
            continue

        for p in sorted(mode_dir.iterdir()):
            if not p.is_file() or not is_image_file(p):
                continue

            meta = parse_filename(p.name)
            if meta is None:
                continue

            rows.append({
                "image_path": str(p.resolve()),
                "mode": meta["mode"],
                "grid_x": meta["grid_x"],
                "grid_y": meta["grid_y"],
                "uid": meta["uid"],
            })

    if len(rows) == 0:
        print("[ERROR] no valid images found")
        return

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] manifest saved: {out_csv}")
    print(df["mode"].value_counts())


# =========================================================
# Feature extraction
# =========================================================
@torch.no_grad()
def extract_features(model, df):
    feats = []
    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract feats"):
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        feat = model.encode_image(img)
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()

        feat = feat.astype(np.float32)
        feat /= (np.linalg.norm(feat) + 1e-12)

        feats.append(feat)
        valid_rows.append(row.to_dict())

    if len(feats) == 0:
        raise RuntimeError("no valid features extracted")

    feats = np.stack(feats, axis=0)
    out_df = pd.DataFrame(valid_rows)
    return feats, out_df


# =========================================================
# Eval
# =========================================================
def run_eval(args):
    manifest_csv = Path(args.manifest_csv)
    df = pd.read_csv(manifest_csv)

    required_cols = {"image_path", "mode", "grid_x", "grid_y", "uid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    bank_df = df[df["mode"] == "BANK"].reset_index(drop=True)
    query_df = df[df["mode"] == "QUERY"].reset_index(drop=True)

    if len(bank_df) == 0:
        raise ValueError("BANK images not found")
    if len(query_df) == 0:
        raise ValueError("QUERY images not found")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda not available, fallback to cpu")
        device = "cpu"

    print(f"[INFO] device={device}")
    print(f"[INFO] num_bank={len(bank_df)}")
    print(f"[INFO] num_query={len(query_df)}")

    model = MegaLocWrapper(device=device)

    bank_feats, bank_df = extract_features(model, bank_df)
    query_feats, query_df = extract_features(model, query_df)

    topk = args.topk
    topk_idx, topk_sims = cosine_search(query_feats, bank_feats, topk=topk)

    per_query_rows = []
    success = {k: 0 for k in range(1, topk + 1)}

    for qi in range(len(query_df)):
        qrow = query_df.iloc[qi]
        qx = int(qrow["grid_x"])
        qy = int(qrow["grid_y"])

        candidates = []
        hit_rank = None

        for rank in range(topk):
            bi = topk_idx[qi, rank]
            brow = bank_df.iloc[bi]

            bx = int(brow["grid_x"])
            by = int(brow["grid_y"])
            sim = float(topk_sims[qi, rank])

            is_match = (qx == bx) and (qy == by)
            if is_match and hit_rank is None:
                hit_rank = rank + 1

            candidates.append({
                "rank": rank + 1,
                "bank_image_path": brow["image_path"],
                "bank_grid_x": bx,
                "bank_grid_y": by,
                "sim": sim,
                "is_match": is_match,
            })

        if hit_rank is not None:
            for k in range(hit_rank, topk + 1):
                success[k] += 1

        top1 = candidates[0]
        per_query_rows.append({
            "query_image_path": qrow["image_path"],
            "query_grid_x": qx,
            "query_grid_y": qy,
            "top1_bank_image_path": top1["bank_image_path"],
            "top1_bank_grid_x": top1["bank_grid_x"],
            "top1_bank_grid_y": top1["bank_grid_y"],
            "top1_sim": top1["sim"],
            "top1_is_match": top1["is_match"],
            "hit_rank": hit_rank if hit_rank is not None else -1,
            "topk_candidates": json.dumps(candidates, ensure_ascii=False),
        })

    num_query = len(query_df)
    summary = {
        "num_bank": int(len(bank_df)),
        "num_query": int(num_query),
    }

    for k in range(1, topk + 1):
        summary[f"Recall@{k}"] = float(success[k] / max(num_query, 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    per_query_csv = out_dir / "per_query_results.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(per_query_rows).to_csv(per_query_csv, index=False)

    print("\n================ EVAL SUMMARY ================")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=============================================\n")

    print(f"[OK] saved: {summary_path}")
    print(f"[OK] saved: {per_query_csv}")


# =========================================================
# Optional visualization
# =========================================================
def draw_text_block(img, lines, start_xy=(10, 20), line_gap=25):
    x, y = start_xy
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def safe_imread(path, resize_wh=None):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if resize_wh is not None:
        img = cv2.resize(img, resize_wh)
    return img


def run_vis(args):
    df = pd.read_csv(args.result_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.num_samples, len(df))
    sample_df = df.head(n)

    for i, row in sample_df.iterrows():
        qimg = safe_imread(row["query_image_path"], (320, 240))
        bimg = safe_imread(row["top1_bank_image_path"], (320, 240))
        if qimg is None or bimg is None:
            continue

        qimg = draw_text_block(qimg, [
            "QUERY",
            f"x={row['query_grid_x']} y={row['query_grid_y']}",
        ])

        bimg = draw_text_block(bimg, [
            "TOP1 BANK",
            f"x={row['top1_bank_grid_x']} y={row['top1_bank_grid_y']}",
            f"sim={row['top1_sim']:.4f}",
            f"match={row['top1_is_match']}",
        ])

        canvas = np.concatenate([qimg, bimg], axis=1)
        out_path = out_dir / f"vis_{i:04d}.jpg"
        cv2.imwrite(str(out_path), canvas)

    print(f"[OK] visualizations saved to: {out_dir}")


# =========================================================
# Main
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser("MegaLoc grid retrieval eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="interactive capture for BANK/QUERY")
    p.add_argument("--save-root", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["BANK", "QUERY", "bank", "query"])
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--start-x", type=int, default=0)
    p.add_argument("--start-y", type=int, default=0)
    p.set_defaults(func=run_collect)

    p = sub.add_parser("build-manifest", help="scan saved images and create csv")
    p.add_argument("--save-root", type=str, required=True)
    p.add_argument("--out-csv", type=str, required=True)
    p.set_defaults(func=run_build_manifest)

    p = sub.add_parser("eval", help="MegaLoc retrieval evaluation")
    p.add_argument("--manifest-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--topk", type=int, default=5)
    p.set_defaults(func=run_eval)

    p = sub.add_parser("vis", help="save simple query/top1 visualization")
    p.add_argument("--result-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=50)
    p.set_defaults(func=run_vis)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()