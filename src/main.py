from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from dino_emb import load_model
from banker import load_bank_by_place
from distance import calibrate_place, infer_one


# -------------------------
# 설정
# -------------------------
BANK_ROOT = "/home/choisuhyun/scene_ad_for_patrol_robot/data/ref_bank"
PLC_IDX = "transistor" 
K = 3
PERCENTILE = 97


# -------------------------
# label rule (너 규칙 그대로)
#  - nomal*.png => 정상(0)
#  - 나머지 => 이상(1)
# -------------------------
def label_from_name(name: str) -> int:
    name = name.lower()
    if name.startswith("unnormal"):
        return 1   # 비정상
    if name.startswith("nomal"):
        return 0   # 정상
    return 1       # 기본값: 비정상


def compute_metrics_from_results(results):
    # results: list of (name, dist, pred_change_bool)
    y_true = np.array([label_from_name(n) for (n, _, __) in results], dtype=np.int32)
    y_pred = np.array([1 if pred else 0 for (_, __, pred) in results], dtype=np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")  # recall
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": acc,
        "Recall/TPR": tpr,
        "Precision": prec,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "N_total": n,
        "N_pos": int((y_true == 1).sum()),
        "N_neg": int((y_true == 0).sum()),
    }


# -------------------------
# k개 ref + query 시각화
# -------------------------
def save_pair_vis_k(topk_paths, query_path, out_path, score, topk_sims, is_change, vis_size=384):
    k = len(topk_paths)

    ref_imgs = [
        Image.open(p).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)
        for p in topk_paths
    ]
    qry_img = Image.open(query_path).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)

    # change면 query에 빨간 테두리
    if is_change:
        draw_q = ImageDraw.Draw(qry_img)
        draw_q.rectangle([0, 0, vis_size - 1, vis_size - 1], outline=(255, 0, 0), width=6)

    canvas_w = vis_size * (k + 1)
    canvas = Image.new("RGB", (canvas_w, vis_size))

    for i, ref_img in enumerate(ref_imgs):
        canvas.paste(ref_img, (i * vis_size, 0))
    canvas.paste(qry_img, (k * vis_size, 0))

    # 상단 텍스트
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas_w, 24], fill=(0, 0, 0))
    sim_txt = " | ".join([f"{s:.3f}" for s in topk_sims])
    txt = f"score={score:.4f}  sims=[{sim_txt}]"
    draw.text((5, 4), txt, fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# -------------------------
# main
# -------------------------
def main():
    model, device = load_model()

    # 1) threshold 계산 (bank/th_calib rebuild + threshold.json 생성)
    thr, scores, _ = calibrate_place(
        BANK_ROOT, PLC_IDX, model, device,
        k=K,
        percentile=PERCENTILE
    )
    print("Threshold:", thr)

    out_dir = Path(BANK_ROOT) / PLC_IDX / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) th_calib score curve (점+선)
    scores_arr = np.array(scores, dtype=np.float32)
    xs = np.arange(len(scores_arr))

    plt.figure()
    plt.plot(xs, scores_arr, marker="o")     # 점+선
    plt.axhline(thr)                         # thr 수평선
    plt.xlabel("th_calib index")
    plt.ylabel("score")
    plt.title(f"th_calib score curve (thr={thr:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "calib_score_curve.png", dpi=200)
    plt.close()
    print("Saved:", out_dir / "calib_score_curve.png")

    # bank path list (혹시 필요하면)
    _, bank_paths = load_bank_by_place(BANK_ROOT, PLC_IDX, mode="bank")
    bank_paths = bank_paths or []

    # 3) query 추론
    query_dir = Path(BANK_ROOT) / PLC_IDX / "query"
    query_imgs = sorted(query_dir.glob("*.png"))

    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    dists = []
    results = []  # (name, dist, is_change)

    for i, p in enumerate(query_imgs):
        # query는 PNG라 RGB로 읽히는데, infer_one은 BGR uint8을 가정하니까 변환
        img_rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        img_bgr = img_rgb[:, :, ::-1]

        dist, thr_used, is_change, topk_paths, topk_sims = infer_one(
            img_bgr, PLC_IDX, BANK_ROOT, model, device
        )

        dists.append(dist)
        results.append((p.name, float(dist), bool(is_change)))

        print(f"{p.name}  dist={dist:.4f}  thr={thr_used:.4f}  change={is_change}")

        # k ref + query 저장
        out_pair = pair_dir / f"{i:04d}_{p.stem}_chg{int(is_change)}.png"
        save_pair_vis_k(
            topk_paths=topk_paths,
            query_path=p,
            out_path=out_pair,
            score=dist,
            topk_sims=topk_sims,
            is_change=is_change,
            vis_size=384
        )

    print("Saved pairs to:", pair_dir)

    # 4) query dist curve (점+선)
    dists = np.array(dists, dtype=np.float32)
    xs = np.arange(len(dists))

    plt.figure()
    plt.plot(xs, dists, marker="o")          # 점+선
    plt.axhline(thr)                         # thr 수평선
    plt.xlabel("query index")
    plt.ylabel("dist")
    plt.title(f"query dist curve (thr={thr:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "query_dist_curve.png", dpi=200)
    plt.close()
    print("Saved:", out_dir / "query_dist_curve.png")

    # 5) metrics 출력
    m = compute_metrics_from_results(results)

    print("\n=== Confusion Matrix ===")
    print(f"TP={m['TP']}  FP={m['FP']}")
    print(f"FN={m['FN']}  TN={m['TN']}")

    print("\n=== Metrics ===")
    print(f"Accuracy  : {m['Accuracy']*100:.2f}%")
    print(f"Recall/TPR: {m['Recall/TPR']*100:.2f}%")
    print(f"Precision : {m['Precision']*100:.2f}%")
    print(f"F1        : {m['F1']*100:.2f}%")
    print(f"FPR       : {m['FPR']*100:.2f}%")
    print(f"FNR       : {m['FNR']*100:.2f}%")
    print(f"N_total={m['N_total']}  N_pos={m['N_pos']}  N_neg={m['N_neg']}")

    # FP/FN 파일 리스트
    fps = [n for (n, _, pred) in results if label_from_name(n) == 0 and pred]
    fns = [n for (n, _, pred) in results if label_from_name(n) == 1 and (not pred)]
    print("\nFP files:", fps)
    print("FN files:", fns)
    txt_path = out_dir / "metrics.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"place={PLC_IDX}\n")
        f.write(f"K={K}\n")
        f.write(f"PERCENTILE={PERCENTILE}\n")
        f.write(f"threshold={thr:.6f}\n\n")

        f.write("=== Confusion Matrix ===\n")
        f.write(f"TP={m['TP']}  FP={m['FP']}\n")
        f.write(f"FN={m['FN']}  TN={m['TN']}\n\n")

        f.write("=== Metrics ===\n")
        f.write(f"Accuracy  : {m['Accuracy']*100:.2f}%\n")
        f.write(f"Recall/TPR: {m['Recall/TPR']*100:.2f}%\n")
        f.write(f"Precision : {m['Precision']*100:.2f}%\n")
        f.write(f"F1        : {m['F1']*100:.2f}%\n")
        f.write(f"FPR       : {m['FPR']*100:.2f}%\n")
        f.write(f"FNR       : {m['FNR']*100:.2f}%\n")
        f.write(f"N_total={m['N_total']}  N_pos={m['N_pos']}  N_neg={m['N_neg']}\n\n")

        f.write("FP files:\n")
        for n in fps:
            f.write(f"  {n}\n")

        f.write("\nFN files:\n")
        for n in fns:
            f.write(f"  {n}\n")

    print("Saved:", txt_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
