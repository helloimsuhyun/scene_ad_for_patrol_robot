import os, glob, csv
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_dinov2(device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # small + fast
    model.eval().to(device)
    return model


# -------------------------
# 2) Image -> embedding (patch mean pooling)
# -------------------------
@torch.no_grad()
def image_to_embed(model, img_path, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,H,W]

    feats = model.forward_features(x)
    patch = feats["x_norm_patchtokens"]   # [1, N, D]
    cls = feats["x_norm_clstoken"] # [1, D]

    patch_emb = patch.mean(dim=1)               # [1, D]

    patch_emb = F.normalize(patch_emb, dim=-1)
    cls_emb   = F.normalize(cls,   dim=-1) # 크기 1 벡터로 정규화

    alpha = 0.3
    emb = (1 - alpha) * patch_emb + alpha * cls_emb #mix
    emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).cpu()           # [D]


# -------------------------
# 3) Build reference bank
# -------------------------
def build_ref_bank(model, ref_paths, device="cuda"):
    bank = []
    for p in ref_paths:
        bank.append(image_to_embed(model, p, device=device))
    return torch.stack(bank, dim=0)  # [K, D]


# -------------------------
# 4) Score query + return best ref
# -------------------------
@torch.no_grad()
def score_query_with_best_ref(model, bank, ref_paths, query_path, device="cuda"):
    q = image_to_embed(model, query_path, device=device)  # [D]
    sims = bank @ q                                       # [K]
    best_idx = int(torch.argmax(sims).item())
    best_sim = float(sims[best_idx].item())
    score = 1.0 - best_sim
    best_ref_path = ref_paths[best_idx]
    return score, best_sim, best_idx, best_ref_path


# -------------------------
# 5) Save side-by-side visualization (ref | query)
# -------------------------
def save_pair_vis(best_ref_path, query_path, out_path, score, best_sim, best_idx, vis_size=512):
    # load and resize for visualization
    ref_img = Image.open(best_ref_path).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)
    qry_img = Image.open(query_path).convert("RGB").resize((vis_size, vis_size), Image.BICUBIC)

    # concatenate horizontally
    canvas = Image.new("RGB", (vis_size * 2, vis_size))
    canvas.paste(ref_img, (0, 0))
    canvas.paste(qry_img, (vis_size, 0))

    # simple text overlay (top-left)
    draw = ImageDraw.Draw(canvas)
    txt = f"best_ref_idx={best_idx}  sim={best_sim:.4f}  score={score:.4f}"
    # black bg rectangle for readability
    draw.rectangle([0, 0, vis_size * 2, 22], fill=(0, 0, 0))
    draw.text((5, 4), txt, fill=(255, 255, 255))

    canvas.save(out_path)


def list_images(folder: str):
    allp = sorted(glob.glob(os.path.join(folder, "*")))
    return [p for p in allp if os.path.splitext(p)[1].lower() in IMG_EXTS]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dinov2(device=device)

    # ---- set your folders ----
    REF_DIR   = "/home/choisuhyun/scene_ad_for_patrol_robot/data/ref"   # multiple refs
    QUERY_DIR = "/home/choisuhyun/scene_ad_for_patrol_robot/data/q"      # frames/shots

    # ---- output folder (save everything here) ----
    OUT_DIR = os.path.join(QUERY_DIR, "_infer_dino_out")
    PAIR_DIR = os.path.join(OUT_DIR, "pairs")
    ensure_dir(OUT_DIR)
    ensure_dir(PAIR_DIR)

    ref_paths = list_images(REF_DIR)
    qry_paths = list_images(QUERY_DIR)

    assert len(ref_paths) > 0, "No ref images found"
    assert len(qry_paths) > 0, "No query images found"

    bank = build_ref_bank(model, ref_paths, device=device)

    results = []
    for qi, qpath in enumerate(qry_paths):
        score, best_sim, best_idx, best_ref_path = score_query_with_best_ref(
            model, bank, ref_paths, qpath, device=device
        )

        results.append({
            "q_index": qi,
            "q_path": qpath,
            "score": score,
            "best_cos_sim": best_sim,
            "best_ref_idx": best_idx,
            "best_ref_path": best_ref_path,
        })

        # log
        print(f"[{qi:04d}] score={score:.4f} sim={best_sim:.4f} "
              f"best_ref_idx={best_idx:03d} best_ref={os.path.basename(best_ref_path)} "
              f"q={os.path.basename(qpath)}")

        # save pair visualization
        out_pair = os.path.join(
            PAIR_DIR,
            f"{qi:04d}_ref{best_idx:03d}__{os.path.splitext(os.path.basename(qpath))[0]}__score{score:.4f}.png"
        )
        save_pair_vis(best_ref_path, qpath, out_pair, score, best_sim, best_idx, vis_size=512)

    # -------------------------
    # Save CSV
    # -------------------------
    out_csv = os.path.join(OUT_DIR, "dino_scores_with_best_ref.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["q_index","score","best_cos_sim","best_ref_idx","q_path","best_ref_path","pair_vis_path"])
        for r in results:
            qi = r["q_index"]
            # find the saved pair (reconstruct name prefix)
            # (simple: just store folder + index prefix; you can change if you want exact file)
            pair_prefix = f"{qi:04d}_"
            w.writerow([
                r["q_index"],
                f"{r['score']:.6f}",
                f"{r['best_cos_sim']:.6f}",
                r["best_ref_idx"],
                r["q_path"],
                r["best_ref_path"],
                os.path.join(PAIR_DIR, pair_prefix + "...")
            ])
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved pair visualizations: {PAIR_DIR}")

    # -------------------------
    # Plot score vs query index (save)
    # -------------------------
    xs = [r["q_index"] for r in results]
    ys = [r["score"] for r in results]

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Query index")
    plt.ylabel("Score = 1 - best cosine similarity")
    plt.title("Scene change score per query (DINOv2 patch-mean)")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "dino_score_plot.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot: {out_png}")

    # 필요하면 화면에도 띄우기
    # plt.show()


if __name__ == "__main__":
    main()
