import os
import json
import cv2
import matplotlib.pyplot as plt


from DINO_change_map import (compute_change_map, 
                             extract_rois_from_change_map , 
                             prepare_vlm_pairs_from_rois , 
                             visualize_vlm_pairs, 
                             draw_rois_on_image
                            )
from vlm_Reasoning import VLMWrapper


def make_vlm_input(ref_path,query_path):

    change_map, overlay_q, overlay_r, score, q_rgb_518, r_rgb_518 = compute_change_map(
        query_path=query_path,
        ref_path=ref_path,
        top_p=None,
        ms_pool_ks=(1,2,4),
        ms_pool_type="avg",
        ms_agg="top2mean",
    )

    rois = extract_rois_from_change_map(change_map, min_area=150)
    print("num rois (DINO):", len(rois))

    candidates = prepare_vlm_pairs_from_rois(
        q_rgb=q_rgb_518,
        r_rgb=r_rgb_518,
        rois=rois,
        expand_scale=2.0,
        img_size=518,
        out_h=336,
        border=4,
    )

    debug = {
        "change_map": change_map,
        "overlay_q": overlay_q,
        "overlay_r": overlay_r,
        "score": score,
        "q_rgb": q_rgb_518,
        "r_rgb": r_rgb_518,
        "rois": rois,
    }

    return candidates , debug

prompt = []
max_new_tokens = []
label_texts = []
show_viz = True


if __name__ == "__main__":
    ref_path = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/000/RGB/1_01.png"
    qry_path = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/000/RGB/2_01.png"

    candidates , debug = make_vlm_input(ref_path,qry_path)
    print(len(candidates))


    vlm = VLMWrapper().load(        
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        device_map="auto",
        torch_dtype="auto"
        )
    
    candidates_with_vlm = vlm.infer_candidates(
        candidates,
        prompt=prompt,
        topk=None,
        max_new_tokens=max_new_tokens,
        label_texts=label_texts,
    )

    for c in candidates_with_vlm:
        print(
            f"  - rank={c.get('rank')} "
            f"label={c.get('label')} "
            f"dino={c.get('score',0):.3f} "
            f"vlm={c.get('vlm_score',0):.3f} "
            f"best={c.get('vlm_extra',{}).get('best_label','')} "
            f"bbox_exp={c.get('bbox_exp')}"
        )

    if show_viz == True :
        droi_vis = draw_rois_on_image(debug['q_rgb'], debug['rois'], color=(0,255,0), thickness=2)
        visualize_vlm_pairs(candidates, max_show=10, cols=5)

        plt.figure(); plt.title("rois"); plt.imshow(droi_vis); plt.colorbar(); plt.axis("off")
        plt.figure(); plt.title("change_map"); plt.imshow(debug['change_map']); plt.axis("off")
        plt.figure(); plt.title("Query + heatmap"); plt.imshow(debug['overlay_q']); plt.axis("off")
        plt.figure(); plt.title("ref + heatmap"); plt.imshow(debug['overlay_r']); plt.axis("off")
        plt.show()
