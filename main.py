import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from clip_embed import CLIPTextEmbedder


from DINO_change_map import (compute_change_map, 
                             pick_local_maxima_topk , 
                             prepare_vlm_pairs_from_rois ,
                             peaks_to_rois, 
                             visualize_vlm_pairs, 
                             draw_rois_on_image
                            ) # make roi

from vlm_Reasoning import VLMWrapper #Vqa

from q_file import QUESTIONS #q


def make_vlm_input(ref_path,query_path):

    change_map, overlay_q, overlay_r, score, q_rgb_518, r_rgb_518 = compute_change_map(
        query_path=query_path,
        ref_path=ref_path,
        top_p=None,
        ms_pool_ks=(1,2,4),
        ms_pool_type="avg",
        ms_agg="top2mean",
    )
    
    #find peak 
    peaks = pick_local_maxima_topk(change_map,topk=3,k=3)

    rois = peaks_to_rois(change_map,peaks)
    print("num rois (DINO):", len(rois))

    # VLM 입력(pair) 만들기
    candidates = prepare_vlm_pairs_from_rois(
        q_rgb=q_rgb_518,
        r_rgb=r_rgb_518,
        rois=rois,
        expand_scale=1.6,
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


def norm_yesno(text: str) -> str:
    if not text:
        return "unknown"
    t = text.lower().strip()

    # 강한 신호 먼저
    if re.search(r"\byes\b", t): return "yes"
    if re.search(r"\bno\b", t): return "no"
    if "unknown" in t or "not sure" in t or "cannot" in t or "can't" in t: return "unknown"

    # 모델이 문장으로 답할 때: 부정 표현 잡기
    if "no " in t or "there is no" in t or "not" in t:
        return "no"

    return "unknown"


def classify_roi_4class(ref_ans: dict, qry_ans: dict, clip_embed=None,
                        scene_thr=0.25, objects_thr=0.35):
    """
    ref_ans/qry_ans: {question_id: answer_text}
    clip_embed: CLIPTextEmbedder 
    """

    # --- 문 열림/닫힘 확인
    door_ref = [norm_yesno(ref_ans.get("door_1","")), norm_yesno(ref_ans.get("door_2",""))]
    door_qry = [norm_yesno(qry_ans.get("door_1","")), norm_yesno(qry_ans.get("door_2",""))]
    door_ref_yes = sum(x=="yes" for x in door_ref)
    door_qry_yes = sum(x=="yes" for x in door_qry)

    if door_ref_yes == 0 and door_qry_yes >= 1:
        return "door"

    # --- hazard 판정 
    dmg_qry = [norm_yesno(qry_ans.get("damage_1","")), norm_yesno(qry_ans.get("damage_2",""))]
    if any(x=="yes" for x in dmg_qry):
        return "hazard"

    # --- 3) object_change 판정 ---
    # (a) objects 텍스트 비교 (가장 단순)
    obj_ref_text = (ref_ans.get("objects_1","") + " | " + ref_ans.get("objects_2","")).strip()
    obj_qry_text = (qry_ans.get("objects_1","") + " | " + qry_ans.get("objects_2","")).strip()

    # (b) scene 텍스트 비교
    scene_ref_text = (ref_ans.get("scene_1","") + " | " + ref_ans.get("scene_2","")).strip()
    scene_qry_text = (qry_ans.get("scene_1","") + " | " + qry_ans.get("scene_2","")).strip()

    # embedding 있으면 cosine distance로 판단(더 안정적)
    e_obj_ref = clip_embed("OBJ: " + obj_ref_text)
    e_obj_qry = clip_embed("OBJ: " + obj_qry_text)
    d_obj = 1.0 - float(e_obj_ref @ e_obj_qry)

    e_sc_ref = clip_embed("SCENE: " + scene_ref_text)
    e_sc_qry = clip_embed("SCENE: " + scene_qry_text)
    d_sc = 1.0 - float(e_sc_ref @ e_sc_qry)

    if (d_obj > objects_thr) or (d_sc > scene_thr):
        return "object_change"

    return "no_change"




show_viz = True


if __name__ == "__main__":
    qry_path = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/001/RGB/1_00.png"
    ref_path   = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/001/RGB/2_00.png"

    candidates , debug = make_vlm_input(ref_path,qry_path)
    print(len(candidates))

    clip_embeder = CLIPTextEmbedder()

    vlm = VLMWrapper().load(        
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        device_map="auto",
        torch_dtype="auto"
        )
    
    ref_ans = {}
    qry_ans = {}

    for roi_i, cand in enumerate(candidates, start=1):
        qry_crop = cand["qry_crop"]   
        ref_crop = cand["ref_crop"]   

        print(f"\n================ ROI {roi_i} ================")
        for q in QUESTIONS:
            qtext = q["text"]

            prompt_ref = (
                "Answer in one short phrase (max 5 words). If unsure, answer 'unknown'. No explanation.\n"
                f"Question: {qtext}\n"
                "Answer for the REFERENCE image only."
            )
            prompt_qry = (
                "Answer in one short phrase (max 5 words). If unsure, answer 'unknown'. No explanation.\n"
                f"Question: {qtext}\n"
                "Answer for the CURRENT image only."
            )

            # ref만 묻기: (ref, ref)로 넣어서 단일 이미지처럼 사용
            ans_ref = vlm.infer_image(ref_crop, qtext)
            ans_qry = vlm.infer_image(qry_crop, qtext)

            print(f"- {q['id']}: {qtext}")
            print(f"    REF: {ans_ref}")
            print(f"    QRY: {ans_qry}")
            ref_ans[q["id"]] = ans_ref
            qry_ans[q["id"]] = qry_ans

        cls = classify_roi_4class(ref_ans,qry_ans,clip_embeder)
        print(f"class :  {cls}")
                    

    if show_viz == True :
        droi_vis = draw_rois_on_image(debug['q_rgb'], debug['rois'], color=(0,255,0), thickness=2)
        visualize_vlm_pairs(candidates, max_show=10, cols=5)

        plt.figure(); plt.title("rois"); plt.imshow(droi_vis); plt.colorbar(); plt.axis("off")
        plt.figure(); plt.title("change_map"); plt.imshow(debug['change_map']); plt.axis("off")
        plt.figure(); plt.title("Query + heatmap"); plt.imshow(debug['overlay_q']); plt.axis("off")
        plt.figure(); plt.title("ref + heatmap"); plt.imshow(debug['overlay_r']); plt.axis("off")
        plt.show()
