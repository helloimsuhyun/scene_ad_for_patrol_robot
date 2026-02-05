import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from clip_embed import CLIPTextEmbedder

import re
from typing import Any, Dict, Optional


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

def get_answer_text(ans: Dict[str, Any], key: str) -> str:
    """
    ans[key]가 str이든 dict든 안전하게 answer 텍스트만 뽑는다.
    - str -> 그대로
    - dict -> ["answer"] 우선, 없으면 ["text"], 없으면 "" (혹은 str(value))
    """
    v = ans.get(key, "")
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # 너의 출력 포맷에 맞게 우선순위 설정
        if isinstance(v.get("answer", None), str):
            return v["answer"]
        if isinstance(v.get("text", None), str):
            return v["text"]
        # 마지막 방어
        return ""
    # list/number 등 들어오면 안전하게 문자열화(원하면 ""로 처리해도 됨)
    return str(v)

def norm_yesno(text: str) -> str:
    if not text:
        return "unknown"
    t = text.lower().strip()

    if re.search(r"\byes\b", t): return "yes"
    if re.search(r"\bno\b", t): return "no"
    if "unknown" in t or "not sure" in t or "cannot" in t or "can't" in t:
        return "unknown"
    if "no " in t or "there is no" in t or "not" in t:
        return "no"
    return "unknown"

def classify_roi_4class(ref_ans: dict, qry_ans: dict, clip_embed=None,
                        scene_thr=0.30, objects_thr=0.30):
    """
    ref_ans/qry_ans: {question_id: str | dict(answer=...)}
    clip_embed: callable(str)->embedding (optional)
    """

    # --- 1) 문 열림/닫힘 ---
    door_ref = [
        norm_yesno(get_answer_text(ref_ans, "door_1")),
        norm_yesno(get_answer_text(ref_ans, "door_2")),
    ]
    door_qry = [
        norm_yesno(get_answer_text(qry_ans, "door_1")),
        norm_yesno(get_answer_text(qry_ans, "door_2")),
    ]
    door_ref_yes = sum(x == "yes" for x in door_ref)
    door_qry_yes = sum(x == "yes" for x in door_qry)

    if door_ref_yes == 0 and door_qry_yes >= 1:
        return "door"

    # --- 2) hazard ---
    dmg_qry = [
        norm_yesno(get_answer_text(qry_ans, "damage_1")),
        norm_yesno(get_answer_text(qry_ans, "damage_2")),
    ]
    if any(x == "yes" for x in dmg_qry):
        return "hazard"

    # --- 3) object_change ---
    obj_ref_text = (get_answer_text(ref_ans, "objects_1") + " | " +
                    get_answer_text(ref_ans, "objects_2")).strip()
    obj_qry_text = (get_answer_text(qry_ans, "objects_1") + " | " +
                    get_answer_text(qry_ans, "objects_2")).strip()

    scene_ref_text = (get_answer_text(ref_ans, "scene_1") + " | " +
                      get_answer_text(ref_ans, "scene_2")).strip()
    scene_qry_text = (get_answer_text(qry_ans, "scene_1") + " | " +
                      get_answer_text(qry_ans, "scene_2")).strip()

    # embedding cosine distance 기반
    e_obj_ref = clip_embed("OBJ: " + obj_ref_text)
    e_obj_qry = clip_embed("OBJ: " + obj_qry_text)
    d_obj = 1.0 - float(e_obj_ref @ e_obj_qry)

    e_sc_ref = clip_embed("SCENE: " + scene_ref_text)
    e_sc_qry = clip_embed("SCENE: " + scene_qry_text)
    d_sc = 1.0 - float(e_sc_ref @ e_sc_qry)

    if (d_obj > objects_thr) or (d_sc > scene_thr):
        return "object_change"

    return "no_change"

# -------------최종 vis
def draw_rois_colored_with_labels(rgb_img, rois, classes, thickness=2):

    color_map = {
        "no_change":     (0, 255, 0),    # green
        "object_change": (255, 0, 0),    # blue
        "door":          (0, 255, 255),  # yellow
        "hazard":        (0, 0, 255),    # red
    }

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.9
    font_thickness = 1

    img_bgr = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)

    for i, (roi, cls) in enumerate(zip(rois, classes), start=1):
        if isinstance(roi, dict):
            x1, y1, x2, y2 = roi.get("x1"), roi.get("y1"), roi.get("x2"), roi.get("y2")
            if None in (x1, y1, x2, y2) and "bbox" in roi:
                x1, y1, x2, y2 = roi["bbox"]
        else:
            x1, y1, x2, y2 = roi

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        color = color_map.get(cls, (255, 255, 255))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        label = f"ROI{i}: {cls}"

        (tw, th), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        y_text = max(th + 2, y1 - 4)

        cv2.rectangle(
            img_bgr,
            (x1, y_text - th - baseline),
            (x1 + tw + 4, y_text + baseline),
            color,
            -1
        )

        cv2.putText(
            img_bgr,
            label,
            (x1 + 2, y_text),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)



show_viz = True


if __name__ == "__main__":
    qry_path = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/003/RGB/1_00.png"
    ref_path   = "/home/choisuhyun/scene_ad_for_patrol_robot/data/VL-CMU-CD/images/003/RGB/2_00.png"

    candidates , debug = make_vlm_input(ref_path,qry_path)
    print(len(candidates))

    clip_embeder = CLIPTextEmbedder()

    vlm = VLMWrapper().load(        
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        device_map="auto",
        torch_dtype="auto"
        )
    

    roi_classes = []

    for roi_i, cand in enumerate(candidates, start=1):
        ref_ans = {}
        qry_ans = {}
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
            IGNORE = "Ignore lighting changes, shadows, reflections, and viewpoint differences. Focus only on physical/object changes."
            qtext2 = f"{IGNORE}\n{qtext}"
            # ref만 묻기: (ref, ref)로 넣어서 단일 이미지처럼 사용
            ans_ref = vlm.infer_image(ref_crop, qtext2)
            ans_qry = vlm.infer_image(qry_crop, qtext2)

            print(f"- {q['id']}: {qtext}")
            print(f"    REF: {ans_ref}")
            print(f"    QRY: {ans_qry}")
            ref_ans[q["id"]] = ans_ref
            qry_ans[q["id"]] = ans_qry

        cls = classify_roi_4class(ref_ans,qry_ans,clip_embeder)
        roi_classes.append(cls)
        print(f"class :  {cls}")
                    

    if show_viz == True:
        droi_vis = draw_rois_colored_with_labels(debug["q_rgb"], debug["rois"], roi_classes, thickness=2)

        visualize_vlm_pairs(candidates, max_show=10, cols=5)

        plt.figure(); plt.title("rois (final output)"); plt.imshow(droi_vis); plt.axis("off")

        plt.figure(); plt.title("change_map"); plt.imshow(debug['change_map']); plt.axis("off")
        plt.figure(); plt.title("Query + heatmap"); plt.imshow(debug['overlay_q']); plt.axis("off")
        plt.figure(); plt.title("ref + heatmap"); plt.imshow(debug['overlay_r']); plt.axis("off")
        plt.show()

