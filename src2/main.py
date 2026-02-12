import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from clip_embed import CLIPTextEmbedder

import re
from typing import Any, Dict, Optional


from DINO_change_map_ori import (compute_change_map, 
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

def pair_yes(vlm_ans: Dict[str, str], base: str) -> str:
    """
    base_1, base_2 둘 중 하나라도 yes면 yes.
    둘 다 no면 no.
    그 외 unknown.
    """
    a1 = norm_yesno(vlm_ans.get(f"{base}_1", ""))
    a2 = norm_yesno(vlm_ans.get(f"{base}_2", ""))

    if a1 == "yes" or a2 == "yes":
        return "yes"
    if a1 == "no" and a2 == "no":
        return "no"
    return "unknown"


def norm_yesno(text: str) -> str:
    """
    VLM 답을 yes/no/unknown으로 정규화 (짧은 답/문장 답 모두 방어)
    """
    if not text:
        return "unknown"
    t = str(text).lower().strip()

    # 흔한 변형 정리
    t = t.replace("_", " ").replace("-", " ")

    # 1) 명시적 unknown
    if any(k in t for k in ["unknown", "not sure", "unsure", "cannot tell", "can't tell", "unclear", "hard to tell"]):
        return "unknown"

    # 2) 강한 yes 패턴 (yes/no 질문이라면 yes 우선)
    if re.search(r"\byes\b", t):
        return "yes"
    if any(k in t for k in ["changed", "different", "has changed", "there is a change", "a new object", "added", "removed", "moved"]):
        return "yes"

    # 3) 강한 no 패턴
    if re.search(r"\bno\b", t):
        return "no"
    if any(k in t for k in ["no change", "same", "identical", "unchanged", "nothing changed", "no difference"]):
        return "no"

    return "unknown"


def classify_roi_4class_from_compare_questions(vlm_ans: Dict[str, str]) -> str:
    # 2개 질문을 base로 통합
    change = pair_yes(vlm_ans, "change_detected")
    door   = pair_yes(vlm_ans, "door_change")
    newobj = pair_yes(vlm_ans, "new_object")
    hazard = pair_yes(vlm_ans, "hazard_damage")

    # 강한 no-change
    if change == "no" and door != "yes" and newobj != "yes" and hazard != "yes":
        return "no_change"

    # hazard 최우선
    if hazard == "yes":
        return "hazard"

    # door
    if door == "yes":
        return "door"

    # object change
    if newobj == "yes" or change == "yes":
        return "object_change"

    # description 보조
    desc = str(vlm_ans.get("change_description", "")).lower().strip()
    if desc and desc not in ["unknown", "n/a", "na", "none", "no change", "same"]:
        if any(k in desc for k in ["added", "removed", "moved", "missing", "new", "changed", "broken", "crack", "damage", "debris", "open", "closed"]):
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
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="auto",
        )
    

    roi_classes = []
    

    for roi_i, cand in enumerate(candidates, start=1):
        qry_crop = cand["pair_rgb"]
        vlm_ans = {}     

        print(f"\n================ ROI {roi_i} ================")
        for q in QUESTIONS:
            qtext = q["text"]

            PROMPT_TEMPLATE = (
                "This image shows a REFERENCE (left) and a CURRENT (right) view of the same area.\n"
                "Compare them carefully and answer the question in one short phrase (max 5 words).\n"
                "Ignore lighting changes, shadows, reflections, and slight viewpoint shifts.\n"
                "Focus only on physical/object changes. If no change, answer 'no change'.\n"
                "Question: {qtext}\n"
                "Answer:"
            )
            prompt = PROMPT_TEMPLATE.format(qtext=qtext)
            ans = vlm.infer_image(qry_crop, prompt)

            print(f"- {q['id']}: {qtext}")
            print(f"    ans: {ans}")

            prompt = PROMPT_TEMPLATE.format(qtext=q["text"])
            vlm_ans[q["id"]] = ans

        cls = classify_roi_4class_from_compare_questions(vlm_ans)
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

