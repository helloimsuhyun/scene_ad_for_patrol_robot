# config.py
import json, copy
from pathlib import Path

DEFAULT_CFG = {
    "embed": {
        "model_name": "dinov2_vits14", 
        "img_size": 560, 
        "global_mode": "cls"
    },
    "repr": {
        "repr_mode": "global"
    },
    "patchcore": {
    "top_p": 0.05,
    "preselect_m": 3,
    "radius": 1,
    "alpha": 0.6,
    "min_cut": 0.2,
    "singleton_weight": 0.25,
    "component_min_area": 2
    },
    "superglue": {  
        "resize_long_side": 640,
        "weights": "indoor",
        "max_keypoints": 1024,
        "keypoint_threshold": 0.003,
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,

        "min_matches": 10,
        "min_inliers": 15,
        "min_inlier_ratio": 0.2,
        "max_reproj_error": 5.0,

        "valid_patch_thr": 0.6,
        "mask_erode_kernel": 3,
        "mask_erode_iter": 1
    },
    "calib": {
        "k": 3, 
        "method": "robust", 
        "percentile": 97, 
        "robust_k": 2.5, 
        "gaussian_k": 2.5
    },
    "infer": {
        "event_rule": "max", 
        "use_two_stage_vlm": False
    },
}


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_cfg(bank_root: Path) -> dict:
    p = Path(bank_root) / "config.json"
    user = {}
    if p.exists():
        user = json.loads(p.read_text(encoding="utf-8"))
    out = copy.deepcopy(DEFAULT_CFG)
    _deep_update(out, user)
    return out