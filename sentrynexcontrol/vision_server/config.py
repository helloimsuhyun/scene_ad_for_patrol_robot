import json, copy
from pathlib import Path

DEFAULT_CFG = {
    "embed": {
        "model_name": "dinov2_vits14",
        "img_size": 560,
        "global_mode": "cls"
    },

    "repr": {
        "repr_mode": "global_patch_with_aligned"
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

    "repr_modes": {
        "global_patch_with_aligned": {
            "global_preselect": {
                "mode": "vpr",
                "top_m": 3
            },
            "cc": {
                "radius": 1,
                "top_p": 0.05,
                "alpha": 0.6,
                "min_cut": 0.2,
                "singleton_weight": 0.25,
                "component_min_area": 2
            },
            "proposal": {
                "top_k": 5,
                "patch_margin": 1,
                "crop_margin_ratio": 0.20,
                "min_patch_area": 2,
                "min_crop_size": 96
            },
            "verifier": {
                "radius": 1,
                "top_p": 0.10
            },
            "calibration": {
                "cc_k": 2.5,
                "final_k": 2.5,
                "final_threshold_floor": 0.49,
                "max_imgs": 30
            }
        }
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
        "event_rule": "vote",
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

    # 하위호환: patchcore 값들을 repr_modes 쪽 기본값으로 복사
    gpa = out.setdefault("repr_modes", {}).setdefault("global_patch_with_aligned", {})
    gpa.setdefault("global_preselect", {}).setdefault(
        "top_m", out.get("patchcore", {}).get("preselect_m", 3)
    )

    cc = gpa.setdefault("cc", {})
    cc.setdefault("radius", out.get("patchcore", {}).get("radius", 1))
    cc.setdefault("top_p", out.get("patchcore", {}).get("top_p", 0.05))
    cc.setdefault("alpha", out.get("patchcore", {}).get("alpha", 0.6))
    cc.setdefault("min_cut", out.get("patchcore", {}).get("min_cut", 0.2))
    cc.setdefault("singleton_weight", out.get("patchcore", {}).get("singleton_weight", 0.25))
    cc.setdefault("component_min_area", out.get("patchcore", {}).get("component_min_area", 2))

    proposal = gpa.setdefault("proposal", {})
    proposal.setdefault("top_k", 5)
    proposal.setdefault("patch_margin", 1)
    proposal.setdefault("crop_margin_ratio", 0.20)
    proposal.setdefault("min_patch_area", 2)
    proposal.setdefault("min_crop_size", 96)

    verifier = gpa.setdefault("verifier", {})
    verifier.setdefault("radius", 1)
    verifier.setdefault("top_p", 0.10)

    calib = gpa.setdefault("calibration", {})
    calib.setdefault("cc_k", out.get("calib", {}).get("robust_k", 2.5))
    calib.setdefault("final_k", out.get("calib", {}).get("robust_k", 2.5))
    calib.setdefault("final_threshold_floor", 0.49)
    calib.setdefault("max_imgs", 30)

    return out