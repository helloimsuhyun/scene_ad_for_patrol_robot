import json, copy
from pathlib import Path

DEFAULT_CFG = {
    "embed": {
        "model_name": "dinov2_vits14",
        "img_size": 560,
        "global_mode": "patch_mean"
    },

    "repr": {
        "repr_mode": "global_patch_with_aligned"
    },

    "threshold": {
        "method": "percentile",
        "percentile_value": 97,
        "robust_std_k": 2.5,
        "gaussian_std_k": 2.5,
        "topk_neighbors": 3
    },

    "infer": {
        "event_rule": "vote",
        "use_two_stage_vlm": False
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
        "valid_patch_thr": 1.0,
        "mask_erode_kernel": 3,
        "mask_erode_iter": 1
    },

    "modes": {
        "global_patch": {
            "preselect": {
                "mode": "vpr",
                "top_m": 3
            },
            "patch_score": {
                "top_p": 0.05,
                "alpha": 0.6
            },
            "calibration": {
                "threshold_param": 97,
                "threshold_floor": 0.0,
                "max_calib_images": 100,
                "min_th_calib_images": 5
            }
        },

        "global_patch_with_aligned": {
            "preselect": {
                "mode": "vpr",
                "top_m": 3
            },
            "cc": {
                "radius": 1,
                "top_p": 0.05,
                "alpha": 0.6,
                "min_cut": 0.20,
                "singleton_weight": 0.25,
                "component_min_area": 2
            },
            "proposal": {
                "top_k": 3,
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
                "threshold_param": 97,
                "threshold_floor": 0.45,
                "max_calib_images": 100,
                "min_th_calib_images": 5
            }
        }
    }
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


def get_cfg_bundle(cfg: dict):
    repr_mode = str(cfg.get("repr", {}).get("repr_mode", "global"))

    mode_cfg = cfg.get("modes", {}).get(repr_mode, {})
    threshold_cfg = cfg.get("threshold", {})

    return {
        "repr_mode": repr_mode,
        "mode_cfg": mode_cfg,
        "threshold_cfg": threshold_cfg,
        "preselect": mode_cfg.get("preselect", {}),
        "calibration": mode_cfg.get("calibration", {}),
        "cc": mode_cfg.get("cc", {}),
        "proposal": mode_cfg.get("proposal", {}),
        "verifier": mode_cfg.get("verifier", {}),
    }