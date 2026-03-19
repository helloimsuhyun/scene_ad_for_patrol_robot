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
        "top_p": 0.1,
        "preselect_m": 10
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