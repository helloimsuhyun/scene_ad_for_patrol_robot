import os
import base64
import io
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
from PIL import Image


def vlm_gate(
    q_img_bgr: np.ndarray,
    ref_img_path,
    model: str = "gpt-4.1-mini",
    timeout_s: int = 30,
    max_tokens: int = 200,
) -> Dict[str, Any]:
    return {}