# superpoint & superglue 기반 matching > RANSAC하여 변환행렬과 reprojection error를 반환

import cv2
import torch
import numpy as np
from dataclasses import dataclass
from .superglue.models.matching import Matching


@dataclass
class SuperGlueMatchConfig:
    resize_long_side: int = 640
    weights: str = "indoor"
    max_keypoints: int = 1024
    keypoint_threshold: float = 0.003
    match_threshold: float = 0.2
    sinkhorn_iterations: int = 20


class SuperGlueMatcher:
    def __init__(self, cfg: SuperGlueMatchConfig, device="cuda"):
        self.cfg = cfg
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        self.matching = Matching({
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": cfg.keypoint_threshold,
                "max_keypoints": cfg.max_keypoints,
            },
            "superglue": {
                "weights": cfg.weights,
                "sinkhorn_iterations": cfg.sinkhorn_iterations,
                "match_threshold": cfg.match_threshold,
            },
        }).eval().to(device)

    def _resize(self, img):
        h, w = img.shape[:2]
        scale = self.cfg.resize_long_side / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        return resized, scale

    @staticmethod
    def _grid_coverage(pts_xy, img_shape, grid_size=4, min_pts_per_cell=1):
        """
        pts_xy: (N, 2), x-y 좌표
        img_shape: (H, W) 또는 image.shape[:2]
        return: 0.0 ~ 1.0
        """
        if pts_xy is None or len(pts_xy) == 0:
            return 0.0

        h, w = img_shape[:2]
        if h <= 0 or w <= 0:
            return 0.0

        pts = np.asarray(pts_xy, dtype=np.float32)

        xs = pts[:, 0]
        ys = pts[:, 1]

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        pts = pts[valid]

        if len(pts) == 0:
            return 0.0

        xs = pts[:, 0]
        ys = pts[:, 1]

        gx = np.floor(xs / max(w, 1) * grid_size).astype(np.int32)
        gy = np.floor(ys / max(h, 1) * grid_size).astype(np.int32)

        gx = np.clip(gx, 0, grid_size - 1)
        gy = np.clip(gy, 0, grid_size - 1)

        counts = np.zeros((grid_size, grid_size), dtype=np.int32)

        for x_cell, y_cell in zip(gx, gy):
            counts[y_cell, x_cell] += 1

        occupied = counts >= min_pts_per_cell
        return float(occupied.sum() / float(grid_size * grid_size))

    def match_and_estimate(self, query_bgr, bank_bgr):
        q, scale_q = self._resize(query_bgr)
        b, scale_b = self._resize(bank_bgr)

        q_gray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        q_tensor = torch.from_numpy(q_gray / 255.0).float()[None, None].to(self.device)
        b_tensor = torch.from_numpy(b_gray / 255.0).float()[None, None].to(self.device)
        
        with torch.inference_mode():
            pred = self.matching({"image0": q_tensor, "image1": b_tensor})

        kpts_q = pred["keypoints0"][0].cpu().numpy()
        kpts_b = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()

        valid = matches > -1
        if valid.sum() < 10:
            return {"ok": False, "reason": "too_few_matches"}

        pts_q = kpts_q[valid]
        pts_b = kpts_b[matches[valid]]

        H_resized, inliers = cv2.findHomography(pts_q, pts_b, cv2.RANSAC)

        if H_resized is None or inliers is None:
            return {"ok": False, "reason": "homography_failed"}

        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(pts_q)

        if inlier_count < 15 or inlier_ratio < 0.2:
            return {"ok": False, "reason": "low_inlier_quality"}

        # resized 좌표계 -> 원본 좌표계로 복원
        S_q = np.array([
            [scale_q, 0.0, 0.0],
            [0.0, scale_q, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        S_b = np.array([
            [scale_b, 0.0, 0.0],
            [0.0, scale_b, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        H = np.linalg.inv(S_b) @ H_resized @ S_q

        # reprojection error
        pts_q_orig = pts_q[inlier_mask] / scale_q
        pts_b_orig = pts_b[inlier_mask] / scale_b

        query_grid_coverage = self._grid_coverage(
            pts_q_orig,
            query_bgr.shape[:2],
            grid_size=4,
            min_pts_per_cell=1,
        )

        bank_grid_coverage = self._grid_coverage(
            pts_b_orig,
            bank_bgr.shape[:2],
            grid_size=4,
            min_pts_per_cell=1,
        )

        inlier_grid_coverage = min(query_grid_coverage, bank_grid_coverage)

        pts_q_orig_h = pts_q_orig.reshape(-1, 1, 2).astype(np.float32)
        pts_b_orig_h = pts_b_orig.reshape(-1, 1, 2).astype(np.float32)

        pts_q_proj = cv2.perspectiveTransform(pts_q_orig_h, H)

        error = np.linalg.norm(pts_q_proj - pts_b_orig_h, axis=2)
        mean_error = float(error.mean())
        median_error = float(np.median(error))

        if mean_error > 5.0:
            return {"ok": False, "reason": "high_reprojection_error"}

        return {
            "ok": True,
            "H": H,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "reproj_error_mean": mean_error,
            "reproj_error_median": median_error,
            "num_matches": int(valid.sum()),

            # 전체 정합 품질 판단용
            "query_inlier_grid_coverage": float(query_grid_coverage),
            "bank_inlier_grid_coverage": float(bank_grid_coverage),
            "inlier_grid_coverage": float(inlier_grid_coverage),
        }