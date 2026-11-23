import os
import sys
import cv2
import numpy as np
import onnxruntime as ort


def fast_guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 16, eps: float = 1e-4) -> np.ndarray:
    """纯 OpenCV 实现的快速导向滤波，让深度图边缘死死咬住原图物理轮廓"""
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    else:
        guide_gray = guide

    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide_gray, -1, ksize)
    mean_p = cv2.boxFilter(src, -1, ksize)
    mean_ip = cv2.boxFilter(guide_gray * src, -1, ksize)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide_gray * guide_gray, -1, ksize)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    q = mean_a * guide_gray + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


class DepthEngine:
    def __init__(self, model_path: str = os.path.join("models", "depth_anything_v2_vits.onnx")):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            model_path,
            os.path.join(base_dir, "models", "depth_anything_v2_vits.onnx"),
            os.path.join(base_dir, "assets", "depth_anything_v2_vits.onnx"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "depth_anything_v2_vits.onnx"),
        ]
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "models", "depth_anything_v2_vits.onnx"))
            candidates.append(os.path.join(sys._MEIPASS, "assets", "depth_anything_v2_vits.onnx"))

        resolved = next((p for p in candidates if os.path.exists(p)), None)
        if not resolved:
            raise FileNotFoundError(f"找不到深度模型 depth_anything_v2_vits.onnx，请放入 assets/ 或 models/ 目录")
        model_path = resolved

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            except Exception:
                pass

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_providers = self.session.get_providers()

        # ImageNet 标准化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def estimate_depth(self, bg_bgr_u8: np.ndarray, refine_radius: int = 16) -> np.ndarray:
        """
        推理相对深度图: 返回 HxW float32 [0.0, 1.0]
        约定: 1.0 = 离镜头最近 (前景), 0.0 = 离镜头最远 (远景/天空)
        """
        h, w = bg_bgr_u8.shape[:2]
        rgb = cv2.cvtColor(bg_bgr_u8, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        normed = (resized - self.mean) / self.std
        tensor = np.transpose(normed, (2, 0, 1))[None, ...]  # [1, 3, 518, 518]

        raw_out = self.session.run([self.output_name], {self.input_name: tensor})[0][0, 0]
        d_min, d_max = float(raw_out.min()), float(raw_out.max())
        depth_norm = (raw_out - d_min) / (d_max - d_min + 1e-8)

        depth_full = cv2.resize(depth_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        depth_full = np.clip(depth_full, 0.0, 1.0)

        guide = bg_bgr_u8.astype(np.float32) / 255.0
        r = max(4, int(refine_radius * max(h, w) / 1600.0))
        depth_refined = fast_guided_filter(guide, depth_full, radius=r, eps=2e-4)
        return depth_refined

    @staticmethod
    def sample_character_depth(depth_map: np.ndarray, char_alpha: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """根据角色脚底接触区域采样 Z_char 深度锚点"""
        x1, y1, x2, y2 = bbox
        h_bg, w_bg = depth_map.shape
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(w_bg, x2), min(h_bg, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.5

        sub_alpha = char_alpha[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1]
        ys, xs = np.where(sub_alpha > 0.35)
        if len(ys) == 0:
            return float(np.median(depth_map[cy1:cy2, cx1:cx2]))

        # 取角色最底部 12% 像素作为脚底接触面
        y_thresh = np.percentile(ys, 88)
        foot_mask = ys >= y_thresh
        foot_ys = ys[foot_mask] + cy1
        foot_xs = xs[foot_mask] + cx1
        return float(np.median(depth_map[foot_ys, foot_xs]))

    @staticmethod
    def compute_occlusion_mask(
        depth_map: np.ndarray, z_char: float, bias: float = 0.025, feather: float = 0.03
    ) -> np.ndarray:
        """生成前景遮挡掩膜 (值越接近 1 代表该像素在角色前方，将遮挡角色)"""
        diff = depth_map - (z_char + bias)
        occ = np.clip(diff / max(feather, 1e-4), 0.0, 1.0).astype(np.float32)
        return cv2.GaussianBlur(occ, (3, 3), 0.6)