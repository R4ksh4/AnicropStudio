from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable
import cv2
import numpy as np

from .depth_engine import DepthEngine
from .optics_engine import (
    apply_depth_bokeh,
    apply_lens_flare_fx,
    apply_spotlight_fx,
    apply_tyndall_rays,
    match_character_optics,
)
from .lighting_engine import estimate_light_direction_and_color, azimuth_to_vec3, apply_lighting_and_rim
from .color_shadow_engine import match_color_and_tone, render_shadows, render_real_contact_shadows


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img

import struct


def _extract_exif_fnumber(src_path: str) -> float | None:
    """从底图 EXIF APP1(0x829D FNumber) 直接解析原片光圈值"""
    segs = _extract_metadata_segments(src_path)
    for seg in segs:
        if len(seg) < 12 or seg[0:2] != b"\xff\xe1" or seg[4:10] != b"Exif\x00\x00":
            continue
        tiff = seg[10:]
        if len(tiff) < 8:
            continue
        endian = "<" if tiff[0:2] == b"II" else (">" if tiff[0:2] == b"MM" else "")
        if not endian:
            continue
        ifd0_off = struct.unpack(endian + "I", tiff[4:8])[0]
        if ifd0_off + 2 > len(tiff):
            continue
        n0 = struct.unpack(endian + "H", tiff[ifd0_off : ifd0_off + 2])[0]
        exif_ifd_off = None
        for i in range(n0):
            e = ifd0_off + 2 + i * 12
            if e + 12 > len(tiff):
                break
            tag = struct.unpack(endian + "H", tiff[e : e + 2])[0]
            if tag == 0x8769:  # ExifIFDPointer
                exif_ifd_off = struct.unpack(endian + "I", tiff[e + 8 : e + 12])[0]
                break
        if exif_ifd_off is None or exif_ifd_off + 2 > len(tiff):
            continue
        n1 = struct.unpack(endian + "H", tiff[exif_ifd_off : exif_ifd_off + 2])[0]
        for i in range(n1):
            e = exif_ifd_off + 2 + i * 12
            if e + 12 > len(tiff):
                break
            tag, typ, cnt, val_off = struct.unpack(endian + "HHII", tiff[e : e + 12])
            if tag == 0x829D and typ == 5 and cnt >= 1:  # FNumber (RATIONAL)
                if val_off + 8 <= len(tiff):
                    num, den = struct.unpack(endian + "II", tiff[val_off : val_off + 8])
                    if den > 0:
                        f_val = num / float(den)
                        if 0.3 <= f_val <= 64.0:
                            return round(f_val, 2)
    return None

def _extract_metadata_segments(src_path: str) -> list[bytes]:
    segments: list[bytes] = []
    if not src_path or not os.path.exists(src_path):
        return segments
    try:
        with open(src_path, "rb") as f:
            raw_data = f.read(2 * 1024 * 1024)
    except Exception:
        return segments

    if not raw_data or len(raw_data) < 4 or raw_data[:2] != b"\xff\xd8":
        return segments

    idx = 2
    n = len(raw_data)
    while idx < n - 4:
        if raw_data[idx] != 0xFF:
            break
        marker = raw_data[idx + 1]
        if marker in (0xD9, 0xDA, 0xC0, 0xC1, 0xC2, 0xC4, 0xDB):
            break
        length = int.from_bytes(raw_data[idx + 2 : idx + 4], "big")
        seg_end = idx + 2 + length
        if seg_end > n:
            break
        if 0xE1 <= marker <= 0xEF or marker == 0xFE:
            segments.append(raw_data[idx:seg_end])
        idx = seg_end
    return segments


def imwrite_unicode(path: str, img_u8: np.ndarray, quality: int = 96, orig_file_path: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = Path(path).suffix.lower() or ".png"
    if ext in (".jpg", ".jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif ext == ".png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        encode_params = []

    ok, buf = cv2.imencode(ext, img_u8, encode_params)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")

    data_bytes = buf.tobytes()
    if ext in (".jpg", ".jpeg") and orig_file_path:
        meta_segs = _extract_metadata_segments(orig_file_path)
        if meta_segs and len(data_bytes) >= 2 and data_bytes[:2] == b"\xff\xd8":
            insert_pos = 2
            if len(data_bytes) > 4 and data_bytes[2:4] == b"\xff\xe0":
                app0_len = int.from_bytes(data_bytes[4:6], "big")
                insert_pos = 2 + 2 + app0_len
            data_bytes = data_bytes[:insert_pos] + b"".join(meta_segs) + data_bytes[insert_pos:]

    with open(path, "wb") as f:
        f.write(data_bytes)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    c = np.clip(img, 0.0, 1.0)
    return (c * c).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(img, 0.0, 1.0)).astype(np.float32)


def rgb255_to_bgr_linear(rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = rgb
    bgr = np.array([b, g, r], dtype=np.float32) / 255.0
    return (bgr * bgr).astype(np.float32)


@dataclass
class MergeParams:
    pos_x: float = 0.50
    pos_y: float = 0.69
    scale: float = 0.62

    light_dir_mode: str = "自动识别"  # 自动识别 | 自定义 | 绑定光效
    azimuth_deg: float = -60.0
    elevation_deg: float = 35.0
    auto_light_color: bool = True
    custom_light_color_rgb: tuple[int, int, int] = (255, 195, 110)

    bg_exposure: float = 0.0
    char_exposure: float = 0.0

    # --- 3 种独立光效开关与各自专属参数 ---
    light_fx_enabled: bool = False
    light_fx_type: str = "丁达尔光"       # 丁达尔光 | 聚光灯 | 光晕
    light_fx_layer: str = "置于人物前方"  # 置于人物前方 | 置于人物后方 | 人后体积逆光
    fx_edge_spill: float = 0.45          # 光束漫过人体强度 (上下关系+迎光面穿透)
    fx_occlusion_strength: float = 0.72  # 光束被人体截断强度 (下游体积暗影)
    fx_pos_x: float = 0.0
    fx_pos_y: float = 0.0
    fx_aim_deg: float = 40.0

    # 1) 丁达尔光
    tyndall_strength: float = 0.55
    tyndall_count: float = 1.0
    tyndall_softness: float = 0.50
    tyndall_length: float = 0.70

    # 2) 聚光灯
    spot_strength: float = 0.70
    spot_width: float = 0.45
    spot_softness: float = 0.50
    spot_dust: float = 0.50
    spot_tip_hide: float = 0.45

    face_protect_enabled: bool = True    # 🛡️ 人脸保护开关

    # 3) 光晕与太阳光斑
    halo_strength: float = 0.65
    halo_radius: float = 1.10
    halo_core_strength: float = 0.35
    halo_arc_hide: float = 0.50
    halo_width: float = 0.45
    halo_inner_rgb: tuple[int, int, int] = (255, 200, 110)
    halo_outer_rgb: tuple[int, int, int] = (255, 110, 185)
    flare_strength: float = 0.60
    flare_size: float = 0.75
    flare_shape: str = "六边形光圈"  # 圆形散景 | 六边形光圈 | 八边形光圈 | 星芒十字
    flare_color_rgb: tuple[int, int, int] = (255, 215, 145)
    flare_angle: float = 40.0

    sensor_type: str = "半画幅 (APS-C)"  # 全画幅 | 半画幅 (APS-C) | 手机
    aperture_strength: float = 0.30     # 0.0=原片光圈 ~ 1.0=f0.33
    edge_feather: float = 1.5
    rim_strength: float = 0.90
    light_wrap_strength: float = 0.55
    shading_25d_strength: float = 0.45

    brightness_match: float = 0.70
    color_temp_match: float = 0.45
    shadow_lift: float = 0.45
    shadow_opacity: float = 0.65
    shadow_length: float = 0.85

    blur_match_strength: float = 0.20
    noise_match_strength: float = 0.20

    occlusion_enabled: bool = False
    occlusion_bias: float = 0.025

    # --- 场照模式 / 真人物理边缘与画笔扩展 ---
    work_mode: str = "anime"          # anime | cosplay
    edge_mode: str = "anime"          # anime | real
    shadow_mode: str = "anime"        # anime | real
    edge_choke: float = 0.0           # -3.0 ~ +3.0 px
    rim_texture_weight: float = 0.65  # 0.0 ~ 1.0
    rim_falloff: float = 0.50         # 0.0 ~ 1.0
    hair_transmission: float = 0.55   # 0.0 ~ 1.0
    normal_detail_mix: float = 0.55   # 0.0 ~ 1.0
    dof_tolerance: float = 0.06       # 0.0 ~ 0.35 焦平面清晰带厚度
    is_same_bg_aligned: bool = False  # 自动识别同图1:1重合


class PhotoMergePipeline:
    def __init__(self, model_path: str = os.path.join("models", "depth_anything_v2_vits.onnx")):
        self.depth_engine = DepthEngine(model_path)

        self.bg_path: str = ""
        self.bg_exif_fnumber: float = 4.0
        self.bg_has_exif_f: bool = False
        self.bg_full_u8: np.ndarray | None = None
        self.bg_proxy_u8: np.ndarray | None = None
        self.bg_fast_u8: np.ndarray | None = None

        self.depth_proxy: np.ndarray | None = None
        self.depth_fast: np.ndarray | None = None
        self.depth_full: np.ndarray | None = None

        self.char_full_bgra: np.ndarray | None = None
        self.char_fade_mask: np.ndarray | None = None
        self.char_rim_erase_mask: np.ndarray | None = None
        self.bg_sharp_mask: np.ndarray | None = None
        self.auto_azimuth_deg: float = -60.0
        self.auto_light_color_rgb: tuple[int, int, int] = (255, 235, 200)

        self._bokeh_cache: dict[str, tuple[tuple, np.ndarray]] = {}

    def load_background(self, path: str, proxy_long_edge: int = 1600, fast_long_edge: int = 720) -> None:
        img = imread_unicode(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.bg_path = path
        f_exif = _extract_exif_fnumber(path)
        self.bg_has_exif_f = f_exif is not None
        self.bg_exif_fnumber = f_exif if f_exif is not None else 4.0
        self.bg_full_u8 = img
        h, w = img.shape[:2]
        long_edge = max(h, w)

        if long_edge > proxy_long_edge:
            r = proxy_long_edge / float(long_edge)
            self.bg_proxy_u8 = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)
        else:
            self.bg_proxy_u8 = img.copy()

        if long_edge > fast_long_edge:
            r_f = fast_long_edge / float(long_edge)
            self.bg_fast_u8 = cv2.resize(img, (int(round(w * r_f)), int(round(h * r_f))), interpolation=cv2.INTER_AREA)
        else:
            self.bg_fast_u8 = img.copy()

        self.depth_proxy = self.depth_engine.estimate_depth(self.bg_proxy_u8)
        fh, fw = self.bg_fast_u8.shape[:2]
        self.depth_fast = cv2.resize(self.depth_proxy, (fw, fh), interpolation=cv2.INTER_LINEAR)
        self.depth_full = None
        self._bokeh_cache.clear()

        bg_scale = 512.0 / max(h, w)
        self.bg_sharp_mask = np.zeros(
            (max(32, int(round(h * bg_scale))), max(32, int(round(w * bg_scale)))),
            dtype=np.float32,
        )

    def load_character(self, path: str) -> None:
        img = imread_unicode(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        self.char_full_bgra = img
        self.reset_fade_mask()

    def auto_extract_character_from_bg(self) -> None:
        """场照单图直通：复用 MODNet 底层模型全图抠人（绕开旧版左边 22% 截断逻辑）+ 导向滤波锁边"""
        if self.bg_full_u8 is None:
            raise RuntimeError("请先加载底图")
        from portrait_matting import PortraitMatting
        from .depth_engine import fast_guided_filter

        session = PortraitMatting._get_session()
        in_tensor = session.get_inputs()[0]
        in_shape = in_tensor.shape
        req_h = in_shape[2] if (len(in_shape) > 2 and isinstance(in_shape[2], int) and in_shape[2] > 0) else 512
        req_w = in_shape[3] if (len(in_shape) > 3 and isinstance(in_shape[3], int) and in_shape[3] > 0) else 512

        h, w = self.bg_full_u8.shape[:2]
        resized = cv2.resize(self.bg_full_u8, (req_w, req_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None, :, :, :]

        pred = session.run([session.get_outputs()[0].name], {in_tensor.name: blob})[0]
        matte_small = np.clip(np.squeeze(pred), 0.0, 1.0).astype(np.float32)
        matte_full = cv2.resize(matte_small, (w, h), interpolation=cv2.INTER_LINEAR)

        guide = self.bg_full_u8.astype(np.float32) / 255.0
        matte_refined = fast_guided_filter(guide, matte_full, radius=max(3, int(round(max(h, w) / 400.0))), eps=1e-3)
        alpha_u8 = np.clip(matte_refined * 255.0, 0, 255).astype(np.uint8)

        b, g, r = cv2.split(self.bg_full_u8)
        self.char_full_bgra = cv2.merge([b, g, r, alpha_u8])
        self.reset_fade_mask()

    def detect_same_bg_alignment(self) -> tuple[bool, float, float, float]:
        """
        检测前景图是否从当前底图原位抠出：
        - 命中同一张原图：返回 (True, 0.5, 0.5, 1.0) 实现 1:1 绝对重合；
        - 未命中（换背景 / 没识别到同一人）：返回 (False, 0.5, 0.58, 0.68) 默认放到中间。
        """
        if self.bg_proxy_u8 is None or self.char_full_bgra is None:
            return False, 0.50, 0.62, 0.62

        bh, bw = self.bg_proxy_u8.shape[:2]
        ch, cw = self.char_full_bgra.shape[:2]
        ar_bg = bw / float(max(1, bh))
        ar_ch = cw / float(max(1, ch))
        if abs(ar_bg - ar_ch) > 0.04:
            return False, 0.50, 0.58, 0.68

        sw, sh = 256, max(32, int(round(256.0 / max(0.1, ar_bg))))
        bg_sm = cv2.resize(self.bg_proxy_u8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ch_sm = cv2.resize(self.char_full_bgra, (sw, sh), interpolation=cv2.INTER_AREA)
        ch_rgb = ch_sm[:, :, :3].astype(np.float32) / 255.0
        ch_a = ch_sm[:, :, 3].astype(np.float32) / 255.0

        fg = ch_a > 0.75
        if np.sum(fg) < 80:
            return False, 0.50, 0.58, 0.68

        diff = float(np.median(np.abs(bg_sm[fg] - ch_rgb[fg])))
        if diff < 0.085:
            return True, 0.50, 0.50, 1.00
        return False, 0.50, 0.58, 0.68

    def _get_char_face_norm(self) -> tuple[float, float, float, float] | None:
        """定位角色面部归一化矩形 (fx, fy, fw, fh)"""
        if self.char_full_bgra is None:
            return None
        if hasattr(self, "_cached_char_face_norm"):
            return self._cached_char_face_norm

        ch_h, ch_w = self.char_full_bgra.shape[:2]
        bgr = self.char_full_bgra[:, :, :3]
        try:
            from focus_inspector import FocusInspector
            scale = 1280.0 / max(ch_w, ch_h)
            gw, gh = int(ch_w * scale), int(ch_h * scale)
            pad_w, pad_h = (gw + 31) // 32 * 32, (gh + 31) // 32 * 32
            pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            pad_img[:gh, :gw] = cv2.resize(bgr, (gw, gh))
            faces = FocusInspector._detect_faces_on_chip(pad_img)
            if faces:
                faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
                bx, by, bw, bh = faces[0]["box"]
                norm_box = (
                    float(bx / scale / ch_w),
                    float(by / scale / ch_h),
                    float(bw / scale / ch_w),
                    float(bh / scale / ch_h),
                )
                self._cached_char_face_norm = norm_box
                return norm_box
        except Exception:
            pass

        try:
            ys, xs = np.where(self.char_full_bgra[:, :, 3] > 100)
            if len(ys) > 100:
                y_min, y_max = float(np.min(ys)), float(np.max(ys))
                x_min, x_max = float(np.min(xs)), float(np.max(xs))
                bw = (x_max - x_min) * 0.40
                bh = (y_max - y_min) * 0.22
                bx = (x_min + x_max) / 2.0 - bw / 2.0
                by = y_min + (y_max - y_min) * 0.05
                norm_box = (bx / ch_w, by / ch_h, bw / ch_w, bh / ch_h)
                self._cached_char_face_norm = norm_box
                return norm_box
        except Exception:
            pass

        self._cached_char_face_norm = None
        return None

    def reset_fade_mask(self) -> None:
        if hasattr(self, "_cached_char_face_norm"):
            del self._cached_char_face_norm
        if self.char_full_bgra is not None:
            h, w = self.char_full_bgra.shape[:2]
            scale = 512.0 / max(h, w)
            mw, mh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
            self.char_fade_mask = np.zeros((mh, mw), dtype=np.float32)
            self.char_rim_erase_mask = np.zeros((mh, mw), dtype=np.float32)
        else:
            self.char_fade_mask = None
            self.char_rim_erase_mask = None

        if self.bg_full_u8 is not None:
            bh, bw = self.bg_full_u8.shape[:2]
            b_sc = 512.0 / max(bh, bw)
            self.bg_sharp_mask = np.zeros(
                (max(32, int(round(bh * b_sc))), max(32, int(round(bw * b_sc)))),
                dtype=np.float32,
            )

    def paint_fade_mask(
        self,
        u: float,
        v: float,
        radius_norm: float,
        erase: bool = False,
        hardness: float = 0.50,
        opacity: float = 0.65,
        brush_mode: str = "blur",
        u_bg: float | None = None,
        v_bg: float | None = None,
    ) -> None:
        if brush_mode == "sharp":
            target = self.bg_sharp_mask
            pu = u if u_bg is None else u_bg
            pv = v if v_bg is None else v_bg
        elif brush_mode == "rim_erase":
            target = self.char_rim_erase_mask
            pu, pv = u, v
        else:
            target = self.char_fade_mask
            pu, pv = u, v

        if target is None:
            return

        mh, mw = target.shape
        cx, cy = int(round(pu * mw)), int(round(pv * mh))
        r = max(2, int(round(radius_norm * max(mw, mh))))

        y1, y2 = max(0, cy - r), min(mh, cy + r + 1)
        x1, x2 = max(0, cx - r), min(mw, cx + r + 1)
        if y2 <= y1 or x2 <= x1:
            return

        yy, xx = np.ogrid[y1:y2, x1:x2]
        norm_dist = np.clip(np.hypot(xx - cx, yy - cy) / float(r), 0.0, 1.0)

        # 硬度控制：0%=平滑余弦渐变软边；100%=锐利硬圆盘
        h_clamped = float(np.clip(hardness, 0.0, 0.99))
        t = np.clip((norm_dist - h_clamped * 0.95) / max(1e-4, 1.0 - h_clamped * 0.95), 0.0, 1.0)
        brush = (0.5 + 0.5 * np.cos(t * np.pi)) ** 1.4
        op = float(np.clip(opacity, 0.05, 1.0))

        if erase:
            target[y1:y2, x1:x2] = np.clip(target[y1:y2, x1:x2] - brush * op, 0.0, 1.0)
        else:
            target[y1:y2, x1:x2] = np.clip(target[y1:y2, x1:x2] + brush * op * 0.65, 0.0, 1.0)

    def get_target_fnumber(self, t: float) -> float:
        f_orig = max(0.5, self.bg_exif_fnumber)
        return f_orig * ((0.33 / f_orig) ** float(np.clip(t, 0.0, 1.0)))

    def compute_effective_bokeh(self, params: MergeParams) -> float:
        if params.aperture_strength <= 0.01:
            return 0.0
        f_orig = max(0.5, self.bg_exif_fnumber)
        f_target = self.get_target_fnumber(params.aperture_strength)
        sensor_scale = {"全画幅": 1.0, "半画幅 (APS-C)": 0.75, "手机": 0.45}.get(params.sensor_type, 0.75)
        return max(0.0, (1.0 / f_target - 1.0 / f_orig) * sensor_scale * 0.72)

    def render_raw_overlay(self, params: MergeParams) -> np.ndarray:
        if self.bg_proxy_u8 is None or self.char_full_bgra is None:
            raise RuntimeError("请先加载背景图和角色图")

        bg = self.bg_proxy_u8.copy()
        h_bg, w_bg = bg.shape[:2]
        ch_orig_h, ch_orig_w = self.char_full_bgra.shape[:2]
        target_h = max(16, int(round(h_bg * params.scale)))
        target_w = max(16, int(round(ch_orig_w * (target_h / float(ch_orig_h)))))

        char_resized = cv2.resize(self.char_full_bgra, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cx, cy = int(round(params.pos_x * w_bg)), int(round(params.pos_y * h_bg))
        x1, y1 = cx - target_w // 2, cy - target_h // 2
        x2, y2 = x1 + target_w, y1 + target_h

        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w_bg, x2), min(h_bg, y2)
        if cx2 > cx1 and cy2 > cy1:
            sy1, sy2, sx1, sx2 = cy1 - y1, cy2 - y1, cx1 - x1, cx2 - x1
            alpha = (char_resized[sy1:sy2, sx1:sx2, 3:4].astype(np.float32)) / 255.0
            fg = char_resized[sy1:sy2, sx1:sx2, :3].astype(np.float32)
            bg_sub = bg[cy1:cy2, cx1:cx2].astype(np.float32)
            bg[cy1:cy2, cx1:cx2] = np.clip(bg_sub * (1.0 - alpha) + fg * alpha, 0, 255).astype(np.uint8)
        return bg

    def render(
        self,
        params: MergeParams,
        quality: str = "proxy",
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> np.ndarray:
        if self.bg_proxy_u8 is None or self.char_full_bgra is None:
            raise RuntimeError("请先加载背景图和角色图")

        def _report(pct: int, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(pct, msg)

        _report(8, "正在准备图像矩阵与深度映射...")
        if quality == "fast":
            bg_u8 = self.bg_fast_u8
            depth_map = self.depth_fast
            interp_flag = cv2.INTER_LINEAR
        elif quality == "proxy":
            bg_u8 = self.bg_proxy_u8
            depth_map = self.depth_proxy
            interp_flag = cv2.INTER_LANCZOS4
        else:
            bg_u8 = self.bg_full_u8
            if self.depth_full is None:
                h_f, w_f = bg_u8.shape[:2]
                self.depth_full = cv2.resize(self.depth_proxy, (w_f, h_f), interpolation=cv2.INTER_CUBIC)
            depth_map = self.depth_full
            interp_flag = cv2.INTER_LANCZOS4

        h_bg, w_bg = bg_u8.shape[:2]
        bg_srgb = bg_u8.astype(np.float32) / 255.0

        ch_orig_h, ch_orig_w = self.char_full_bgra.shape[:2]
        target_h = max(16, int(round(h_bg * params.scale)))
        target_w = max(16, int(round(ch_orig_w * (target_h / float(ch_orig_h)))))

        char_resized = cv2.resize(self.char_full_bgra, (target_w, target_h), interpolation=interp_flag)
        char_srgb = char_resized[:, :, :3].astype(np.float32) / 255.0
        char_alpha = char_resized[:, :, 3].astype(np.float32) / 255.0

        # 同图 1:1 自溶不擦掉触边肢体；仅异图拼贴时对触边硬截面做淡出
        if not params.is_same_bg_aligned:
            border_fade = max(12, int(round(min(target_w, target_h) * 0.045)))
            if border_fade < target_w // 2 and border_fade < target_h // 2:
                if np.any(char_alpha[:, 0] > 0.05):
                    char_alpha[:, :border_fade] *= np.linspace(0.0, 1.0, border_fade, dtype=np.float32)[None, :]
                if np.any(char_alpha[:, -1] > 0.05):
                    char_alpha[:, -border_fade:] *= np.linspace(1.0, 0.0, border_fade, dtype=np.float32)[None, :]
                if np.any(char_alpha[0, :] > 0.05):
                    char_alpha[:border_fade, :] *= np.linspace(0.0, 1.0, border_fade, dtype=np.float32)[:, None]

        # 边缘微收/外扩 (Edge Choke: 专治抠图残留白边/黑边)
        if abs(params.edge_choke) > 0.1:
            choke_px = int(round(abs(params.edge_choke) * (h_bg / 1600.0)))
            if choke_px >= 1:
                kc = choke_px * 2 + 1
                ker_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
                if params.edge_choke < 0:
                    char_alpha = cv2.erode(char_alpha, ker_c, iterations=1)
                else:
                    char_alpha = cv2.dilate(char_alpha, ker_c, iterations=1)

        if params.edge_feather > 0.05:
            f_px = params.edge_feather * (h_bg / 1600.0)
            if f_px > 0.3:
                k = max(3, int(round(f_px)) * 2 + 1)
                pad = k // 2
                a_pad = cv2.copyMakeBorder(char_alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
                a_blur = cv2.GaussianBlur(a_pad, (k, k), sigmaX=max(0.4, f_px * 0.5))[pad:-pad, pad:-pad]
                char_alpha = np.minimum(char_alpha, a_blur)

        fade_snap = self.char_fade_mask.copy() if self.char_fade_mask is not None else None
        if fade_snap is not None and np.any(fade_snap > 0.005):
            fade_full = cv2.resize(fade_snap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            char_alpha = np.clip(char_alpha * (1.0 - fade_full), 0.0, 1.0)
            char_blur = cv2.GaussianBlur(char_srgb, (0, 0), sigmaX=max(1.5, target_w * 0.006))
            fw = np.clip(fade_full * 1.2, 0.0, 1.0)[:, :, None]
            char_srgb = char_srgb * (1.0 - fw) + char_blur * fw

        rim_erase_map = None
        if self.char_rim_erase_mask is not None and np.any(self.char_rim_erase_mask > 0.005):
            rim_erase_map = cv2.resize(self.char_rim_erase_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        cx, cy = int(round(params.pos_x * w_bg)), int(round(params.pos_y * h_bg))
        x1, y1 = cx - target_w // 2, cy - target_h // 2
        x2, y2 = x1 + target_w, y1 + target_h
        bbox = (x1, y1, x2, y2)

        z_char = self.depth_engine.sample_character_depth(depth_map, char_alpha, bbox)

        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w_bg, x2), min(h_bg, y2)
        fg_excl = None
        if params.is_same_bg_aligned and cx2 > cx1 and cy2 > cy1:
            fg_excl = np.zeros((h_bg, w_bg), dtype=np.float32)
            fg_excl[cy1:cy2, cx1:cx2] = (char_alpha[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1] > 0.25).astype(np.float32)

        _report(28, "正在计算背景曝光与光学散景...")
        eff_bokeh = self.compute_effective_bokeh(params)
        bokeh_key = (
            round(eff_bokeh, 3),
            round(params.bg_exposure, 3),
            round(params.dof_tolerance, 3),
            bool(params.is_same_bg_aligned),
        )
        cached = self._bokeh_cache.get(quality)
        if quality != "full" and cached is not None and cached[0] == bokeh_key:
            bg_bokeh_linear = cached[1]
        else:
            bg_linear = srgb_to_linear(bg_srgb)
            if abs(params.bg_exposure) > 0.01:
                bg_linear = np.clip(bg_linear * (2.0 ** params.bg_exposure), 0.0, 3.0).astype(np.float32)
            bg_bokeh_linear = apply_depth_bokeh(
                bg_linear,
                depth_map,
                z_focus=z_char,
                aperture_strength=eff_bokeh,
                highlight_boost=min(1.5, eff_bokeh * 0.8),
                dof_tolerance=params.dof_tolerance,
                fg_exclusion_mask=fg_excl,
            )
            if quality != "full":
                self._bokeh_cache[quality] = (bokeh_key, bg_bokeh_linear)

        auto_az, auto_color_linear = estimate_light_direction_and_color(bg_srgb, (float(cx), float(cy)))
        self.auto_azimuth_deg = auto_az
        srgb_bgr = np.sqrt(np.clip(auto_color_linear, 0.0, 1.0)) * 255.0
        self.auto_light_color_rgb = (int(round(srgb_bgr[2])), int(round(srgb_bgr[1])), int(round(srgb_bgr[0])))

        if params.light_dir_mode == "自动识别":
            az_deg = auto_az
        elif params.light_dir_mode == "绑定光效":
            dx_l = (params.fx_pos_x - params.pos_x) * w_bg
            dy_l = (params.fx_pos_y - params.pos_y) * h_bg
            az_deg = -135.0 if (abs(dx_l) < 1e-3 and abs(dy_l) < 1e-3) else float(np.degrees(np.arctan2(dy_l, dx_l)))
            self.auto_azimuth_deg = az_deg
        else:
            az_deg = params.azimuth_deg
        light_vec3 = azimuth_to_vec3(az_deg, params.elevation_deg)
        light_color_linear = auto_color_linear if params.auto_light_color else rgb255_to_bgr_linear(params.custom_light_color_rgb)

        bg_bokeh_srgb = linear_to_srgb(bg_bokeh_linear)

        if cx2 > cx1 and cy2 > cy1:
            bg_roi_orig_srgb = bg_srgb[cy1:cy2, cx1:cx2]
            bg_roi_bokeh_srgb = bg_bokeh_srgb[cy1:cy2, cx1:cx2]
            bg_patch_linear = np.zeros_like(char_srgb)
            bg_patch_linear[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1] = bg_bokeh_linear[cy1:cy2, cx1:cx2]
        else:
            bg_roi_orig_srgb = bg_srgb
            bg_roi_bokeh_srgb = bg_bokeh_srgb
            bg_patch_linear = np.zeros_like(char_srgb)

        _report(48, "正在对齐局部环境影调与光学质感...")
        char_toned_srgb = match_color_and_tone(
            char_srgb,
            char_alpha,
            bg_roi_bokeh_srgb,
            brightness_match=params.brightness_match,
            color_temp_match=params.color_temp_match,
            shadow_lift=params.shadow_lift,
        )
        char_opt_srgb = match_character_optics(
            char_toned_srgb,
            char_alpha,
            bg_roi_orig_srgb,
            blur_match_strength=params.blur_match_strength,
            noise_match_strength=params.noise_match_strength,
        )

        char_linear = srgb_to_linear(char_opt_srgb)
        if abs(params.char_exposure) > 0.01:
            char_linear = np.clip(char_linear * (2.0 ** params.char_exposure), 0.0, 3.0).astype(np.float32)

        _report(64, "正在渲染 2.5D 立体光影与边缘轮廓光...")
        rim_w = max(2, int(round(7.0 * (h_bg / 1600.0))))
        char_lit_linear = apply_lighting_and_rim(
            char_linear,
            char_alpha,
            bg_patch_linear,
            light_vec3,
            light_color_linear=light_color_linear,
            shading_25d_strength=params.shading_25d_strength,
            rim_strength=params.rim_strength,
            rim_width=rim_w,
            rim_erase_sharpness=1.6,
            light_wrap_strength=params.light_wrap_strength,
            edge_mode=params.edge_mode,
            rim_texture_weight=params.rim_texture_weight,
            rim_falloff=params.rim_falloff,
            hair_transmission=params.hair_transmission,
            normal_detail_mix=params.normal_detail_mix,
            rim_erase_map=rim_erase_map,
        )

        _report(78, "正在生成物理阴影与深度前景遮挡...")
        if params.shadow_mode == "real":
            comp_linear = render_real_contact_shadows(
                bg_bokeh_linear,
                depth_map,
                z_char,
                char_alpha,
                bbox,
                azimuth_deg=az_deg,
                shadow_opacity=params.shadow_opacity,
                shadow_length=params.shadow_length,
            )
        else:
            comp_linear = render_shadows(
                bg_bokeh_linear,
                depth_map,
                z_char,
                char_alpha,
                bbox,
                azimuth_deg=az_deg,
                shadow_opacity=params.shadow_opacity,
                shadow_length=params.shadow_length,
            )

        face_mask = None
        if params.face_protect_enabled and self.char_full_bgra is not None:
            fn = self._get_char_face_norm()
            if fn is not None:
                fn_x, fn_y, fn_w, fn_h = fn
                fc_x = x1 + (fn_x + fn_w / 2.0) * target_w
                fc_y = y1 + (fn_y + fn_h / 2.0) * target_h
                fr_w = max(10, int(round(fn_w * target_w * 0.65)))
                fr_h = max(12, int(round(fn_h * target_h * 0.75)))
                if 0 <= fc_x < w_bg and 0 <= fc_y < h_bg:
                    fm = np.zeros((h_bg, w_bg), dtype=np.float32)
                    cv2.ellipse(fm, (int(round(fc_x)), int(round(fc_y))), (fr_w, fr_h), 0, 0, 360, 1.0, -1)
                    k_f = max(15, int(round(max(fr_w, fr_h) * 0.65)) | 1)
                    face_mask = cv2.GaussianBlur(fm, (k_f, k_f), 0)

        def _run_light_fx(target_linear: np.ndarray, occ_str: float, **kwargs) -> np.ndarray:
            if params.light_fx_type == "丁达尔光":
                return apply_tyndall_rays(
                    target_linear,
                    char_alpha,
                    bbox,
                    azimuth_deg=az_deg,
                    light_color_linear=light_color_linear,
                    strength=params.tyndall_strength,
                    ray_count=params.tyndall_count,
                    ray_softness=params.tyndall_softness,
                    ray_length=params.tyndall_length,
                    pos_x=params.fx_pos_x,
                    pos_y=params.fx_pos_y,
                    aim_deg=params.fx_aim_deg,
                    occlusion_strength=occ_str,
                    face_mask=face_mask,
                )
            if params.light_fx_type == "聚光灯":
                return apply_spotlight_fx(
                    target_linear,
                    azimuth_deg=az_deg,
                    light_color_linear=light_color_linear,
                    strength=params.spot_strength,
                    width=params.spot_width,
                    softness=params.spot_softness,
                    dust_amount=params.spot_dust,
                    pos_x=params.fx_pos_x,
                    pos_y=params.fx_pos_y,
                    aim_deg=params.fx_aim_deg,
                    tip_hide=params.spot_tip_hide,
                    char_alpha=char_alpha,
                    bbox=bbox,
                    occlusion_strength=occ_str,
                    dust_pass=kwargs.get("dust_pass", "all"),
                    face_mask=face_mask,
                )
            if params.light_fx_type == "光晕":
                return apply_lens_flare_fx(
                    target_linear,
                    azimuth_deg=az_deg,
                    halo_strength=params.halo_strength,
                    halo_radius=params.halo_radius,
                    halo_inner_linear=rgb255_to_bgr_linear(params.halo_inner_rgb),
                    halo_outer_linear=rgb255_to_bgr_linear(params.halo_outer_rgb),
                    flare_strength=params.flare_strength,
                    flare_size=params.flare_size,
                    flare_shape=params.flare_shape,
                    flare_color_linear=rgb255_to_bgr_linear(params.flare_color_rgb),
                    pos_x=params.fx_pos_x,
                    pos_y=params.fx_pos_y,
                    aim_deg=params.fx_aim_deg,
                    halo_core_strength=params.halo_core_strength,
                    halo_arc_hide=params.halo_arc_hide,
                    halo_width=params.halo_width,
                    flare_angle=params.flare_angle,
                    face_mask=face_mask,
                )
            return target_linear

        # 光晕属于相机镜头镜片级光学现象，不在人身后被身体挖空，始终在最前端完整渲染
        is_lens_level_fx = (params.light_fx_type == "光晕")
        fx_behind = (
            params.light_fx_enabled
            and (not is_lens_level_fx)
            and params.light_fx_layer in ("置于人物后方", "人后体积逆光")
        )
        fx_delta_roi = None

        if fx_behind:
            _report(85, f"正在计算 {params.light_fx_type} 体积射线遮挡与人后微尘...")
            want_penetration = (
                cx2 > cx1 and cy2 > cy1
                and params.light_fx_layer == "人后体积逆光"
                and params.fx_edge_spill > 0.01
            )
            pre_fx_roi = comp_linear[cy1:cy2, cx1:cx2].copy() if want_penetration else None
            comp_linear = _run_light_fx(
                comp_linear,
                params.fx_occlusion_strength,
                **({"dust_pass": "bg"} if params.light_fx_type == "聚光灯" else {})
            )
            if pre_fx_roi is not None:
                fx_delta_roi = np.maximum(0.0, comp_linear[cy1:cy2, cx1:cx2] - pre_fx_roi)

        if cx2 > cx1 and cy2 > cy1:
            sy1, sy2, sx1, sx2 = cy1 - y1, cy2 - y1, cx1 - x1, cx2 - x1
            sub_bg = comp_linear[cy1:cy2, cx1:cx2]
            sub_fg = char_lit_linear[sy1:sy2, sx1:sx2]
            sub_alpha = char_alpha[sy1:sy2, sx1:sx2]

            if params.occlusion_enabled:
                occ_full = self.depth_engine.compute_occlusion_mask(depth_map, z_char, bias=params.occlusion_bias)
                sub_occ = occ_full[cy1:cy2, cx1:cx2]
                sub_alpha = sub_alpha * (1.0 - sub_occ)

            a_3 = sub_alpha[:, :, None]
            sub_comp = sub_bg * (1.0 - a_3) + sub_fg * a_3

            if fx_delta_roi is not None:
                rh, rw = sub_alpha.shape[:2]
                yy_norm = np.linspace(1.0, 0.12, rh, dtype=np.float32)[:, None]
                lx_dir = math.cos(math.radians(az_deg))
                xx_dir = np.linspace(1.0 if lx_dir < 0 else 0.25, 0.25 if lx_dir < 0 else 1.0, rw, dtype=np.float32)[None, :]
                spatial_pass = np.clip(0.65 * yy_norm + 0.35 * xx_dir, 0.08, 1.0)

                k_sp = max(5, int(round(min(target_w, target_h) * 0.025)) | 1)
                eroded_sp = cv2.erode(sub_alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sp, k_sp)), iterations=1)
                edge_band = np.clip(sub_alpha - eroded_sp, 0.0, 1.0)
                edge_band = cv2.GaussianBlur(edge_band, (0, 0), sigmaX=max(2.5, k_sp * 0.55))

                spill_strength = float(params.fx_edge_spill)
                body_pass = np.clip(edge_band * 0.85 + spatial_pass * (spill_strength * 0.65), 0.0, 1.0) * sub_alpha

                fx_soft = cv2.GaussianBlur(fx_delta_roi, (0, 0), sigmaX=max(2.0, k_sp * 0.45))
                sub_comp = sub_comp + fx_soft * body_pass[:, :, None] * (0.35 + spill_strength * 0.75) * np.maximum(0.0, 1.0 - sub_comp * 0.42)

            comp_linear[cy1:cy2, cx1:cx2] = sub_comp

        if self.bg_sharp_mask is not None and np.any(self.bg_sharp_mask > 0.005):
            sharp_full = cv2.resize(self.bg_sharp_mask, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR)[:, :, None]
            bg_orig_linear = srgb_to_linear(bg_srgb)
            comp_linear = comp_linear * (1.0 - sharp_full) + bg_orig_linear * sharp_full

        if params.light_fx_enabled and fx_behind and params.light_fx_type == "聚光灯":
            comp_linear = apply_spotlight_fx(
                comp_linear,
                azimuth_deg=az_deg,
                light_color_linear=light_color_linear,
                strength=0.0,
                width=params.spot_width,
                softness=params.spot_softness,
                dust_amount=params.spot_dust,
                pos_x=params.fx_pos_x,
                pos_y=params.fx_pos_y,
                aim_deg=params.fx_aim_deg,
                tip_hide=params.spot_tip_hide,
                dust_pass="fg",
                face_mask=face_mask,
            )

        if params.light_fx_enabled and (is_lens_level_fx or not fx_behind):
            _report(90, f"正在叠加 {params.light_fx_type} 特效...")
            comp_linear = _run_light_fx(comp_linear, 0.0)

        _report(95, "正在转换色彩空间并准备编码...")
        out_srgb = linear_to_srgb(comp_linear)
        return np.clip(out_srgb * 255.0, 0, 255).astype(np.uint8)