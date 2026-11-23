import cv2
import numpy as np


def match_color_and_tone(
    char_srgb: np.ndarray,
    char_alpha: np.ndarray,
    bg_roi_srgb: np.ndarray,
    brightness_match: float = 0.70,
    color_temp_match: float = 0.45,
    shadow_lift: float = 0.50,
) -> np.ndarray:
    if bg_roi_srgb.size == 0:
        return char_srgb

    fg_mask = char_alpha > 0.35
    if not np.any(fg_mask):
        return char_srgb

    char_lab = cv2.cvtColor(np.clip(char_srgb, 0.0, 1.0), cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(np.clip(bg_roi_srgb, 0.0, 1.0), cv2.COLOR_BGR2LAB)

    ch_L = char_lab[:, :, 0]
    bg_L = bg_lab[:, :, 0]

    if brightness_match > 0.01:
        ch_med = float(np.median(ch_L[fg_mask])) + 1e-4
        bg_med = float(np.median(bg_L))
        target_med = np.clip(bg_med * 1.18, 16.0, 88.0)
        ideal_scale = np.clip(target_med / ch_med, 0.35, 1.50)
        final_scale = 1.0 + (ideal_scale - 1.0) * brightness_match
        ch_L = np.clip(ch_L * final_scale, 0.0, 100.0)

    if shadow_lift > 0.01:
        bg_black = float(np.percentile(bg_L, 5))
        ch_black = float(np.percentile(ch_L[fg_mask], 5))
        if bg_black > ch_black:
            lift = (bg_black - ch_black) * shadow_lift
            shadow_weight = np.clip(1.0 - (ch_L / 55.0), 0.0, 1.0)
            ch_L = np.clip(ch_L + lift * shadow_weight, 0.0, 100.0)

    char_lab[:, :, 0] = ch_L

    if color_temp_match > 0.01:
        for c in (1, 2):
            ch_c_med = float(np.median(char_lab[:, :, c][fg_mask]))
            bg_c_med = float(np.median(bg_lab[:, :, c]))
            shift = np.clip((bg_c_med - ch_c_med) * color_temp_match, -28.0, 28.0)
            char_lab[:, :, c] = np.clip(char_lab[:, :, c] + shift, -127.0, 127.0)

    out_bgr = cv2.cvtColor(char_lab, cv2.COLOR_LAB2BGR)
    return np.clip(out_bgr, 0.0, 1.0).astype(np.float32)


def render_shadows(
    bg_linear: np.ndarray,
    depth_map: np.ndarray,
    z_char: float,
    char_alpha: np.ndarray,
    bbox: tuple[int, int, int, int],
    azimuth_deg: float,
    shadow_opacity: float = 0.70,
    shadow_length: float = 0.85,
) -> np.ndarray:
    """
    双层物理阴影 (所有大半径高斯模糊统一限幅在 1000px 内算完再上采样，2400W 导出提速 100 倍)
    """
    if shadow_opacity <= 0.01:
        return bg_linear

    h_bg, w_bg = bg_linear.shape[:2]
    x1, y1, x2, y2 = bbox
    ch_h, ch_w = char_alpha.shape

    ys, xs = np.where(char_alpha > 0.35)
    if len(ys) == 0:
        return bg_linear

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w_bg, x2), min(h_bg, y2)
    roi = bg_linear[cy1:cy2, cx1:cx2]
    if roi.size > 0:
        roi_lum = 0.0722 * roi[:, :, 0] + 0.7152 * roi[:, :, 1] + 0.2126 * roi[:, :, 2]
        dark_mask = roi_lum <= np.percentile(roi_lum, 20)
        if np.any(dark_mask):
            shadow_tint = np.median(roi[dark_mask], axis=0)
            shadow_tint = np.clip(shadow_tint / (shadow_tint.max() + 1e-4) * 0.18, 0.01, 0.22)
        else:
            shadow_tint = np.array([0.05, 0.04, 0.03], dtype=np.float32)
    else:
        shadow_tint = np.array([0.05, 0.04, 0.03], dtype=np.float32)

    # 在 <= 1000px 工作分辨率上计算阴影掩膜
    sc = min(1.0, 1000.0 / float(max(h_bg, w_bg)))
    sw, sh = max(64, int(round(w_bg * sc))), max(64, int(round(h_bg * sc)))
    sx1, sy1 = int(round(x1 * sc)), int(round(y1 * sc))
    sx2, sy2 = int(round(x2 * sc)), int(round(y2 * sc))
    sch_w, sch_h = max(8, sx2 - sx1), max(8, sy2 - sy1)

    alpha_sm = cv2.resize(char_alpha, (sch_w, sch_h), interpolation=cv2.INTER_AREA) if sc < 1.0 else char_alpha
    depth_sm = cv2.resize(depth_map, (sw, sh), interpolation=cv2.INTER_LINEAR) if sc < 1.0 else depth_map

    ys_s, xs_s = np.where(alpha_sm > 0.35)
    if len(ys_s) == 0:
        return bg_linear

    # 1. 脚底 AO
    y_foot_local = int(np.percentile(ys_s, 97))
    foot_band = np.zeros((sch_h, sch_w), dtype=np.float32)
    band_top = max(0, y_foot_local - max(3, int(sch_h * 0.04)))
    foot_band[band_top : min(sch_h, y_foot_local + 3), :] = alpha_sm[band_top : min(sch_h, y_foot_local + 3), :]

    ao_sm = np.zeros((sh, sw), dtype=np.float32)
    dcx1, dcy1 = max(0, sx1), max(0, sy1)
    dcx2, dcy2 = min(sw, sx2), min(sh, sy2)
    if dcy2 > dcy1 and dcx2 > dcx1:
        ao_sm[dcy1:dcy2, dcx1:dcx2] = foot_band[dcy1 - sy1 : dcy2 - sy1, dcx1 - sx1 : dcx2 - sx1]
    ao_blur = cv2.GaussianBlur(ao_sm, (0, 0), sigmaX=max(4.0, sch_w * 0.045), sigmaY=max(2.0, sch_h * 0.012))
    total_shadow_sm = np.clip(ao_blur * 1.55 * shadow_opacity, 0.0, 0.90)

    # 2. 投射阴影
    if shadow_length > 0.05:
        az = np.radians(azimuth_deg)
        skew_x = -np.cos(az) * shadow_length * 0.9
        squash_y = max(0.18, abs(np.sin(az)) * 0.45 * shadow_length)

        foot_abs_x = sx1 + float(np.median(xs_s[ys_s >= np.percentile(ys_s, 88)]))
        foot_abs_y = sy1 + float(y_foot_local)

        M = np.array(
            [
                [1.0, -skew_x, sx1 + skew_x * y_foot_local],
                [0.0, squash_y, foot_abs_y - squash_y * y_foot_local],
            ],
            dtype=np.float32,
        )
        cast_raw = cv2.warpAffine(alpha_sm, M, (sw, sh), flags=cv2.INTER_LINEAR, borderValue=0.0)
        depth_valid = np.clip((depth_sm - (z_char - 0.16)) / 0.08, 0.0, 1.0)
        cast_raw *= depth_valid

        near_blur = cv2.GaussianBlur(cast_raw, (0, 0), sigmaX=max(2.5, sch_w * 0.02))
        far_blur = cv2.GaussianBlur(cast_raw, (0, 0), sigmaX=max(8.0, sch_w * 0.09))
        yy, xx = np.ogrid[:sh, :sw]
        dist_from_foot = np.hypot(xx - foot_abs_x, (yy - foot_abs_y) * 2.0)
        blend_far = np.clip(dist_from_foot / max(60.0, sch_h * 0.8), 0.0, 1.0)
        cast_soft = near_blur * (1.0 - blend_far) + far_blur * blend_far
        cast_alpha = np.clip(cast_soft * 0.78 * shadow_opacity, 0.0, 0.80)
        total_shadow_sm = np.maximum(total_shadow_sm, cast_alpha)

    total_shadow = cv2.resize(total_shadow_sm, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR) if sc < 1.0 else total_shadow_sm
    sh_3 = total_shadow[:, :, None]
    out = bg_linear * (1.0 - sh_3) + (bg_linear * shadow_tint[None, None, :]) * sh_3
    return np.clip(out, 0.0, 4.0).astype(np.float32)

def render_real_contact_shadows(
    bg_linear: np.ndarray,
    depth_map: np.ndarray,
    z_char: float,
    char_alpha: np.ndarray,
    bbox: tuple[int, int, int, int],
    azimuth_deg: float,
    shadow_opacity: float = 0.50,
    shadow_length: float = 0.65,
) -> np.ndarray:
    """
    真人场照专用接触环境遮蔽 (Depth-Aware Contact AO):
    保留原片现场楼梯/地面真实投影，仅在人物与同深度或略后表面（台阶、坐面、贴墙处）
    的轮廓下方与背光侧进行柔和接触压暗，把人“按”进画面而不破坏台阶立面。
    """
    if shadow_opacity <= 0.01:
        return bg_linear

    h_bg, w_bg = bg_linear.shape[:2]
    x1, y1, x2, y2 = bbox

    sc = min(1.0, 1000.0 / float(max(h_bg, w_bg)))
    sw, sh = max(64, int(round(w_bg * sc))), max(64, int(round(h_bg * sc)))
    sx1, sy1 = int(round(x1 * sc)), int(round(y1 * sc))
    sx2, sy2 = int(round(x2 * sc)), int(round(y2 * sc))
    sch_w, sch_h = max(8, sx2 - sx1), max(8, sy2 - sy1)

    alpha_sm = cv2.resize(char_alpha, (sch_w, sch_h), interpolation=cv2.INTER_AREA) if sc < 1.0 else char_alpha
    depth_sm = cv2.resize(depth_map, (sw, sh), interpolation=cv2.INTER_LINEAR) if sc < 1.0 else depth_map

    ys_s, xs_s = np.where(alpha_sm > 0.35)
    if len(ys_s) == 0:
        return bg_linear

    full_alpha = np.zeros((sh, sw), dtype=np.float32)
    dcx1, dcy1 = max(0, sx1), max(0, sy1)
    dcx2, dcy2 = min(sw, sx2), min(sh, sy2)
    if dcy2 <= dcy1 or dcx2 <= dcx1:
        return bg_linear
    full_alpha[dcy1:dcy2, dcx1:dcx2] = alpha_sm[dcy1 - sy1 : dcy2 - sy1, dcx1 - sx1 : dcx2 - sx1]

    # 1. 沿光反方向+重力下方微偏移，形成短距离贴合接触遮蔽
    az = np.radians(azimuth_deg)
    off_x = -np.cos(az) * max(4.0, sch_w * 0.035 * shadow_length)
    off_y = max(4.0, sch_h * 0.018) + max(0.0, -np.sin(az) * sch_h * 0.015)

    M = np.array([[1.0, 0.0, off_x], [0.0, 1.0, off_y]], dtype=np.float32)
    shifted = cv2.warpAffine(full_alpha, M, (sw, sh), flags=cv2.INTER_LINEAR, borderValue=0.0)

    # 接触带：人物外围一圈紧贴轮廓的环状带
    k_ring = max(5, int(round(min(sch_w, sch_h) * 0.045)) | 1)
    dilated = cv2.dilate(shifted, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ring, k_ring)))
    contact_band = np.clip(dilated - full_alpha * 0.85, 0.0, 1.0)

    # 下半身/接触面权重略高，头顶天空不产生接触暗影
    y_min, y_max = float(np.min(ys_s) + sy1), float(np.max(ys_s) + sy1)
    yy_grid = np.linspace(0.0, 1.0, sh, dtype=np.float32)[:, None]
    y_norm = np.clip((yy_grid - y_min / sh) / max(1e-3, (y_max - y_min) / sh), 0.0, 1.0)
    vertical_weight = 0.25 + 0.75 * (y_norm ** 1.3)

    # 深度门控：只暗化深度跟人物接近的接触物体表面（防止糊到远墙上）
    depth_diff = np.abs(depth_sm - float(z_char))
    depth_gate = np.clip(1.0 - (depth_diff - 0.03) / 0.14, 0.0, 1.0)

    ao_sm = contact_band * vertical_weight * depth_gate
    ao_sm = cv2.GaussianBlur(ao_sm, (0, 0), sigmaX=max(3.5, sch_w * 0.025), sigmaY=max(3.0, sch_h * 0.018))
    ao_sm = np.clip(ao_sm * 1.15 * shadow_opacity, 0.0, 0.65)

    ao_full = cv2.resize(ao_sm, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR) if sc < 1.0 else ao_sm
    ao_3 = ao_full[:, :, None]

    # 纯物理乘算压暗，保留台阶/地砖 100% 原始纹理
    out = bg_linear * (1.0 - ao_3 * 0.85)
    return np.clip(out, 0.0, 4.0).astype(np.float32)