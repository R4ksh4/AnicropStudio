import math
import cv2
import numpy as np
from scipy.signal import fftconvolve


def _make_soft_disk_kernel(radius: float) -> np.ndarray:
    r_int = max(1, int(math.ceil(radius + 1.0)))
    y, x = np.ogrid[-r_int : r_int + 1, -r_int : r_int + 1]
    dist = np.sqrt(x * x + y * y)
    kernel = np.clip((radius + 0.75 - dist) / 1.5, 0.0, 1.0).astype(np.float32)
    kernel /= kernel.sum() + 1e-8
    return kernel


def _masked_disk_blur(
    src_small: np.ndarray,
    weight_small: np.ndarray | None,
    radius_small: float,
) -> np.ndarray:
    """掩膜归一化散景卷积：人物区权重为 0，彻底消除背景模糊时衣服/鞋子像素向外渗色，同时保留真实地面阴影"""
    if radius_small < 0.65:
        return src_small

    kernel = _make_soft_disk_kernel(radius_small)
    pad = kernel.shape[0] // 2

    if weight_small is None:
        src_pad = cv2.copyMakeBorder(src_small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        return fftconvolve(src_pad, kernel[:, :, None], mode="valid", axes=(0, 1)).astype(np.float32)

    weighted_src = src_small * weight_small[:, :, None]
    w_src_pad = cv2.copyMakeBorder(weighted_src, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    w_pad = cv2.copyMakeBorder(weight_small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    num = fftconvolve(w_src_pad, kernel[:, :, None], mode="valid", axes=(0, 1)).astype(np.float32)
    den = fftconvolve(w_pad, kernel, mode="valid", axes=(0, 1)).astype(np.float32)

    normalized = num / np.maximum(den, 1e-4)[:, :, None]
    blend_orig = np.clip((0.03 - den) / 0.03, 0.0, 1.0)[:, :, None]
    if np.any(blend_orig > 0.0):
        fallback = cv2.GaussianBlur(src_small, (0, 0), sigmaX=max(1.0, radius_small * 0.5))
        normalized = normalized * (1.0 - blend_orig) + fallback * blend_orig
    return normalized


def apply_depth_bokeh(
    bg_linear: np.ndarray,
    depth_map: np.ndarray,
    z_focus: float,
    aperture_strength: float = 0.45,
    highlight_boost: float = 0.5,
    num_layers: int = 5,
    dof_tolerance: float = 0.06,
    fg_exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    if aperture_strength <= 0.01:
        return bg_linear

    h, w = bg_linear.shape[:2]
    max_radius = (max(h, w) / 1600.0) * 15.0 * float(aperture_strength)
    if max_radius < 0.7:
        return bg_linear

    if highlight_boost > 0.01:
        lum = 0.0722 * bg_linear[:, :, 0] + 0.7152 * bg_linear[:, :, 1] + 0.2126 * bg_linear[:, :, 2]
        extra = np.clip(lum - 0.72, 0.0, 2.0) * (highlight_boost * 1.8)
        src_boosted = bg_linear * (1.0 + extra[:, :, None])
    else:
        src_boosted = bg_linear

    tol = max(0.0, float(dof_tolerance))
    depth_diff = np.abs(depth_map - float(z_focus))
    falloff_span = max(0.14, 0.55 - tol * 0.4)
    coc_full = np.clip((depth_diff - tol) / falloff_span, 0.0, 1.0).astype(np.float32)

    if highlight_boost > 0.01:
        src_boosted = bg_linear * (1.0 - coc_full[:, :, None]) + src_boosted * coc_full[:, :, None]

    work_long = min(960, max(h, w) if max(h, w) <= 1200 else max(h, w) // 2)
    scale = work_long / float(max(h, w))
    hw, hh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))

    src_small = cv2.resize(src_boosted, (hw, hh), interpolation=cv2.INTER_AREA)
    coc_small = cv2.resize(coc_full, (hw, hh), interpolation=cv2.INTER_LINEAR)
    coc_small = cv2.GaussianBlur(coc_small, (0, 0), sigmaX=max(1.5, min(hw, hh) * 0.008))

    weight_small = None
    if fg_exclusion_mask is not None:
        fg_sm = cv2.resize(fg_exclusion_mask, (hw, hh), interpolation=cv2.INTER_AREA)
        weight_small = np.clip(1.0 - fg_sm, 0.0, 1.0).astype(np.float32)

    layers = max(2, int(num_layers))
    r_work = max_radius * scale

    blur_stack = [src_small]
    for k in range(1, layers):
        rk = r_work * (k / float(layers - 1))
        if rk < 0.65:
            blur_stack.append(cv2.GaussianBlur(src_small, (0, 0), sigmaX=max(0.4, rk * 0.6)))
        else:
            blur_stack.append(_masked_disk_blur(src_small, weight_small, rk))

    pos = np.clip(coc_small * (layers - 1), 0.0, float(layers - 1) - 1e-5)
    idx0 = np.floor(pos).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, layers - 1)
    frac = (pos - idx0)[:, :, None]

    blurred_small = np.zeros_like(src_small)
    for k in range(layers):
        m0 = idx0 == k
        if np.any(m0):
            blurred_small[m0] += blur_stack[k][m0] * (1.0 - frac[m0])
        m1 = idx1 == k
        if np.any(m1):
            blurred_small[m1] += blur_stack[k][m1] * frac[m1]

    out_bokeh = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 焦平面内直接保留全分辨率原片，保证脚底台阶纹理 100% 锐利
    coc_smooth = cv2.GaussianBlur(coc_full, (0, 0), sigmaX=max(2.0, min(w, h) * 0.005))
    sharp_keep = np.clip(1.0 - coc_smooth / 0.18, 0.0, 1.0)[:, :, None]
    out = bg_linear * sharp_keep + out_bokeh * (1.0 - sharp_keep)
    return np.clip(out, 0.0, 4.0).astype(np.float32)


def compute_volumetric_occlusion_map(
    sw: int,
    sh: int,
    char_alpha: np.ndarray,
    bbox: tuple[int, int, int, int],
    w_bg: int,
    h_bg: int,
    pos_x: float,
    pos_y: float,
    occlusion_strength: float = 0.70,
) -> np.ndarray:
    """
    从光源原点 (pos_x, pos_y) 沿射线向外积分人体遮挡:
    光线碰到人体前不受挡；穿过人体后顺着射线继续往下游传播的光束被人体挡出真实体积暗影。
    """
    if occlusion_strength <= 0.01 or char_alpha is None or bbox is None:
        return np.zeros((sh, sw), dtype=np.float32)

    occ = np.zeros((sh, sw), dtype=np.float32)
    sc_x = sw / float(max(1, w_bg))
    sc_y = sh / float(max(1, h_bg))
    x1, y1, x2, y2 = bbox
    sx1, sy1 = int(round(x1 * sc_x)), int(round(y1 * sc_y))
    sx2, sy2 = int(round(x2 * sc_x)), int(round(y2 * sc_y))
    cx1, cy1, cx2, cy2 = max(0, sx1), max(0, sy1), min(sw, sx2), min(sh, sy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return occ

    ch_sm = cv2.resize(char_alpha, (max(1, sx2 - sx1), max(1, sy2 - sy1)), interpolation=cv2.INTER_AREA)
    occ[cy1:cy2, cx1:cx2] = ch_sm[cy1 - sy1 : cy2 - sy1, cx1 - sx1 : cx2 - sx1]

    # 稍微向内收一点，让发丝和最外圈衣物边缘依然透出绕射亮边
    occ = cv2.erode(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    sx = float(pos_x) * sw
    sy = float(pos_y) * sh
    shadow = np.zeros_like(occ)
    scale_i = 1.04
    for _ in range(26):
        M = np.array([[scale_i, 0.0, sx * (1.0 - scale_i)], [0.0, scale_i, sy * (1.0 - scale_i)]], dtype=np.float32)
        warped = cv2.warpAffine(occ, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        shadow = np.maximum(shadow, warped)
        scale_i *= 1.055

    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(2.0, min(sw, sh) * 0.012))
    return np.clip(shadow * float(occlusion_strength), 0.0, 0.94).astype(np.float32)


def apply_tyndall_rays(
    comp_linear: np.ndarray,
    char_alpha: np.ndarray,
    bbox: tuple[int, int, int, int],
    azimuth_deg: float,
    light_color_linear: np.ndarray,
    strength: float = 0.6,
    ray_count: float = 1.0,
    ray_softness: float = 0.50,
    ray_length: float = 0.70,
    pos_x: float = 0.20,
    pos_y: float = 0.15,
    aim_deg: float = 55.0,
    occlusion_strength: float = 0.65,
    face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """模式 1: 丁达尔光 (固定发光中心 X/Y + 照射角度 + 数量/柔化/长度 + 体积遮挡)"""
    if strength <= 0.01:
        return comp_linear

    h, w = comp_linear.shape[:2]
    sc = 420.0 / float(max(h, w))
    sw, sh = max(48, int(round(w * sc))), max(48, int(round(h * sc)))

    bg_small = cv2.resize(comp_linear, (sw, sh), interpolation=cv2.INTER_AREA)
    lum = 0.0722 * bg_small[:, :, 0] + 0.7152 * bg_small[:, :, 1] + 0.2126 * bg_small[:, :, 2]

    diag = math.hypot(sw, sh)
    sx = float(pos_x) * sw
    sy = float(pos_y) * sh

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    angles = np.arctan2(yy - sy, xx - sx)

    aim_rad = math.radians(aim_deg)
    ang_diff = np.arctan2(np.sin(angles - aim_rad), np.cos(angles - aim_rad))
    aim_bias = 0.28 + 0.72 * np.exp(-0.5 * (ang_diff / 0.85) ** 2)

    fm = max(0.25, float(ray_count))
    raw_wave = (
        0.50 * np.sin(angles * (12.0 * fm) + 0.4)
        + 0.30 * np.sin(angles * (27.0 * fm) + 1.7)
        + 0.20 * np.cos(angles * (47.0 * fm))
    )
    rays_norm = np.clip(0.5 + 0.5 * raw_wave, 0.0, 1.0)
    soft = np.clip(float(ray_softness), 0.0, 1.0)
    gamma = 2.8 - soft * 1.9
    rays_pattern = 0.08 + 0.92 * (rays_norm ** max(0.4, gamma))
    emitter = (0.28 + 0.72 * np.clip((lum - 0.25) / 0.75, 0.0, 1.0)) * rays_pattern * aim_bias

    x1, y1, x2, y2 = bbox
    sx1, sy1, sx2, sy2 = int(round(x1 * sc)), int(round(y1 * sc)), int(round(x2 * sc)), int(round(y2 * sc))
    cx1, cy1, cx2, cy2 = max(0, sx1), max(0, sy1), min(sw, sx2), min(sh, sy2)
    if cx2 > cx1 and cy2 > cy1 and occlusion_strength > 0.01:
        ch_small = cv2.resize(char_alpha, (max(1, sx2 - sx1), max(1, sy2 - sy1)), interpolation=cv2.INTER_AREA)
        sub_occ = ch_small[cy1 - sy1 : cy2 - sy1, cx1 - sx1 : cx2 - sx1]
        emitter[cy1:cy2, cx1:cx2] *= (1.0 - sub_occ * np.clip(float(occlusion_strength), 0.0, 0.95))

    length_c = np.clip(float(ray_length), 0.15, 1.50)
    step_scale = 0.025 + length_c * 0.055
    decay_factor = min(0.95, 0.78 + length_c * 0.11)

    accum = np.zeros_like(emitter)
    decay = 1.0
    w_sum = 0.0
    for i in range(14):
        scale_i = 1.0 + step_scale * i
        M = np.array([[scale_i, 0.0, sx * (1.0 - scale_i)], [0.0, scale_i, sy * (1.0 - scale_i)]], dtype=np.float32)
        warped = cv2.warpAffine(emitter, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        accum += warped * decay
        w_sum += decay
        decay *= decay_factor

    accum /= max(1e-4, w_sum) * 0.85
    dist_map = np.hypot(xx - sx, yy - sy) / (diag + 1e-5)
    falloff = np.clip(1.22 - dist_map / (0.50 + length_c * 0.85), 0.02, 1.0) ** 1.2

    blur_sigma = 0.6 + soft * 3.8
    rays_small = cv2.GaussianBlur(accum * falloff, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    rays_full = cv2.resize(rays_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if face_mask is not None:
        rays_full *= np.clip(1.0 - face_mask * 0.85, 0.0, 1.0)

    ray_rgb = rays_full[:, :, None] * light_color_linear[None, None, :] * (strength * 0.65)
    out = comp_linear + ray_rgb * np.maximum(0.0, 1.0 - comp_linear * 0.42)
    return np.clip(out, 0.0, 4.0).astype(np.float32)


def apply_spotlight_fx(
    comp_linear: np.ndarray,
    azimuth_deg: float,
    light_color_linear: np.ndarray,
    strength: float = 0.7,
    width: float = 0.45,
    softness: float = 0.50,
    dust_amount: float = 0.50,
    pos_x: float = 0.20,
    pos_y: float = 0.15,
    aim_deg: float = 55.0,
    tip_hide: float = 0.45,
    char_alpha: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    occlusion_strength: float = 0.0,
    dust_pass: str = "all",
    face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """模式 2: 聚光灯 (3D 分层微尘: 人前大颗粒柔焦微尘 + 人后背景微尘 + 人体体积遮挡)"""
    render_beam = (dust_pass != "fg") and (strength > 0.01)
    render_dust = (dust_amount > 0.01)
    if not render_beam and not render_dust:
        return comp_linear

    h, w = comp_linear.shape[:2]
    sc = min(1.0, 720.0 / float(max(h, w)))
    sw, sh = max(64, int(round(w * sc))), max(64, int(round(h * sc)))

    diag = math.hypot(sw, sh)
    sx = float(pos_x) * sw
    sy = float(pos_y) * sh

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    angles = np.arctan2(yy - sy, xx - sx)
    center_angle = math.radians(aim_deg)
    ang_diff = np.abs(np.arctan2(np.sin(angles - center_angle), np.cos(angles - center_angle)))

    half_angle = 0.06 + np.clip(width, 0.05, 1.0) * 0.40
    feather = 0.02 + np.clip(softness, 0.02, 1.0) * 0.35
    cone = np.clip((half_angle + feather - ang_diff) / feather, 0.0, 1.0) ** 1.6

    dist = np.hypot(xx - sx, yy - sy)
    dist_falloff = np.exp(-dist / (diag * 1.15))
    if tip_hide > 0.01:
        tip_fade = 1.0 - np.exp(-((dist / max(8.0, diag * 0.18 * float(tip_hide))) ** 2))
        dist_falloff = dist_falloff * tip_fade
    beam_sm_raw = cone * dist_falloff
    beam_sm = beam_sm_raw.copy()

    # 人体射线遮挡：光束穿过人体下游被截断
    if occlusion_strength > 0.01 and char_alpha is not None and bbox is not None:
        occ_sm = compute_volumetric_occlusion_map(
            sw, sh, char_alpha, bbox, w, h, pos_x, pos_y, occlusion_strength=occlusion_strength
        )
        beam_sm = beam_sm * (1.0 - occ_sm)

    # 3D 分层空气微尘粒子（固定随机种子，前后景无缝空间交错）
    dust_sm = np.zeros((sh, sw), dtype=np.float32)
    if render_dust:
        rng = np.random.default_rng(2026)
        num_motes = int(60 + dust_amount * 240)
        xs_m = rng.integers(0, sw, size=num_motes)
        ys_m = rng.integers(0, sh, size=num_motes)
        rs_m = rng.uniform(0.8, 3.2, size=num_motes)
        bs_m = rng.uniform(0.35, 1.0, size=num_motes)
        depth_m = rng.uniform(0.0, 1.0, size=num_motes)

        for mx, my, mr, mb, md in zip(xs_m, ys_m, rs_m, bs_m, depth_m):
            # 空间分流：45% 粒子飘在人前面（前景），55% 留在人身后（背景）
            if dust_pass == "bg" and md <= 0.45:
                continue
            if dust_pass == "fg" and md > 0.45:
                continue
            # 前景微尘离镜头更近，颗粒更大、边缘更柔
            actual_r = mr * (1.35 if md <= 0.45 else 0.85)
            cv2.circle(dust_sm, (int(mx), int(my)), max(1, int(round(actual_r))), float(mb), -1, lineType=cv2.LINE_AA)

        blur_sig = 1.6 if dust_pass == "fg" else 1.1
        dust_sm = cv2.GaussianBlur(dust_sm, (0, 0), sigmaX=blur_sig, sigmaY=blur_sig)
        # 前景微尘在空间迎光锥内发亮，背景微尘跟随遮挡光锥
        beam_mask = beam_sm if dust_pass == "bg" else beam_sm_raw
        dust_sm *= np.clip(beam_mask * 1.35, 0.0, 1.0)

    if render_beam:
        beam_full = cv2.resize(beam_sm, (w, h), interpolation=cv2.INTER_LINEAR)
        beam_rgb = beam_full[:, :, None] * light_color_linear[None, None, :] * (strength * 0.55)
    else:
        beam_rgb = 0.0

    if render_dust:
        dust_full = cv2.resize(dust_sm, (w, h), interpolation=cv2.INTER_LINEAR)
        if face_mask is not None:
            # 驱散面部灰尘：五官区域 100% 绝对不留任何灰尘噪点
            dust_full *= np.clip(1.0 - face_mask * 1.5, 0.0, 1.0)
        dust_rgb = dust_full[:, :, None] * light_color_linear[None, None, :] * (dust_amount * 1.35)
    else:
        dust_rgb = 0.0

    out = comp_linear + (beam_rgb + dust_rgb) * np.maximum(0.0, 1.0 - comp_linear * 0.38)
    return np.clip(out, 0.0, 4.0).astype(np.float32)

def _draw_flare_ghost(canvas: np.ndarray, cx: float, cy: float, radius: float, shape: str, col: np.ndarray) -> None:
    r = max(3.0, radius)
    h, w = canvas.shape[:2]
    if cx + r < 0 or cx - r > w or cy + r < 0 or cy - r > h:
        return

    layer = np.zeros((h, w), dtype=np.float32)
    icx, icy = int(round(cx)), int(round(cy))
    ir = int(round(r))

    
    if shape in ("六边形光圈", "八边形光圈"):
        sides = 6 if shape == "六边形光圈" else 8
        pts = np.array(
            [
                [int(round(cx + r * math.cos(2 * math.pi * k / sides + 0.3))),
                 int(round(cy + r * math.sin(2 * math.pi * k / sides + 0.3)))]
                for k in range(sides)
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(layer, [pts], 0.35, lineType=cv2.LINE_AA)
        cv2.polylines(layer, [pts], True, 0.85, max(1, int(r * 0.08)), lineType=cv2.LINE_AA)
    else:  # 圆形散景
        cv2.circle(layer, (icx, icy), ir, 0.35, -1, lineType=cv2.LINE_AA)
        cv2.circle(layer, (icx, icy), ir, 0.90, max(1, int(r * 0.08)), lineType=cv2.LINE_AA)

    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(1.2, r * 0.08))
    canvas += layer[:, :, None] * col[None, None, :]


def apply_lens_flare_fx(
    comp_linear: np.ndarray,
    azimuth_deg: float,
    halo_strength: float = 0.70,
    halo_radius: float = 1.10,
    halo_inner_linear: np.ndarray | None = None,
    halo_outer_linear: np.ndarray | None = None,
    flare_strength: float = 0.60,
    flare_size: float = 0.75,
    flare_shape: str = "圆形散景",
    flare_color_linear: np.ndarray | None = None,
    pos_x: float = 0.08,
    pos_y: float = 0.08,
    aim_deg: float = 40.0,
    halo_core_strength: float = 0.35,
    halo_arc_hide: float = 0.50,
    halo_width: float = 0.45,
    flare_angle: float = 40.0,
    face_mask: np.ndarray | None = None,
) -> np.ndarray:
    """模式 3: 大半径弧形光学彩虹光晕 + 入射光核/反向隐藏 + 镜头光斑"""
    if halo_strength <= 0.01 and flare_strength <= 0.01:
        return comp_linear

    if halo_inner_linear is None:
        halo_inner_linear = np.array([0.45, 0.78, 1.0], dtype=np.float32)
    if halo_outer_linear is None:
        halo_outer_linear = np.array([0.85, 0.35, 0.95], dtype=np.float32)
    if flare_color_linear is None:
        flare_color_linear = np.array([0.55, 0.85, 1.0], dtype=np.float32)

    h, w = comp_linear.shape[:2]
    sc = min(1.0, 640.0 / float(max(h, w)))
    sw, sh = max(64, int(round(w * sc))), max(64, int(round(h * sc)))

    diag = math.hypot(sw, sh)
    sx = float(pos_x) * sw
    sy = float(pos_y) * sh
    aim_rad = math.radians(aim_deg)

    fx_sm = np.zeros((sh, sw, 3), dtype=np.float32)

    # 1. 大半径弧形彩虹色散光晕 + 入射光核独立控制 + 反向渐隐遮罩
    if halo_strength > 0.01:
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        dx = xx - sx
        dy = yy - sy
        dist = np.hypot(dx, dy)

        # 大半径弧线 (覆盖 0.35 ~ 2.5 倍对角线)
        rad_ring = max(30.0, diag * (0.28 + 0.42 * float(halo_radius)))
        ring_half_w = max(5.0, rad_ring * 0.10 * float(halo_width))

        # 入射点中心光核 (独立滑块控制，拉到 0 彻底隐藏)
        core_rgb = 0.0
        if halo_core_strength > 0.01:
            r_core = dist / max(18.0, diag * 0.14)
            core_glow = np.exp(-r_core * 2.4) * float(halo_core_strength)
            hot_white = np.array([1.0, 0.98, 0.95], dtype=np.float32)[None, None, :]
            core_rgb = (hot_white * 0.6 + halo_inner_linear[None, None, :] * 0.4) * core_glow[:, :, None]

        # 反向/背侧渐隐遮罩 (沿 aim_rad 前向保留，背侧平滑隐藏，绝无尖角)
        if halo_arc_hide > 0.01:
            ang = np.arctan2(dy, dx)
            cos_a = np.cos(ang - aim_rad)
            arc_mask = np.clip(0.5 + 0.5 * cos_a, 0.0, 1.0) ** (float(halo_arc_hide) * 3.5)
        else:
            arc_mask = 1.0

        u = (dist - (rad_ring - ring_half_w)) / (2.0 * ring_half_w)
        ring_env = np.where((u > 0.0) & (u < 1.0), np.sin(np.clip(u, 0.0, 1.0) * np.pi) ** 1.4, 0.0)
        ring_env = ring_env * arc_mask

        xp = np.array([0.0, 0.22, 0.45, 0.68, 0.85, 1.0], dtype=np.float32)
        b_ch = np.interp(u, xp, [0.95, 0.95, 0.35, 0.05, 0.04, 0.08]).astype(np.float32)
        g_ch = np.interp(u, xp, [0.12, 0.78, 0.95, 0.85, 0.38, 0.08]).astype(np.float32)
        r_ch = np.interp(u, xp, [0.55, 0.08, 0.15, 0.98, 1.00, 1.00]).astype(np.float32)
        rainbow_bgr = np.stack([b_ch, g_ch, r_ch], axis=-1)

        rainbow_tinted = rainbow_bgr * 0.78 + halo_outer_linear[None, None, :] * 0.22
        ring_rgb = rainbow_tinted * (ring_env[:, :, None] * 0.85)

        fx_sm += (core_rgb + ring_rgb) * float(halo_strength)

    # 2. 沿照射角度分布的镜头光斑
    if flare_strength > 0.01:
        flare_rad = math.radians(flare_angle)
        ax_x, ax_y = math.cos(flare_rad) * diag * 0.68, math.sin(flare_rad) * diag * 0.68
        ghost_specs = [
            (0.18, 0.35, 0.55),
            (0.42, 0.24, 0.85),
            (0.68, 0.58, 0.45),
            (0.95, 0.32, 0.75),
            (1.25, 0.82, 0.40),
            (1.52, 0.48, 0.62),
        ]
        base_r = diag * 0.045 * float(flare_size)
        for idx, (t_pos, rel_sz, rel_alpha) in enumerate(ghost_specs):
            gx = sx + ax_x * t_pos
            gy = sy + ax_y * t_pos
            shift = np.array([1.15, 0.95, 0.85], dtype=np.float32) if idx % 2 == 0 else np.array([0.85, 1.05, 1.15], dtype=np.float32)
            g_col = np.clip(flare_color_linear * shift, 0.0, 1.5) * (flare_strength * rel_alpha * 0.55)
            _draw_flare_ghost(fx_sm, gx, gy, base_r * rel_sz, flare_shape, g_col)

    fx_full = cv2.resize(fx_sm, (w, h), interpolation=cv2.INTER_LINEAR)
    if face_mask is not None:
        # 光晕柔化：面部减弱 82% 炫光，保住眼神反差，发丝与背景光斑不受影响
        fx_full *= np.clip(1.0 - face_mask[:, :, None] * 0.82, 0.0, 1.0)
    out = comp_linear + fx_full * np.maximum(0.0, 1.0 - comp_linear * 0.32)
    return np.clip(out, 0.0, 4.0).astype(np.float32)


def match_character_optics(
    char_srgb: np.ndarray,
    char_alpha: np.ndarray,
    bg_roi_srgb: np.ndarray,
    blur_match_strength: float = 0.25,
    noise_match_strength: float = 0.25,
) -> np.ndarray:
    out = char_srgb.copy()
    if bg_roi_srgb.size == 0:
        return out

    h, w = bg_roi_srgb.shape[:2]
    if max(h, w) > 800:
        sc = 800.0 / max(h, w)
        bg_eval = cv2.resize(bg_roi_srgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        ch_eval = cv2.resize(char_srgb, (int(char_srgb.shape[1] * sc), int(char_srgb.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    else:
        bg_eval = bg_roi_srgb
        ch_eval = char_srgb

    bg_gray = cv2.cvtColor(np.clip(bg_eval, 0.0, 1.0), cv2.COLOR_BGR2GRAY)
    ch_gray = cv2.cvtColor(np.clip(ch_eval, 0.0, 1.0), cv2.COLOR_BGR2GRAY)

    if blur_match_strength > 0.01:
        bg_lap = float(cv2.Laplacian(bg_gray, cv2.CV_32F).var())
        ch_lap = float(cv2.Laplacian(ch_gray, cv2.CV_32F).var())
        if ch_lap > bg_lap * 1.6 and ch_lap > 1e-5:
            ratio = np.clip((ch_lap / (bg_lap + 1e-5)) - 1.0, 0.0, 4.5)
            sigma = np.clip(ratio * 0.11 * blur_match_strength, 0.0, 0.55)
            if sigma > 0.22:
                out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if noise_match_strength > 0.01:
        gx = cv2.Sobel(bg_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bg_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)
        flat_thresh = np.percentile(grad_mag, 25)
        flat_mask = grad_mag <= flat_thresh

        if np.any(flat_mask):
            smooth_bg = cv2.GaussianBlur(bg_gray, (5, 5), 1.0)
            high_freq = np.abs(bg_gray - smooth_bg)[flat_mask]
            noise_std = float(np.median(high_freq) * 1.4826)
            noise_std = np.clip(noise_std * noise_match_strength, 0.0, 0.018)
            if noise_std > 0.0035:
                rng = np.random.default_rng(42)
                grain = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
                out = np.clip(out + grain * char_alpha[:, :, None], 0.0, 1.0)

    return out