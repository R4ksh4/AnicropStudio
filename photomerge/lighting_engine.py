import cv2
import numpy as np


def estimate_light_direction_and_color(
    bg_srgb: np.ndarray, char_center: tuple[float, float]
) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(np.clip(bg_srgb, 0.0, 1.0), cv2.COLOR_BGR2GRAY)
    thresh = np.percentile(gray, 96)
    ys, xs = np.where(gray >= thresh)
    if len(xs) == 0:
        return -60.0, np.array([1.0, 0.95, 0.9], dtype=np.float32)

    lx = float(np.mean(xs))
    ly = float(np.mean(ys))
    dx = lx - char_center[0]
    dy = ly - char_center[1]
    az_deg = -60.0 if (abs(dx) < 1e-3 and abs(dy) < 1e-3) else float(np.degrees(np.arctan2(dy, dx)))

    bright_pixels = bg_srgb[ys, xs]
    mean_col = np.mean(bright_pixels, axis=0)
    max_c = float(np.max(mean_col)) + 1e-5
    light_color_srgb = np.clip(mean_col / max_c, 0.35, 1.0)
    light_color_linear = np.power(light_color_srgb, 2.2).astype(np.float32)
    return az_deg, light_color_linear


def azimuth_to_vec3(azimuth_deg: float, elevation_deg: float = 35.0) -> tuple[float, float, float]:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cos_el = np.cos(el)
    return float(np.cos(az) * cos_el), float(np.sin(az) * cos_el), float(np.sin(el))


def apply_lighting_and_rim(
    char_linear: np.ndarray,
    char_alpha: np.ndarray,
    bg_patch_linear: np.ndarray,
    light_vec3: tuple[float, float, float],
    light_color_linear: np.ndarray,
    shading_25d_strength: float = 0.45,
    rim_strength: float = 0.90,
    rim_width: int = 7,
    rim_erase_sharpness: float = 1.6,
    light_wrap_strength: float = 0.55,
    edge_mode: str = "anime",
    rim_texture_weight: float = 0.65,
    rim_falloff: float = 0.50,
    hair_transmission: float = 0.55,
    normal_detail_mix: float = 0.55,
    rim_erase_map: np.ndarray | None = None,
) -> np.ndarray:
    lx, ly, lz = light_vec3
    len_2d = np.hypot(lx, ly) + 1e-6
    lx2, ly2 = lx / len_2d, ly / len_2d

    h, w = char_linear.shape[:2]
    res = char_linear.copy()
    is_real = (edge_mode == "real")

    # 1. 2.5D 立体明暗（真人模式融合原图衣褶/五官灰度凹凸法线 + 高光保护）
    if shading_25d_strength > 0.01:
        sc_25 = min(1.0, 512.0 / max(h, w))
        sw, sh = max(32, int(w * sc_25)), max(32, int(h * sc_25))
        alpha_sm = cv2.resize(char_alpha, (sw, sh), interpolation=cv2.INTER_AREA) if sc_25 < 1.0 else char_alpha
        mask_u8 = (alpha_sm > 0.35).astype(np.uint8)

        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
        max_d = float(dist.max()) + 1e-5
        dome = np.sqrt(np.clip(dist / max_d, 0.0, 1.0))
        dome = cv2.GaussianBlur(dome, (0, 0), sigmaX=max(3.0, max_d * 0.14))

        gx = cv2.Sobel(dome, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(dome, cv2.CV_32F, 0, 1, ksize=5)

        if is_real and normal_detail_mix > 0.01:
            ch_sm = cv2.resize(char_linear, (sw, sh), interpolation=cv2.INTER_AREA) if sc_25 < 1.0 else char_linear
            lum_sm = np.sqrt(np.clip(0.0722 * ch_sm[:, :, 0] + 0.7152 * ch_sm[:, :, 1] + 0.2126 * ch_sm[:, :, 2], 0.0, 1.0))
            lum_sm = cv2.GaussianBlur(lum_sm, (0, 0), sigmaX=1.5)
            dgx = cv2.Sobel(lum_sm, cv2.CV_32F, 1, 0, ksize=3) * (normal_detail_mix * 1.4)
            dgy = cv2.Sobel(lum_sm, cv2.CV_32F, 0, 1, ksize=3) * (normal_detail_mix * 1.4)
            gx = gx + dgx * (alpha_sm > 0.5)
            gy = gy + dgy * (alpha_sm > 0.5)

        nz = np.full_like(gx, 0.55)
        norm_mag = np.sqrt(gx * gx + gy * gy + nz * nz) + 1e-6
        nx, ny, nz = -gx / norm_mag, -gy / norm_mag, nz / norm_mag

        ndotl = nx * lx + ny * ly + nz * lz
        if is_real:
            shade_sm = np.clip(1.0 + shading_25d_strength * np.clip(ndotl - 0.15, -1.0, 0.35), 0.42, 1.20)
        else:
            shade_sm = np.clip(1.0 + shading_25d_strength * np.clip(ndotl - 0.25, -1.25, 0.45), 0.32, 1.30)

        shade = cv2.resize(shade_sm, (w, h), interpolation=cv2.INTER_LINEAR) if sc_25 < 1.0 else shade_sm
        if is_real:
            char_lum = 0.0722 * res[:, :, 0] + 0.7152 * res[:, :, 1] + 0.2126 * res[:, :, 2]
            hi_protect = np.clip(1.20 - char_lum, 0.25, 1.0)
            shade = 1.0 + (shade - 1.0) * hi_protect
        res = res * shade[:, :, None]

    # 2. 计算轮廓边缘法线方向权重
    alpha_blur = cv2.GaussianBlur(char_alpha, (0, 0), sigmaX=3.5)
    agx = cv2.Sobel(alpha_blur, cv2.CV_32F, 1, 0, ksize=5)
    agy = cv2.Sobel(alpha_blur, cv2.CV_32F, 0, 1, ksize=5)
    amag = np.hypot(agx, agy) + 1e-6
    enx, eny = -agx / amag, -agy / amag

    dot = enx * lx2 + eny * ly2
    dir_weight = np.clip((dot + 0.15) / 1.15, 0.0, 1.0) ** rim_erase_sharpness

    if rim_erase_map is not None:
        dir_weight = dir_weight * np.clip(1.0 - rim_erase_map, 0.0, 1.0)

    # 3. 轮廓光分支：二次元赛璐璐硬边 vs 真人双层菲涅尔物理轮廓光
    if rim_strength > 0.01:
        if not is_real:
            k_size = max(3, int(rim_width) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            eroded = cv2.erode(char_alpha, kernel, iterations=1)
            rim_mask = np.clip(char_alpha - eroded, 0.0, 1.0)
            rim_mask = cv2.GaussianBlur(rim_mask, (0, 0), sigmaX=max(1.0, rim_width * 0.4))
            rim_mask *= char_alpha

            rim_energy = (rim_mask * dir_weight)[:, :, None] * (rim_strength * 1.45)
            rim_glow = light_color_linear[None, None, :] * rim_energy
            res = res + rim_glow * np.maximum(0.0, 1.0 - res * 0.4)
        else:
            # 真人双层菲涅尔边：极细 1~2px 核心掠射高光 + 柔和向内衰减晕
            k_core = max(3, int(max(1, round(rim_width * 0.28))) * 2 + 1)
            ker_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core))
            core_rim = np.clip(char_alpha - cv2.erode(char_alpha, ker_core, iterations=1), 0.0, 1.0)
            core_rim = cv2.GaussianBlur(core_rim, (0, 0), sigmaX=max(0.6, rim_width * 0.14))

            k_soft = max(5, int(max(2, round(rim_width * (0.75 + rim_falloff * 0.9)))) * 2 + 1)
            ker_soft = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_soft, k_soft))
            soft_rim = np.clip(char_alpha - cv2.erode(char_alpha, ker_soft, iterations=1), 0.0, 1.0)
            soft_rim = cv2.GaussianBlur(soft_rim, (0, 0), sigmaX=max(1.8, rim_width * (0.45 + rim_falloff * 1.1)))

            combined_rim = np.clip(core_rim * 0.72 + soft_rim * 0.38, 0.0, 1.0) * char_alpha

            # 纹理与明度调制：亮色发丝/皮肤/高光沿吃光强，纯黑哑光鞋子/深暗牛仔褶皱自动收敛，消除荧光贴纸边
            fg_lum = np.clip(0.0722 * res[:, :, 0] + 0.7152 * res[:, :, 1] + 0.2126 * res[:, :, 2], 0.0, 1.0)
            hf = np.abs(fg_lum - cv2.GaussianBlur(fg_lum, (5, 5), 0))
            tex_mod = np.clip(0.22 + 0.90 * np.sqrt(fg_lum) + 2.4 * hf, 0.18, 1.30)
            mod_factor = (1.0 - rim_texture_weight) + rim_texture_weight * tex_mod

            rim_energy = (combined_rim * dir_weight * mod_factor)[:, :, None] * (rim_strength * 1.15)
            rim_glow = light_color_linear[None, None, :] * rim_energy
            res = res + rim_glow * np.maximum(0.0, 1.0 - res * 0.52)

    # 4. 背景环境溢色 (Light Wrap) + 真人发丝物理透光
    if light_wrap_strength > 0.01 or (is_real and hair_transmission > 0.01):
        w_size = max(5, int(rim_width * 3) * 2 + 1)
        w_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w_size, w_size))
        w_eroded = cv2.erode(char_alpha, w_kernel, iterations=1)
        wrap_mask = np.clip(char_alpha - w_eroded, 0.0, 1.0)
        wrap_mask = cv2.GaussianBlur(wrap_mask, (0, 0), sigmaX=max(3.0, rim_width * (1.5 if is_real else 1.2)))
        wrap_mask *= char_alpha

        sc_w = min(1.0, 480.0 / max(h, w))
        if sc_w < 1.0:
            bg_sm = cv2.resize(bg_patch_linear, (max(16, int(w * sc_w)), max(16, int(h * sc_w))), interpolation=cv2.INTER_AREA)
            bg_amb_sm = cv2.GaussianBlur(bg_sm, (0, 0), sigmaX=max(3.0, rim_width * 2.5 * sc_w))
            bg_ambient = cv2.resize(bg_amb_sm, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bg_ambient = cv2.GaussianBlur(bg_patch_linear, (0, 0), sigmaX=max(6.0, rim_width * 2.5))

        if rim_erase_map is not None:
            wrap_mask = wrap_mask * np.clip(1.0 - rim_erase_map * 0.85, 0.0, 1.0)

        if light_wrap_strength > 0.01:
            wrap_w = (wrap_mask * (0.35 + 0.65 * dir_weight))[:, :, None] * (light_wrap_strength * 0.85)
            res = res + wrap_w * np.maximum(0.0, bg_ambient * 1.3 - res * 0.3)

        if is_real and hair_transmission > 0.01:
            semi_band = np.clip(1.0 - np.abs(char_alpha - 0.45) / 0.45, 0.0, 1.0) * (char_alpha > 0.05)
            if rim_erase_map is not None:
                semi_band = semi_band * np.clip(1.0 - rim_erase_map, 0.0, 1.0)
            trans_w = semi_band[:, :, None] * (hair_transmission * 0.70)
            res = res + trans_w * np.maximum(0.0, bg_ambient * 1.15 - res * 0.25)

    return np.clip(res, 0.0, 4.0).astype(np.float32)