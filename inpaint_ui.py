# -*- coding: utf-8 -*-
"""
inpaint_ui.py - Anicrop Studio 魔法消除工作台 (已修复 InpaintWorkbenchWidget 缺失 _clear_brush_and_erase)
"""

import os
import time
import math
import random
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QSplitter, QMessageBox, QDialog,
    QSlider, QComboBox, QFileDialog, QStackedWidget, QApplication, QSpinBox, QColorDialog
)
from PySide6.QtCore import Qt, Signal, QUrl, QSettings, QPointF, QRectF, QThread, QTimer
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QPixmap, QImage,
    QPainter, QColor, QPen, QWheelEvent, QMouseEvent, QKeyEvent, QFont, QBrush,
    QConicalGradient, QLinearGradient, QPainterPath
)

from album import get_app_cache_root
from portrait_matting import PortraitMatting

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import (
        QWebEngineProfile, QWebEnginePage, QWebEngineDownloadRequest
    )
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _extract_metadata_segments(src_path: str) -> list[bytes]:
    """从底图源文件物理提取 APP1~APP15 (EXIF/XMP/ICC) 与 COM 等全部元数据段"""
    segments = []
    if not src_path or not os.path.exists(src_path):
        return segments

    raw_data = None
    try:
        ext = Path(src_path).suffix.lower()
        # 若底图源自 RAW，定点抽调内嵌 JPEG 预览的元数据段
        if ext in {".cr3", ".cr2", ".arw", ".nef", ".dng", ".orf", ".rw2"}:
            from album import extract_raw_embedded_jpeg
            raw_data = extract_raw_embedded_jpeg(src_path)
        else:
            with open(src_path, "rb") as f:
                raw_data = f.read(2 * 1024 * 1024)  # 读取头部 2MB 足够吃下全部元数据
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
        # 遇到图像压缩数据与表标记停止扫描
        if marker in (0xD9, 0xDA, 0xC0, 0xC1, 0xC2, 0xC4, 0xDB):
            break
        length = int.from_bytes(raw_data[idx + 2 : idx + 4], "big")
        seg_end = idx + 2 + length
        if seg_end > n:
            break
        # 精准保留 APP1 (Exif/XMP), APP2 (ICC), APP13 (IPTC), COM 等全部有效段，避开 APP0 防编码冲突
        if 0xE1 <= marker <= 0xEF or marker == 0xFE:
            segments.append(raw_data[idx:seg_end])
        idx = seg_end

    return segments


def imwrite_unicode(path: str, img: np.ndarray, quality: int = 95, orig_file_path: str = "") -> bool:
    """满血图像写入引擎：支持 Unicode 路径并无损嫁接注入底图的 EXIF/XMP/ICC 元数据（支持 BGRA 透明 PNG）"""
    try:
        ext = Path(path).suffix.lower()
        if ext in ('.jpg', '.jpeg'):
            # 如果是 4 通道误存为 jpg，自动压成白底 3 通道防崩
            if len(img.shape) == 3 and img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
                alpha = a.astype(np.float32) / 255.0
                bgr = cv2.merge([b, g, r]).astype(np.float32)
                white = np.full_like(bgr, 255.0)
                img = (bgr * alpha[:, :, np.newaxis] + white * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == '.png':
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            params = []
        res, buf = cv2.imencode(ext, img, params)
        if not res:
            return False

        data_bytes = buf.tobytes()

        # 对导出的 JPEG 注入底图完整的 EXIF/XMP/相机/镜头/ICC 广色域元数据
        if ext in ('.jpg', '.jpeg') and orig_file_path:
            meta_segs = _extract_metadata_segments(orig_file_path)
            if meta_segs and len(data_bytes) >= 2 and data_bytes[:2] == b'\xff\xd8':
                insert_pos = 2
                if len(data_bytes) > 4 and data_bytes[2:4] == b'\xff\xe0':
                    app0_len = int.from_bytes(data_bytes[4:6], 'big')
                    insert_pos = 2 + 2 + app0_len
                meta_payload = b''.join(meta_segs)
                data_bytes = data_bytes[:insert_pos] + meta_payload + data_bytes[insert_pos:]

        with open(path, 'wb') as f:
            f.write(data_bytes)
        return True
    except Exception:
        return False


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def auto_align_patch(orig_bgr: np.ndarray, patch_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    orig_h, orig_w = orig_bgr.shape[:2]
    patch_h, patch_w = patch_bgr.shape[:2]

    target_dim = 1000.0
    s_orig = target_dim / max(orig_w, orig_h)
    s_patch = target_dim / max(patch_w, patch_h)

    small_orig = cv2.resize(orig_bgr, (int(orig_w * s_orig), int(orig_h * s_orig)), interpolation=cv2.INTER_AREA)
    small_patch = cv2.resize(patch_bgr, (int(patch_w * s_patch), int(patch_h * s_patch)), interpolation=cv2.INTER_AREA)

    gray_orig = cv2.cvtColor(small_orig, cv2.COLOR_BGR2GRAY)
    gray_patch = cv2.cvtColor(small_patch, cv2.COLOR_BGR2GRAY)

    # 剪影护栏：如果 AI 图是低饱和度的黑白剪影蒙版，跳过 SIFT 防止特征点不足导致 Homography 扭曲
    hsv_p = cv2.cvtColor(small_patch, cv2.COLOR_BGR2HSV)
    is_low_sat = float(np.mean(hsv_p[:, :, 1])) < 22.0
    is_bimodal = float(np.mean((gray_patch < 55) | (gray_patch > 195))) > 0.70
    if is_low_sat and is_bimodal:
        return cv2.resize(patch_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4), True

    sift = cv2.SIFT_create(nfeatures=800)
    kp_orig, desc_orig = sift.detectAndCompute(gray_orig, None)
    kp_patch, desc_patch = sift.detectAndCompute(gray_patch, None)

    if desc_orig is None or desc_patch is None or len(kp_orig) < 8 or len(kp_patch) < 8:
        return cv2.resize(patch_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4), False

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    matches = flann.knnMatch(desc_patch, desc_orig, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.72 * n.distance]
    if len(good_matches) < 8:
        return cv2.resize(patch_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4), False

    src_pts = np.float32([kp_patch[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) / s_patch
    dst_pts = np.float32([kp_orig[m.trainIdx].pt for m in good_matches]).reshape(-1, 2) / s_orig

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if H is None:
        return cv2.resize(patch_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4), False

    patch_aligned = cv2.warpPerspective(
        patch_bgr, H, (orig_w, orig_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT
    )
    return patch_aligned, True


def fuse_generous_with_modnet(
    orig_bgr: np.ndarray,
    patch_aligned: np.ndarray,
    subject_matte: np.ndarray,
    generous_mask: np.ndarray,
    manual_brush_mask: np.ndarray,
    override_brush_mask: np.ndarray,
    manual_erase_mask: np.ndarray,
    dilate_px: int = 50,
    feather_px: int = 15
) -> tuple[np.ndarray, np.ndarray, dict]:
    t0 = time.time()
    orig_h, orig_w = orig_bgr.shape[:2]

    total_inpaint_zone = cv2.bitwise_or(generous_mask, manual_brush_mask)

    if cv2.countNonZero(total_inpaint_zone) == 0 and cv2.countNonZero(override_brush_mask) == 0:
        return orig_bgr.copy(), np.zeros((orig_h, orig_w), dtype=np.uint8), {"elapsed_ms": 1, "mask_area_pct": 0.0}

    if dilate_px > 0 and cv2.countNonZero(total_inpaint_zone) > 0:
        kd = dilate_px if dilate_px % 2 != 0 else dilate_px + 1
        total_inpaint_zone = cv2.dilate(total_inpaint_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))

    subject_shield = (subject_matte > 0.35).astype(np.uint8) * 255

    hair_transition = ((subject_matte > 0.20) & (subject_matte < 0.65)).astype(np.uint8) * 255
    if cv2.countNonZero(hair_transition) > 0:
        hsv_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv_orig[:, :, 1]
        dirty_background_hairs = ((sat < 45) & (hair_transition > 0))
        subject_shield[dirty_background_hairs] = 0

    total_inpaint_zone[subject_shield > 0] = 0

    if cv2.countNonZero(override_brush_mask) > 0:
        total_inpaint_zone = cv2.bitwise_or(total_inpaint_zone, override_brush_mask)

    if cv2.countNonZero(manual_erase_mask) > 0:
        total_inpaint_zone[manual_erase_mask > 0] = 0

    nz_cnt = cv2.countNonZero(total_inpaint_zone)
    if nz_cnt == 0:
        return orig_bgr.copy(), total_inpaint_zone, {"elapsed_ms": 1, "mask_area_pct": 0.0}

    bx, by, bw, bh = cv2.boundingRect(total_inpaint_zone)
    pad = feather_px * 2 + 20
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(orig_w, bx + bw + pad), min(orig_h, by + bh + pad)

    sub_orig = orig_bgr[y1:y2, x1:x2]
    sub_patch = patch_aligned[y1:y2, x1:x2]
    sub_mask = total_inpaint_zone[y1:y2, x1:x2]
    sub_shield = subject_shield[y1:y2, x1:x2]

    kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    sub_ring = cv2.subtract(cv2.dilate(sub_mask, kernel_ring), sub_mask)
    sub_ring[sub_shield > 0] = 0

    sub_patch_matched = sub_patch.copy()
    if cv2.countNonZero(sub_ring) > 40:
        m_orig = cv2.mean(sub_orig, mask=sub_ring)[:3]
        m_patch = cv2.mean(sub_patch, mask=sub_ring)[:3]
        gains = [np.clip(m_orig[c] / (m_patch[c] + 1e-4), 0.88, 1.20) for c in range(3)]
        gain_mat = np.array(gains, dtype=np.float32).reshape(1, 1, 3)
        sub_patch_matched = np.clip(sub_patch.astype(np.float32) * gain_mat, 0, 255).astype(np.uint8)

    kf = max(3, feather_px * 2 + 1)
    sub_alpha = cv2.GaussianBlur(sub_mask.astype(np.float32) / 255.0, (kf, kf), 0)[:, :, np.newaxis]
    sub_fused = np.clip(sub_orig.astype(np.float32) * (1.0 - sub_alpha) + sub_patch_matched.astype(np.float32) * sub_alpha, 0, 255).astype(np.uint8)

    fused_bgr = orig_bgr.copy()
    fused_bgr[y1:y2, x1:x2] = sub_fused

    elapsed_ms = int((time.time() - t0) * 1000)
    stats = {
        "elapsed_ms": elapsed_ms,
        "mask_area_pct": (nz_cnt / float(orig_w * orig_h)) * 100.0
    }
    return fused_bgr, total_inpaint_zone, stats

def despill_green(bgr_img: np.ndarray) -> np.ndarray:
    """达芬奇级消绿溢算子：彻底中和发丝边缘沾染的 AI 荧光绿，洗回自然发色"""
    out = bgr_img.copy()
    b = out[:, :, 0].astype(np.float32)
    g = out[:, :, 1].astype(np.float32)
    r = out[:, :, 2].astype(np.float32)

    # 绿光上限绝不允许超过红蓝均值，溢出部分强行中和
    rb_avg = (r + b) * 0.5
    spill = g - rb_avg
    mask_spill = spill > 0

    g_corrected = np.where(mask_spill, rb_avg, g)
    out[:, :, 1] = np.clip(g_corrected, 0, 255).astype(np.uint8)
    return out

def _guided_filter_edge(orig_bgr: np.ndarray, alpha_u8: np.ndarray, r: int = 4, eps: float = 0.002) -> np.ndarray:
    """纯 OpenCV 高速导向滤波：把蒙版边缘自动吸附到原图真实物理轮廓上，消灭错位与锯齿"""
    guide = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p = alpha_u8.astype(np.float32) / 255.0
    k = (2 * r + 1, 2 * r + 1)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
    mean_ip = cv2.boxFilter(guide * p, cv2.CV_32F, k)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, k)
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, k)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k)

    q = mean_a * guide + mean_b
    return np.clip(q * 255.0, 0, 255).astype(np.uint8)


def extract_ai_cutout_alpha(patch_img: np.ndarray, subject_matte: np.ndarray, orig_bgr: np.ndarray = None) -> np.ndarray:
    """
    AI 纯色底蒙版提取器（外围连通域 + 封闭发缝平坦度探测 + 纯净边缘导向内收）
    """
    h, w = patch_img.shape[:2]

    if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
        raw_alpha = patch_img[:, :, 3]
        if cv2.countNonZero(raw_alpha < 240) > (h * w * 0.05):
            return raw_alpha

    bgr = patch_img[:, :, :3]

    # 0. 黑白剪影蒙版专线（支持带线稿防幻觉：只留外轮廓，内部线条全填实，外部杂线全清！）
    gray_mat = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv_mat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if float(np.mean(hsv_mat[:, :, 1])) < 22.0 and float(np.mean((gray_mat < 55) | (gray_mat > 195))) > 0.70:
        edge_pixels = np.concatenate([
            gray_mat[:15, :].ravel(),
            gray_mat[-15:, :].ravel(),
            gray_mat[:, :15].ravel(),
            gray_mat[:, -15:].ravel()
        ])
        bg_lum = float(np.median(edge_pixels))
        matte = (255.0 - gray_mat.astype(np.float32)) if bg_lum >= 128.0 else gray_mat.astype(np.float32)

        # 1) 基础灰度色阶（保留外边缘丝滑抗锯齿）
        alpha_raw = np.clip((matte - 105.0) / (215.0 - 105.0) * 255.0, 0.0, 255.0).astype(np.uint8)
        fg_bin = (alpha_raw >= 128).astype(np.uint8)

        # 2) 清理背景里孤立的散落杂线/噪点（只保留面积足够大的主体块）
        min_fg_area = max(1000, int(h * w * 0.002))
        n_fg, lbl_fg, stats_fg, _ = cv2.connectedComponentsWithStats(fg_bin, connectivity=4)
        fg_clean = np.zeros_like(fg_bin)
        for cid in range(1, n_fg):
            if stats_fg[cid, cv2.CC_STAT_AREA] >= min_fg_area:
                fg_clean[lbl_fg == cid] = 1

        # 3) 只看外边缘，里面的线全部删掉（内部填实，仅保留纯白封闭发缝）
        bg_inv = (1 - fg_clean).astype(np.uint8)
        n_bg, lbl_bg, stats_bg, _ = cv2.connectedComponentsWithStats(bg_inv, connectivity=4)
        border_bg = set(np.unique(np.concatenate([
            lbl_bg[0, :], lbl_bg[-1, :], lbl_bg[:, 0], lbl_bg[:, -1]
        ])).tolist())
        true_bg = np.isin(lbl_bg, list(border_bg - {0}))

        # 内部封闭空洞：只有面积>=250且极度纯白(matte均值<35)的才是真发缝，其余内部线条/五官/褶皱全填实！
        for cid in range(1, n_bg):
            if cid in border_bg:
                continue
            if stats_bg[cid, cv2.CC_STAT_AREA] >= 250:
                bx, by, bw, bh = stats_bg[cid, :4]
                sub_m = (lbl_bg[by:by + bh, bx:bx + bw] == cid)
                if float(np.mean(matte[by:by + bh, bx:bx + bw][sub_m])) < 35.0:
                    true_bg[by:by + bh, bx:bx + bw][sub_m] = True

        fg_solid = np.where(true_bg, 0, 255).astype(np.uint8)

        # 4) 仅在最外沿 2px 轮廓带继承平滑抗锯齿，内部 100% 实心、外部 0% 透明
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rim_band = (cv2.dilate(fg_solid, k5) > 0) & (cv2.erode(fg_solid, k5) == 0)
        alpha_final = fg_solid.copy()
        alpha_final[rim_band] = alpha_raw[rim_band]
        return alpha_final

    # 1. 四周边缘采样锁定背景基准色
    samples = np.concatenate([
        bgr[:15, :].reshape(-1, 3),
        bgr[-15:, :].reshape(-1, 3),
        bgr[:, :15].reshape(-1, 3),
        bgr[:, -15:].reshape(-1, 3)
    ], axis=0)
    bg_color = np.median(samples, axis=0).astype(np.float32)

    # 2. 计算色差
    color_diff = np.linalg.norm(bgr.astype(np.float32) - bg_color, axis=2)

    # 3. 外围连通域漫延：抓取与四边连通的所有外部背景
    bg_cand = (color_diff < 24).astype(np.uint8)
    _, labels = cv2.connectedComponents(bg_cand, connectivity=4)
    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))
    border_labels = border_labels[border_labels > 0]
    true_bg = np.isin(labels, border_labels).astype(np.uint8) * 255

    '''# 4. 专杀封闭发缝：在内部未被连通的区域里，探测“极贴近纯灰底且画面平坦无笔触”的封闭发缝
    gray_p = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad_abs = np.abs(cv2.Laplacian(gray_p, cv2.CV_16S))
    strict_inner = ((color_diff < 12) & (true_bg == 0)).astype(np.uint8)
    n_int, lbl_int, stats_int, _ = cv2.connectedComponentsWithStats(strict_inner, connectivity=4)
    for cid in range(1, n_int):
        area = stats_int[cid, cv2.CC_STAT_AREA]
        if area >= 45:
            x, y, bw, bh = stats_int[cid, :4]
            sub_m = (lbl_int[y:y + bh, x:x + bw] == cid)
            mean_diff = float(np.mean(color_diff[y:y + bh, x:x + bw][sub_m]))
            mean_grad = float(np.mean(grad_abs[y:y + bh, x:x + bw][sub_m]))
            if mean_diff < 9.5 and mean_grad < 6.0:
                true_bg[y:y + bh, x:x + bw][sub_m] = 255'''

    # 5. 背景外沿 2px 过渡带去灰晕收口
    rim_dilate = cv2.dilate(true_bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    rim_zone = (rim_dilate > 0) & (true_bg == 0)
    true_bg[rim_zone & (color_diff < 36)] = 255

    alpha = np.where(true_bg > 0, 0, 255).astype(np.uint8)

    # 6. 平滑去阶梯 + 用纯净的 AI 灰底图(bgr)做导向吸附，不被原图粉紫天空光带偏！
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = _guided_filter_edge(bgr, alpha, r=3, eps=0.002)

    # 7. 边缘向内微收 1.5px (中点 160)，吃死原图和 AI 图那 1~2px 对齐错位杂边
    alpha_f = (alpha.astype(np.float32) - 160.0) * 2.6 + 128.0
    return np.clip(alpha_f, 0, 255).astype(np.uint8)


def fuse_cutout_with_modnet(
    src_bgr: np.ndarray,
    base_alpha: np.ndarray,
    manual_brush_mask: np.ndarray,
    override_brush_mask: np.ndarray,
    manual_erase_mask: np.ndarray,
    feather_px: int = 15
) -> tuple[np.ndarray, np.ndarray, dict]:
    """立绘平滑抗锯齿合成：内部 100% 实心，外沿享受平滑微羽化，彻底告别台阶锯齿"""
    t0 = time.time()
    orig_h, orig_w = src_bgr.shape[:2]

    cur_alpha = base_alpha.astype(np.float32)

    # 1. 手动涂抹区做 3px 软边缘平滑，涂抹擦除自然不留硬疤
    if manual_brush_mask is not None and cv2.countNonZero(manual_brush_mask) > 0:
        b_soft = cv2.GaussianBlur(manual_brush_mask.astype(np.float32) / 255.0, (5, 5), 0)
        cur_alpha = cur_alpha * (1.0 - b_soft)

    if override_brush_mask is not None and cv2.countNonZero(override_brush_mask) > 0:
        o_soft = cv2.GaussianBlur(override_brush_mask.astype(np.float32) / 255.0, (5, 5), 0)
        cur_alpha = cur_alpha * (1.0 - o_soft)

    # 2. 橡皮擦恢复同样带平滑过渡
    if manual_erase_mask is not None and cv2.countNonZero(manual_erase_mask) > 0:
        e_soft = cv2.GaussianBlur(manual_erase_mask.astype(np.float32), (5, 5), 0)
        cur_alpha = np.maximum(cur_alpha, e_soft)

    # 3. 联动左侧【边缘羽化】滑块，换算为 0.5px ~ 1.5px 的精细抗锯齿羽化（不外溢露底）
    sigma = max(0.5, min(1.5, feather_px * 0.05))
    cur_alpha = cv2.GaussianBlur(cur_alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 轻对比度锁定：保住内部 255 实心，外沿顺滑过渡
    cur_alpha = np.clip((cur_alpha - 132.0) * 1.45 + 128.0, 0, 255).astype(np.uint8)

    # 4. 合并为满血原图 BGRA
    b, g, r = cv2.split(src_bgr)
    bgra_fused = cv2.merge([b, g, r, cur_alpha])

    mask_vis = 255 - cur_alpha
    elapsed_ms = int((time.time() - t0) * 1000)
    stats = {
        "elapsed_ms": elapsed_ms,
        "mask_area_pct": (cv2.countNonZero(mask_vis) / float(orig_h * orig_w)) * 100.0
    }
    return bgra_fused, mask_vis, stats


class InpaintInitWorker(QThread):
    sig_status = Signal(str, int)
    sig_finished = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, orig_bgr: np.ndarray, patch_bgr: np.ndarray, session_mode: str = "inpaint"):
        super().__init__()
        self.orig_bgr = orig_bgr
        self.patch_bgr = patch_bgr
        self.session_mode = session_mode

    def run(self):
        try:
            self.sig_status.emit("🎯 正在执行 SIFT + RANSAC 亚像素图层锁死配准...", 15)
            patch_aligned, align_success = auto_align_patch(self.orig_bgr, self.patch_bgr)

            self.sig_status.emit("🤖 正在启动 MODNet 神经网络，构建模特发丝指尖结界...", 45)
            subject_matte = PortraitMatting.extract_subject_matte(self.orig_bgr, ref_size=512)

            orig_h, orig_w = self.orig_bgr.shape[:2]
            candidate_blobs = []

            if self.session_mode == "cutout":
                self.sig_status.emit("🤖 正在让神经网络对 AI 纯净图进行平滑流线建模...", 60)
                # 核心升级：直接在无路人干扰的 AI 图上提取发丝与人体流线，彻底杜绝锯齿撕裂！
                cutout_matte = PortraitMatting.extract_subject_matte(patch_aligned, ref_size=512)
                
                self.sig_status.emit("✂️ 正在执行双核平滑剥离...", 85)
                base_alpha = extract_ai_cutout_alpha(patch_aligned, cutout_matte, orig_bgr=self.orig_bgr)
                
                self.sig_status.emit("✨ 抠图完成，正在载入纯净画布...", 98)
                self.sig_finished.emit({
                    "patch_aligned": patch_aligned,
                    "align_success": align_success,
                    "subject_matte": cutout_matte,
                    "candidate_blobs": [],
                    "base_alpha": base_alpha,
                    "session_mode": "cutout"
                })
                return
            
            else:
                self.sig_status.emit("🔍 正在探测全图路人消除区与影子大包围网...", 75)
                k_b = 21
                diff = cv2.absdiff(
                    cv2.GaussianBlur(self.orig_bgr, (k_b, k_b), 0),
                    cv2.GaussianBlur(patch_aligned, (k_b, k_b), 0)
                )
                diff_max = np.max(diff, axis=2)

                gray = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)
                ceiling_h = int(orig_h * 0.25)
                diff_max[:ceiling_h][gray[:ceiling_h] > 225] = 0

                _, bin_diff = cv2.threshold(diff_max, 13, 255, cv2.THRESH_BINARY)
                kernel_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                fused_blobs = cv2.morphologyEx(bin_diff, cv2.MORPH_CLOSE, kernel_fuse)

                contours, _ = cv2.findContours(fused_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1000:
                        mask_blob = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        cv2.drawContours(mask_blob, [cnt], -1, 255, thickness=-1)
                        overlap = cv2.countNonZero(cv2.bitwise_and(mask_blob, (subject_matte > 0.5).astype(np.uint8) * 255))
                        if overlap < area * 0.75:
                            x, y, w, h = cv2.boundingRect(cnt)
                            candidate_blobs.append({
                                "id": len(candidate_blobs),
                                "mask": mask_blob,
                                "bbox": (x, y, w, h),
                                "area": area
                            })

            candidate_blobs.sort(key=lambda x: x["area"], reverse=True)
            for idx, b in enumerate(candidate_blobs):
                b["id"] = idx

            self.sig_status.emit("✨ 算法准备就绪，正在唤醒工作台...", 98)

            self.sig_finished.emit({
                "patch_aligned": patch_aligned,
                "align_success": align_success,
                "subject_matte": subject_matte,
                "candidate_blobs": candidate_blobs,
                "session_mode": self.session_mode
            })
        except Exception as e:
            self.sig_error.emit(str(e))

    

class InpaintLoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(480, 240)

        self.angle = 0
        self.elapsed_sec = 0.0
        self.status_text = "🪄 正在唤醒魔法消除引擎..."
        self.progress_pct = 5

        self.timer_rot = QTimer(self)
        self.timer_rot.timeout.connect(self._on_tick)
        self.timer_rot.start(16)

    def _on_tick(self):
        self.angle = (self.angle + 4) % 360
        self.elapsed_sec += 0.016
        self.update()

    def set_status(self, text: str, pct: int):
        self.status_text = text
        self.progress_pct = max(self.progress_pct, pct)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setBrush(QColor("#141418"))
        painter.setPen(QPen(QColor("#0284c7"), 1.8))
        painter.drawRoundedRect(rect, 10, 10)

        cx, cy = self.width() / 2.0, 70.0
        radius = 36.0

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.angle)

        grad = QConicalGradient(0, 0, 0)
        grad.setColorAt(0.0, QColor("#38bdf8"))
        grad.setColorAt(0.5, QColor("#0284c7"))
        grad.setColorAt(1.0, QColor(2, 132, 199, 0))

        painter.setPen(QPen(QBrush(grad), 4.5, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(-radius, -radius, radius * 2, radius * 2), 0, 360 * 16)
        painter.restore()

        painter.setBrush(QColor("#38bdf8"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.setPen(QColor("#f4f4f5"))
        painter.drawText(QRectF(16, 122, self.width() - 32, 26), Qt.AlignCenter, "⚡ 正在执行 AI 魔法消除与缝合计算")

        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QRectF(20, 150, self.width() - 40, 22), Qt.AlignCenter, self.status_text)

        bar_x, bar_y, bar_w, bar_h = 40, 182, self.width() - 80, 8
        painter.setBrush(QColor("#27272a"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 4, 4)

        fill_w = (bar_w * self.progress_pct) / 100.0
        painter.setBrush(QColor("#0284c7"))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 4, 4)

        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor("#a1a1aa"))
        painter.drawText(QRectF(40, 198, self.width() - 80, 20), Qt.AlignCenter, f"计算进行中: {self.elapsed_sec:.1f} 秒 | 满血 2400W 尺度处理")


class DissolveParticle:
    def __init__(self, pos: QPointF):
        self.pos = QPointF(pos)
        self.vx = random.uniform(2.5, 7.5)
        self.vy = random.uniform(-2.0, 2.0)
        self.size = random.uniform(2.0, 5.0)
        self.alpha = 255
        self.decay = random.uniform(5.0, 9.0)
        self.color = random.choice([
            QColor(255, 255, 255),
            QColor(254, 240, 138),
            QColor(186, 230, 253),
            QColor(244, 180, 0)
        ])

    def update(self) -> bool:
        self.pos.setX(self.pos.x() + self.vx)
        self.pos.setY(self.pos.y() + self.vy)
        self.alpha = max(0, self.alpha - self.decay)
        return self.alpha > 0


class AppleGlassDock(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        r = rect.height() / 2.0

        for i in range(4, 0, -1):
            spread = i * 2.0
            alpha = int(32 * (1.0 - i / 5.0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, alpha))
            shadow_r = QRectF(rect).adjusted(-spread * 0.5, spread * 0.4, spread * 0.5, spread * 1.0)
            painter.drawRoundedRect(shadow_r, r + spread * 0.2, r + spread * 0.2)

        grad = QLinearGradient(rect.left(), rect.top(), rect.left(), rect.bottom())
        grad.setColorAt(0.0, QColor(36, 36, 48, 225))
        grad.setColorAt(1.0, QColor(18, 18, 26, 240))
        painter.setBrush(QBrush(grad))

        rim_pen = QPen(QColor(255, 255, 255, 48), 1.0)
        painter.setPen(rim_pen)
        painter.drawRoundedRect(QRectF(rect), r, r)

        highlight_rect = QRectF(rect).adjusted(r, 1, -r, 0)
        highlight_grad = QLinearGradient(highlight_rect.left(), highlight_rect.top(), highlight_rect.right(), highlight_rect.top())
        highlight_grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        highlight_grad.setColorAt(0.5, QColor(255, 255, 255, 75))
        highlight_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(QPen(QBrush(highlight_grad), 1.0))
        painter.drawLine(QPointF(highlight_rect.left(), rect.top() + 1.5), QPointF(highlight_rect.right(), rect.top() + 1.5))


# ==============================================================================
# 魔法消除验货工作台组件 (【内嵌页面版】：永久保留主窗口左侧栏与小球联动)
# ==============================================================================
class InpaintWorkbenchWidget(QWidget):
    sig_back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_bgr = None
        self.patch_aligned = None
        self.subject_matte = None
        self.candidate_blobs = []
        self.orig_file_path = ""

        self.selected_blob_ids = set()
        self.hovered_blob_id = -1

        self.manual_brush_mask = None
        self.override_brush_mask = None
        self.manual_erase_mask = None

        self.history_undo = []
        self.history_redo = []
        self.MAX_HISTORY = 16

        self.current_stroke_mask = None
        self.current_stroke_path = QPainterPath()
        self.last_img_pt = None

        self.lasso_pts_img = []
        self.lasso_screen_path = QPainterPath()
        self.is_lasso_active = False

        self.is_processing_mask = False
        self.spinner_angle = 0

        self.current_tool = "brush"
        self.shield_override = False
        self.is_brushing = False
        self.brush_size = 45

        self.mouse_view_pos = QPointF(-100, -100)
        self.particles = []

        self.dilate_px = 55
        self.feather_px = 15

        self.view_mode = "fused"
        self.prev_mode_before_hold = "fused"

        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.is_panning = False
        self.is_space_held = False
        self.is_alt_held = False
        self.last_mouse = QPointF()

        self.pixmap_orig = QPixmap()
        self.pixmap_patch = QPixmap()
        self.pixmap_subject = QPixmap()
        self.pixmap_fused = QPixmap()
        self.pixmap_mask = QPixmap()
        self._custom_bg_pixmap = QPixmap()
        

        self.session_mode = "inpaint"      # "inpaint" (魔法消除) 或 "cutout" (一键抠图)
        self.bg_preview_mode = "checker"   # "checker", "white", "black", "green", "gray"
        self.bgra_fused = None
        self._checker_pixmap = self._generate_checker_pattern(16)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start(16)
        
        self._init_ui()

    def load_session_data(self, orig_bgr: np.ndarray, patch_aligned: np.ndarray,
                          subject_matte: np.ndarray, candidate_blobs: list,
                          orig_file_path: str, dilate_px: int, feather_px: int,
                          session_mode: str = "inpaint"):
        self.session_mode = session_mode
        self.base_alpha = None
        self.orig_bgr = orig_bgr
        self.patch_aligned = patch_aligned
        self.subject_matte = subject_matte
        self.candidate_blobs = candidate_blobs
        self.orig_file_path = orig_file_path
        self.dilate_px = dilate_px
        self.feather_px = feather_px

        is_cutout = (self.session_mode == "cutout")
        self.combo_bg_preview.setVisible(is_cutout)
        self.lbl_bg_preview.setVisible(is_cutout)
        self.btn_fused.setText("✂️ 扣出成片 [1]" if is_cutout else "✨ 快速消除 [1]")
        self.btn_orig.setText("📷 原始原片 [2]" if is_cutout else "📷 消除前 [2]")
        self.btn_mask.setText("🖤 边缘检查 [3]" if is_cutout else "🖤 精修模式 [3]")
        self.btn_patch.setVisible(True)
        self.btn_subject.setVisible(not is_cutout)

        orig_h, orig_w = orig_bgr.shape[:2]
        self.selected_blob_ids = {b["id"] for b in self.candidate_blobs}
        self.hovered_blob_id = -1

        self.manual_brush_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        self.override_brush_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        self.manual_erase_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        self.current_stroke_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        self.history_undo.clear()
        self.history_redo.clear()

        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.view_mode = "fused"

        self.pixmap_orig = QPixmap.fromImage(bgr_to_qimage(self.orig_bgr))
        self.pixmap_patch = QPixmap.fromImage(bgr_to_qimage(self.patch_aligned))
        subject_vis = (self.subject_matte * 255.0).astype(np.uint8)
        self.pixmap_subject = QPixmap.fromImage(QImage(subject_vis.data, orig_w, orig_h, orig_w, QImage.Format_Grayscale8).copy())

        if hasattr(self, "_incoming_base_alpha") and self._incoming_base_alpha is not None:
            self.base_alpha = self._incoming_base_alpha

        self._recompute_fusion()
        self._update_undo_redo_ui()
        self._refresh_stats_label()
        self._update_btn_styles()
        self.viewport.update()

    def _push_undo_snapshot(self):
        if self.manual_brush_mask is None:
            return
        snapshot = {
            "brush_mask": self.manual_brush_mask.copy(),
            "override_mask": self.override_brush_mask.copy(),
            "erase_mask": self.manual_erase_mask.copy(),
            "selected_ids": set(self.selected_blob_ids)
        }
        self.history_undo.append(snapshot)
        if len(self.history_undo) > self.MAX_HISTORY:
            self.history_undo.pop(0)
        self.history_redo.clear()
        self._update_undo_redo_ui()

    def _undo(self):
        if not self.history_undo:
            return
        current_state = {
            "brush_mask": self.manual_brush_mask.copy(),
            "override_mask": self.override_brush_mask.copy(),
            "erase_mask": self.manual_erase_mask.copy(),
            "selected_ids": set(self.selected_blob_ids)
        }
        self.history_redo.append(current_state)
        prev = self.history_undo.pop()
        self.manual_brush_mask = prev["brush_mask"]
        self.override_brush_mask = prev["override_mask"]
        self.manual_erase_mask = prev["erase_mask"]
        self.selected_blob_ids = prev["selected_ids"]
        self._recompute_fusion()
        self._refresh_stats_label()
        self._update_undo_redo_ui()
        self.viewport.update()

    def _redo(self):
        if not self.history_redo:
            return
        current_state = {
            "brush_mask": self.manual_brush_mask.copy(),
            "override_mask": self.override_brush_mask.copy(),
            "erase_mask": self.manual_erase_mask.copy(),
            "selected_ids": set(self.selected_blob_ids)
        }
        self.history_undo.append(current_state)
        nxt = self.history_redo.pop()
        self.manual_brush_mask = nxt["brush_mask"]
        self.override_brush_mask = nxt["override_mask"]
        self.manual_erase_mask = nxt["erase_mask"]
        self.selected_blob_ids = nxt["selected_ids"]
        self._recompute_fusion()
        self._refresh_stats_label()
        self._update_undo_redo_ui()
        self.viewport.update()

    def _update_undo_redo_ui(self):
        can_undo = len(self.history_undo) > 0
        can_redo = len(self.history_redo) > 0
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)
        self.btn_undo.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 6px 16px;" if can_undo else "color: #52525b; padding: 6px 16px;")
        self.btn_redo.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 6px 16px;" if can_redo else "color: #52525b; padding: 6px 16px;")

    def _clear_all_to_clean_canvas(self):
        """【一键清屏·纯手动流】清空全部 AI 自动选区与涂抹笔刷，秒回 100% 纯净底图"""
        if self.orig_bgr is None:
            return
        self._push_undo_snapshot()
        self.selected_blob_ids.clear()
        if self.manual_brush_mask is not None:
            self.manual_brush_mask.fill(0)
        if self.override_brush_mask is not None:
            self.override_brush_mask.fill(0)
        if self.manual_erase_mask is not None:
            self.manual_erase_mask.fill(0)
        self.particles.clear()
        self._recompute_fusion()
        self._refresh_stats_label()
        self._update_undo_redo_ui()
        self.viewport.update()

    def _clear_brush_and_erase(self):
        """【重置】清空手动笔刷，恢复为 AI 自动探测圈选的初始状态"""
        self._push_undo_snapshot()
        if self.manual_brush_mask is not None:
            self.manual_brush_mask.fill(0)
        if self.override_brush_mask is not None:
            self.override_brush_mask.fill(0)
        if self.manual_erase_mask is not None:
            self.manual_erase_mask.fill(0)
        self.selected_blob_ids = {b["id"] for b in self.candidate_blobs}
        self._recompute_fusion()
        self._refresh_stats_label()
        self._update_undo_redo_ui()
        self.viewport.update()

    def _on_anim_tick(self):
        if self.is_processing_mask:
            self.spinner_angle = (self.spinner_angle + 8) % 360
            self.viewport.update()

        if self.particles:
            self.particles = [p for p in self.particles if p.update()]
            self.viewport.update()

        if self.is_brushing or self.is_lasso_active or self.hovered_blob_id != -1:
            self.viewport.update()

    def _spawn_dissolve_particles_only_from_region(self, region_mask: np.ndarray):
        w, h = self.viewport.width(), self.viewport.height()
        if self.orig_bgr is None:
            return
        iw, ih = self.orig_bgr.shape[1], self.orig_bgr.shape[0]
        scale = min(w / float(iw), h / float(ih)) * self.zoom
        dw, dh = iw * scale, ih * scale
        dx = (w - dw) / 2.0 + self.pan.x()
        dy = (h - dh) / 2.0 + self.pan.y()

        y_coords, x_coords = np.where(region_mask > 0)
        if len(y_coords) == 0:
            return

        sample_size = min(220, len(y_coords))
        indices = np.random.choice(len(y_coords), sample_size, replace=False)

        for idx in indices:
            px = dx + x_coords[idx] * scale
            py = dy + y_coords[idx] * scale
            self.particles.append(DissolveParticle(QPointF(px, py)))

    def _get_generous_mask(self) -> np.ndarray:
        if self.orig_bgr is None:
            return np.zeros((10, 10), dtype=np.uint8)
        orig_h, orig_w = self.orig_bgr.shape[:2]
        gen_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for b in self.candidate_blobs:
            if b["id"] in self.selected_blob_ids:
                gen_mask = cv2.bitwise_or(gen_mask, b["mask"])
        return gen_mask

    def _recompute_fusion(self):
        if self.orig_bgr is None:
            return

        # ---------------- 抠图模式专线 ----------------
        if self.session_mode == "cutout":
            if self.base_alpha is None:
                self.base_alpha = (self.subject_matte * 255.0).astype(np.uint8)

            # 核心：从 AI 图提取干净蒙版，把蒙版转嫁回 100% 无损满血原图 orig_bgr！
            target_source = self.orig_bgr

            self.bgra_fused, self.mask, self.stats = fuse_cutout_with_modnet(
                target_source,
                base_alpha=self.base_alpha,
                manual_brush_mask=self.manual_brush_mask,
                override_brush_mask=self.override_brush_mask,
                manual_erase_mask=self.manual_erase_mask,
                feather_px=self.feather_px
            )

            # 转为带 Alpha 的 QImage
            rgba = cv2.cvtColor(self.bgra_fused, cv2.COLOR_BGRA2RGBA)
            h, w = rgba.shape[:2]
            qimg_fused = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            self.pixmap_fused = QPixmap.fromImage(qimg_fused)

            # 边缘检查视图：暗底透视
            vis = target_source.copy()
            if self.mask is not None and cv2.countNonZero(self.mask) > 0:
                vis[self.mask > 0] = (vis[self.mask > 0] * 0.25).astype(np.uint8)
            self.pixmap_mask = QPixmap.fromImage(bgr_to_qimage(vis))
            return

        # ---------------- 魔法消除模式专线 ----------------
        if self.patch_aligned is None:
            return
        gen_mask = self._get_generous_mask()
        self.fused_bgr, self.mask, self.stats = fuse_generous_with_modnet(
            self.orig_bgr, self.patch_aligned,
            subject_matte=self.subject_matte,
            generous_mask=gen_mask,
            manual_brush_mask=self.manual_brush_mask,
            override_brush_mask=self.override_brush_mask,
            manual_erase_mask=self.manual_erase_mask,
            dilate_px=self.dilate_px,
            feather_px=self.feather_px
        )
        self.pixmap_fused = QPixmap.fromImage(bgr_to_qimage(self.fused_bgr))

        vis = self.fused_bgr.copy()
        if cv2.countNonZero(self.mask) > 0:
            vis[self.mask > 0] = (vis[self.mask > 0] * 0.45).astype(np.uint8)
        self.pixmap_mask = QPixmap.fromImage(bgr_to_qimage(vis))

    @staticmethod
    def _generate_checker_pattern(tile_size: int = 16) -> QPixmap:
        pm = QPixmap(tile_size * 2, tile_size * 2)
        p = QPainter(pm)
        c1, c2 = QColor("#222226"), QColor("#18181c")
        p.fillRect(0, 0, tile_size, tile_size, c1)
        p.fillRect(tile_size, 0, tile_size, tile_size, c2)
        p.fillRect(0, tile_size, tile_size, tile_size, c2)
        p.fillRect(tile_size, tile_size, tile_size, tile_size, c1)
        p.end()
        return pm

    def _init_ui(self):
        root_vlayout = QVBoxLayout(self)
        root_vlayout.setContentsMargins(0, 0, 0, 0)
        root_vlayout.setSpacing(0)

        # 1. 顶部模式细轨
        top_bar = QFrame()
        top_bar.setFixedHeight(38)
        top_bar.setStyleSheet("background-color: #121216; border-bottom: 1px solid #27272f; padding: 0 12px;")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(6, 0, 6, 0)
        tb_layout.setSpacing(6)

        self.btn_back = QPushButton("‹ 返回输入")
        self.btn_back.setToolTip("返回底图与网页输入界面")
        self.btn_back.setFocusPolicy(Qt.NoFocus)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                color: #e4e4e7;
                font-weight: bold;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #0284c7;
                color: #ffffff;
            }
        """)
        self.btn_back.clicked.connect(self.sig_back_requested.emit)
        tb_layout.addWidget(self.btn_back)

        tb_layout.addSpacing(4)

        # 视图切换按钮组
        self.btn_fused = QPushButton("✨ 扣出成片 [1]")
        self.btn_orig = QPushButton("📷 原始原片 [2]")
        self.btn_mask = QPushButton("🖤 边缘检查 [3]")
        self.btn_patch = QPushButton("🎯 期望效果图 [4]")
        self.btn_subject = QPushButton("🛡️ 模特保护透视 [5]")

        for b in (self.btn_fused, self.btn_orig, self.btn_mask, self.btn_patch, self.btn_subject):
            b.setFocusPolicy(Qt.NoFocus)
            tb_layout.addWidget(b)

        self.btn_fused.clicked.connect(lambda: self._set_mode("fused"))
        self.btn_orig.clicked.connect(lambda: self._set_mode("orig"))
        self.btn_mask.clicked.connect(lambda: self._set_mode("mask"))
        self.btn_patch.clicked.connect(lambda: self._set_mode("patch"))
        self.btn_subject.clicked.connect(lambda: self._set_mode("subject"))

        tb_layout.addSpacing(6)

        # 核心：背景切换控制坞 (直接挂在按钮旁边，极其显眼，绝不折叠！)
        self.lbl_bg_preview = QLabel("🎨 背景:")
        self.lbl_bg_preview.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        tb_layout.addWidget(self.lbl_bg_preview)

        self.combo_bg_preview = QComboBox()
        self.combo_bg_preview.addItem("🏁 PS灰白方格 (透明底)", "checker")
        self.combo_bg_preview.addItem("⬜ 纯白底色 (验黑边)", "white")
        self.combo_bg_preview.addItem("⬛ 纯黑底色 (验白边)", "black")
        self.combo_bg_preview.addItem("🟩 绿幕影视底", "green")
        self.combo_bg_preview.addItem("🩶 18%中性灰", "gray")
        self.combo_bg_preview.addItem("🌄 导入自定义背景图...", "custom_img")
        self.combo_bg_preview.setFocusPolicy(Qt.NoFocus)
        self.combo_bg_preview.setStyleSheet("""
            QComboBox {
                background-color: #18181e;
                border: 1px solid #0284c7;
                border-radius: 4px;
                padding: 3px 8px;
                color: #38bdf8;
                font-weight: bold;
                font-size: 11px;
                min-width: 140px;
            }
            QComboBox QAbstractItemView {
                background-color: #18181e;
                border: 1px solid #3f3f46;
                selection-background-color: #0284c7;
                color: #f4f4f5;
            }
        """)
        self.combo_bg_preview.currentIndexChanged.connect(self._on_bg_preview_changed)
        tb_layout.addWidget(self.combo_bg_preview)

        

        
        

        

        tb_layout.addStretch()

        self.lbl_stats = QLabel()
        self.lbl_stats.setStyleSheet("color: #38bdf8; font-family: Consolas; font-size: 11px;")
        tb_layout.addWidget(self.lbl_stats)

        root_vlayout.addWidget(top_bar)

        self.viewport = QFrame()
        self.viewport.setMouseTracking(True)
        self.viewport.setStyleSheet("background-color: #050507; border: none;")
        self.viewport.paintEvent = self._viewport_paint
        self.viewport.wheelEvent = self._viewport_wheel
        self.viewport.mousePressEvent = self._viewport_mouse_press
        self.viewport.mouseMoveEvent = self._viewport_mouse_move
        self.viewport.mouseReleaseEvent = self._viewport_mouse_release
        root_vlayout.addWidget(self.viewport, stretch=1)

        self.dock = AppleGlassDock(self.viewport)
        self.dock.setObjectName("AppleGlassDock")
        self.dock.setAttribute(Qt.WA_TranslucentBackground, True)
        self.dock.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 14);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 16px;
                color: #e4e4e7;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 16px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 32);
                border-color: rgba(255, 255, 255, 38);
                color: #ffffff;
            }
            QPushButton:disabled {
                color: #52525b;
                background-color: transparent;
                border-color: transparent;
            }
        """)
        dock_layout = QHBoxLayout(self.dock)
        dock_layout.setContentsMargins(14, 6, 14, 6)
        dock_layout.setSpacing(10)

        self.btn_undo = QPushButton("↩️ 撤销")
        self.btn_undo.setToolTip("撤销上一步操作 (Ctrl+Z)")
        self.btn_undo.setFocusPolicy(Qt.NoFocus)
        self.btn_undo.clicked.connect(self._undo)
        dock_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↪️ 重做")
        self.btn_redo.setToolTip("恢复重做 (Ctrl+Alt+Z / Ctrl+Shift+Z)")
        self.btn_redo.setFocusPolicy(Qt.NoFocus)
        self.btn_redo.clicked.connect(self._redo)
        dock_layout.addWidget(self.btn_redo)

        self.btn_tool_brush = QPushButton("🖌️ 消除涂抹")
        self.btn_tool_eraser = QPushButton("🧼 恢复橡皮擦")
        self.btn_override_toggle = QPushButton("🛡️ 避让主体 [W]")

        for b in (self.btn_tool_brush, self.btn_tool_eraser, self.btn_override_toggle):
            b.setFocusPolicy(Qt.NoFocus)
            dock_layout.addWidget(b)

        self.btn_tool_brush.clicked.connect(lambda: self._set_tool("brush"))
        self.btn_tool_eraser.clicked.connect(lambda: self._set_tool("eraser"))
        self.btn_override_toggle.clicked.connect(self._toggle_override_shield)

        self.btn_clear_screen = QPushButton("🧹 清屏")
        self.btn_clear_screen.setToolTip("一键清空全部消除区域，秒回纯净底图（纯手动党专属）")
        self.btn_clear_screen.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_screen.setStyleSheet("""
            QPushButton {
                background-color: rgba(239, 68, 68, 24);
                border: 1px solid rgba(239, 68, 68, 75);
                color: #f87171;
                border-radius: 16px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 50);
                color: #ffffff;
            }
        """)
        self.btn_clear_screen.clicked.connect(self._clear_all_to_clean_canvas)
        dock_layout.addWidget(self.btn_clear_screen)

        self.btn_clear_all_brush = QPushButton("🔄 重置")
        self.btn_clear_all_brush.setToolTip("恢复为 AI 自动探测圈选的初始状态")
        self.btn_clear_all_brush.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_all_brush.setStyleSheet("""
            QPushButton {
                background-color: rgba(249, 115, 22, 22);
                border: 1px solid rgba(249, 115, 22, 75);
                color: #fb923c;
                border-radius: 16px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: rgba(249, 115, 22, 45);
                color: #ffffff;
            }
        """)
        self.btn_clear_all_brush.clicked.connect(self._clear_brush_and_erase)
        dock_layout.addWidget(self.btn_clear_all_brush)

        btn_save = QPushButton("💾 另存成片")
        btn_save.setFocusPolicy(Qt.NoFocus)
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                color: white;
                border-radius: 16px;
                padding: 6px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        btn_save.clicked.connect(self._save_as)
        dock_layout.addWidget(btn_save)

        self._update_undo_redo_ui()
        self._refresh_stats_label()
        self._update_btn_styles()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        dw = min(1100, max(980, self.viewport.width() - 40))
        dh = 52
        dx = (self.viewport.width() - dw) // 2
        dy = self.viewport.height() - dh - 24
        self.dock.setGeometry(dx, dy, dw, dh)
        self._clamp_pan_offset()

    def _set_mode(self, m: str):
        if self.view_mode == m:
            return
        self.view_mode = m
        self._update_btn_styles()
        self.viewport.update()

    def _set_tool(self, tool: str):
        self.current_tool = tool
        self._update_btn_styles()

    def _toggle_override_shield(self):
        self.shield_override = not self.shield_override
        self._update_btn_styles()

    def _update_btn_styles(self):
        act = "QPushButton { background-color: #0284c7; color: white; font-weight: bold; border-radius: 4px; padding: 4px 10px; }"
        norm = "QPushButton { background-color: #1f1f24; color: #a1a1aa; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 10px; }"
        self.btn_fused.setStyleSheet(act if self.view_mode == "fused" else norm)
        self.btn_orig.setStyleSheet(act if self.view_mode == "orig" else norm)
        self.btn_mask.setStyleSheet(act if self.view_mode == "mask" else norm)
        self.btn_patch.setStyleSheet("QPushButton { background-color: #9333ea; color: white; font-weight: bold; border-radius: 4px; padding: 4px 10px; }" if self.view_mode == "patch" else norm)
        self.btn_subject.setStyleSheet("QPushButton { background-color: #16a34a; color: white; font-weight: bold; border-radius: 4px; padding: 4px 10px; }" if self.view_mode == "subject" else norm)

        act_brush = "background-color: #0284c7; color: white; border: 1px solid #38bdf8; border-radius: 16px; padding: 6px 16px;"
        act_eraser = "background-color: #d97706; color: white; border: 1px solid #fbbf24; border-radius: 16px; padding: 6px 16px;"
        self.btn_tool_brush.setStyleSheet(act_brush if self.current_tool == "brush" else "padding: 6px 16px;")
        self.btn_tool_eraser.setStyleSheet(act_eraser if self.current_tool == "eraser" else "padding: 6px 16px;")
        self.btn_tool_brush.setStyleSheet(act_brush if self.current_tool == "brush" else "padding: 6px 16px;")
        self.btn_tool_eraser.setStyleSheet(act_eraser if self.current_tool == "eraser" else "padding: 6px 16px;")
        
        self.btn_tool_eraser.setStyleSheet(act_eraser if self.current_tool == "eraser" else "padding: 6px 16px;")

        if self.shield_override:
            self.btn_override_toggle.setText("⚔️ 强制破盾 [W]")
            self.btn_override_toggle.setStyleSheet("background-color: #dc2626; color: white; border: 1px solid #f87171; border-radius: 16px; padding: 6px 16px;")
        else:
            self.btn_override_toggle.setText("🛡️ 避让主体 [W]")
            self.btn_override_toggle.setStyleSheet("background-color: rgba(2, 132, 199, 32); border: 1px solid #0284c7; color: #38bdf8; border-radius: 16px; padding: 6px 16px;")

        should_hide_dock = self.view_mode in ("orig", "patch", "subject")
        self.dock.setVisible(not should_hide_dock)

    def _refresh_stats_label(self):
        tot = len(self.candidate_blobs)
        sel = len(self.selected_blob_ids)
        self.lbl_stats.setText(f"🎯 探测 {tot} 处 | 已消除: {sel} 处 | 笔刷: [{self.brush_size}px] | 缩放: {int(self.zoom*100)}%")

    def _clamp_pan_offset(self):
        w, h = self.viewport.width(), self.viewport.height()
        if self.orig_bgr is None:
            return
        iw, ih = self.orig_bgr.shape[1], self.orig_bgr.shape[0]
        base_scale = min(w / float(iw), h / float(ih))
        draw_w = iw * base_scale * self.zoom
        draw_h = ih * base_scale * self.zoom

        limit_x = max(abs(draw_w - w) / 2.0, w * 0.35) + 60.0
        limit_y = max(abs(draw_h - h) / 2.0, h * 0.35) + 60.0

        clamped_x = max(-limit_x, min(limit_x, self.pan.x()))
        clamped_y = max(-limit_y, min(limit_y, self.pan.y()))
        self.pan = QPointF(clamped_x, clamped_y)

    def _zoom_at_cursor(self, cursor_pos: QPointF, multiplier: float):
        old_zoom = self.zoom
        new_zoom = max(0.5, min(12.0, old_zoom * multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        w, h = self.viewport.width(), self.viewport.height()
        if self.orig_bgr is None:
            return
        iw, ih = self.orig_bgr.shape[1], self.orig_bgr.shape[0]
        base_scale = min(w / float(iw), h / float(ih))

        old_dw, old_dh = iw * base_scale * old_zoom, ih * base_scale * old_zoom
        old_dx = (w - old_dw) / 2.0 + self.pan.x()
        old_dy = (h - old_dh) / 2.0 + self.pan.y()

        px = (cursor_pos.x() - old_dx) / (base_scale * old_zoom)
        py = (cursor_pos.y() - old_dy) / (base_scale * old_zoom)

        new_dw, new_dh = iw * base_scale * new_zoom, ih * base_scale * new_zoom
        new_pan_x = cursor_pos.x() - (w - new_dw) / 2.0 - px * base_scale * new_zoom
        new_pan_y = cursor_pos.y() - (h - new_dh) / 2.0 - py * base_scale * new_zoom

        self.zoom = new_zoom
        self.pan = QPointF(new_pan_x, new_pan_y)
        self._clamp_pan_offset()
        self._refresh_stats_label()
        self.viewport.update()

    def _viewport_paint(self, event):
        painter = QPainter(self.viewport)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.viewport.width(), self.viewport.height()
        painter.fillRect(0, 0, w, h, QColor("#050507"))

        if self.view_mode == "fused":
            target = self.pixmap_fused
        elif self.view_mode == "orig":
            target = self.pixmap_orig
        elif self.view_mode == "mask":
            target = self.pixmap_mask
        elif self.view_mode == "patch":
            target = self.pixmap_patch
        else:
            target = self.pixmap_subject

        if target.isNull():
            return

        iw, ih = target.width(), target.height()
        scale = min(w / float(iw), h / float(ih)) * self.zoom
        dw, dh = iw * scale, ih * scale
        dx = (w - dw) / 2.0 + self.pan.x()
        dy = (h - dh) / 2.0 + self.pan.y()
        target_draw_rect = QRectF(dx, dy, dw, dh).toRect()

        # 抠图模式下在主体下方垫一层棋盘格、纯色或自定义背景图
        if self.session_mode == "cutout" and self.view_mode == "fused":
            painter.save()
            painter.setClipRect(target_draw_rect)
            if self.bg_preview_mode == "white":
                painter.fillRect(target_draw_rect, QColor("#FFFFFF"))
            elif self.bg_preview_mode == "black":
                painter.fillRect(target_draw_rect, QColor("#000000"))
            elif self.bg_preview_mode == "green":
                painter.fillRect(target_draw_rect, QColor("#00FF00"))
            elif self.bg_preview_mode == "gray":
                painter.fillRect(target_draw_rect, QColor("#808080"))
            elif self.bg_preview_mode == "custom_img" and hasattr(self, "_custom_bg_pixmap") and not self._custom_bg_pixmap.isNull():
                painter.drawPixmap(target_draw_rect, self._custom_bg_pixmap)
            else:  # "checker"
                painter.drawTiledPixmap(target_draw_rect, self._checker_pixmap)
            painter.restore()

        painter.drawPixmap(target_draw_rect, target)

        if self.is_brushing and not self.current_stroke_path.isEmpty():
            brush_screen_d = (self.brush_size * scale) * 2.0
            swath_color = QColor(255, 255, 255, 95) if self.current_tool == "brush" else QColor(251, 191, 36, 95)
            painter.setPen(QPen(swath_color, brush_screen_d, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(self.current_stroke_path)

        if self.is_lasso_active and not self.lasso_screen_path.isEmpty():
            painter.setPen(QPen(QColor("#38bdf8"), 2.0, Qt.DashLine))
            painter.setBrush(QColor(56, 189, 248, 45))
            painter.drawPath(self.lasso_screen_path)

        if self.view_mode == "fused" and self.hovered_blob_id != -1:
            target_blob = next((b for b in self.candidate_blobs if b["id"] == self.hovered_blob_id), None)
            if target_blob:
                is_sel = self.hovered_blob_id in self.selected_blob_ids
                should_highlight = (self.current_tool == "brush" and not is_sel) or \
                                   (self.current_tool == "eraser" and is_sel)
                if should_highlight:
                    bx, by, bw, bh = target_blob["bbox"]
                    obj_rect = QRectF(dx + bx * scale, dy + by * scale, bw * scale, bh * scale)

                    conic = QConicalGradient(obj_rect.center().x(), obj_rect.center().y(), self.spinner_angle * 4)
                    conic.setColorAt(0.00, QColor("#4285F4"))
                    conic.setColorAt(0.33, QColor("#9B72CF"))
                    conic.setColorAt(0.66, QColor("#D96570"))
                    conic.setColorAt(1.00, QColor("#F4B400"))

                    painter.setPen(QPen(QBrush(conic), 2.2))
                    painter.setBrush(QColor(255, 255, 255, 28))
                    painter.drawRoundedRect(obj_rect, 6, 6)

        if self.is_processing_mask:
            painter.fillRect(self.viewport.rect(), QColor(0, 0, 0, 140))
            badge_w, badge_h = 240, 52
            bx = (w - badge_w) / 2.0
            by = (h - badge_h) / 2.0

            painter.setBrush(QColor("#141418"))
            painter.setPen(QPen(QColor("#0284c7"), 1.5))
            painter.drawRoundedRect(QRectF(bx, by, badge_w, badge_h), 26, 26)

            rcx = bx + 32
            rcy = by + badge_h / 2.0
            painter.save()
            painter.translate(rcx, rcy)
            painter.rotate(self.spinner_angle)
            sp_grad = QConicalGradient(0, 0, 0)
            sp_grad.setColorAt(0.0, QColor("#38bdf8"))
            sp_grad.setColorAt(0.5, QColor("#0284c7"))
            sp_grad.setColorAt(1.0, QColor(2, 132, 199, 0))
            painter.setPen(QPen(QBrush(sp_grad), 3.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(QRectF(-12, -12, 24, 24), 0, 360 * 16)
            painter.restore()

            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.setPen(QColor("#f4f4f5"))
            painter.drawText(QRectF(bx + 54, by, badge_w - 60, badge_h), Qt.AlignVCenter | Qt.AlignLeft, "正在智能消除中...")

        for p in self.particles:
            col = QColor(p.color)
            col.setAlpha(int(p.alpha))
            painter.setPen(Qt.NoPen)
            painter.setBrush(col)
            painter.drawEllipse(p.pos, p.size, p.size)

        can_paint_in_mode = self.view_mode in ("fused", "mask")
        if can_paint_in_mode and not self.is_processing_mask and not self.is_space_held and not self.is_alt_held and self.mouse_view_pos.x() > 0 and self.mouse_view_pos.y() > 0:
            brush_screen_r = self.brush_size * scale
            ring_col = QColor("#d97706") if self.current_tool == "eraser" else (
                QColor("#dc2626") if self.shield_override else QColor("#38bdf8")
            )
            painter.setPen(QPen(ring_col, 2.0))
            painter.setBrush(QColor(ring_col.red(), ring_col.green(), ring_col.blue(), 30))
            painter.drawEllipse(self.mouse_view_pos, brush_screen_r, brush_screen_r)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.mouse_view_pos, 2.5, 2.5)

    def _viewport_wheel(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                mult = 1.15 if delta > 0 else (1.0 / 1.15)
                self._zoom_at_cursor(event.position(), mult)
            event.accept()
        elif event.modifiers() & Qt.ShiftModifier:
            h_delta = event.angleDelta().y() / 2.0 or event.angleDelta().x() / 2.0
            self.pan += QPointF(h_delta, 0)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()
        else:
            dx = event.angleDelta().x() / 2.0
            dy = event.angleDelta().y() / 2.0
            self.pan += QPointF(dx, dy)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()

    def _viewport_mouse_press(self, event: QMouseEvent):
        if self.view_mode in ("orig", "patch", "subject") or self.is_processing_mask:
            if event.button() in (Qt.LeftButton, Qt.MiddleButton):
                self.is_panning = True
                self.last_mouse = event.position()
                self.viewport.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and (self.is_space_held or self.is_alt_held)):
            self.is_panning = True
            self.last_mouse = event.position()
            self.viewport.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        w, h = self.viewport.width(), self.viewport.height()
        if self.orig_bgr is None:
            return
        iw, ih = self.orig_bgr.shape[1], self.orig_bgr.shape[0]
        scale = min(w / float(iw), h / float(ih)) * self.zoom
        dw, dh = iw * scale, ih * scale
        dx = (w - dw) / 2.0 + self.pan.x()
        dy = (h - dh) / 2.0 + self.pan.y()

        ix = int((event.position().x() - dx) / scale)
        iy = int((event.position().y() - dy) / scale)
        

        if event.button() == Qt.RightButton:
            if self.view_mode == "fused":
                self.is_lasso_active = True
                self.lasso_pts_img = [(ix, iy)]
                self.lasso_screen_path = QPainterPath()
                self.lasso_screen_path.moveTo(event.position())
                self.viewport.update()
                event.accept()
                return
            elif self.view_mode == "mask":
                event.ignore()
                return

        if event.button() == Qt.LeftButton:
            if self.view_mode == "fused" and self.hovered_blob_id != -1:
                bid = self.hovered_blob_id
                target_blob = next((b for b in self.candidate_blobs if b["id"] == bid), None)
                is_sel = bid in self.selected_blob_ids

                action_triggered = False
                if self.current_tool == "brush" and not is_sel:
                    self._push_undo_snapshot()
                    self.selected_blob_ids.add(bid)
                    action_triggered = True
                elif self.current_tool == "eraser" and is_sel:
                    self._push_undo_snapshot()
                    self.selected_blob_ids.remove(bid)
                    action_triggered = True

                if action_triggered:
                    self.is_processing_mask = True
                    self.viewport.update()
                    QTimer.singleShot(20, lambda tb=target_blob: self._execute_click_elimination(tb))
                    event.accept()
                    return

            self._push_undo_snapshot()
            self.is_brushing = True
            self.current_stroke_mask.fill(0)
            self.current_stroke_path = QPainterPath()
            self.current_stroke_path.moveTo(event.position())
            self.last_img_pt = (ix, iy)

            is_ctrl = bool(event.modifiers() & Qt.ControlModifier)
            tool = "eraser" if is_ctrl else self.current_tool
            self._record_stroke_pixel(ix, iy, tool)
            self.viewport.update()

    def _execute_click_elimination(self, target_blob):
        if target_blob and target_blob["id"] in self.selected_blob_ids:
            self._spawn_dissolve_particles_only_from_region(target_blob["mask"])
        self._recompute_fusion()
        self._refresh_stats_label()
        self.is_processing_mask = False
        self.viewport.update()

    def _viewport_mouse_move(self, event: QMouseEvent):
        self.mouse_view_pos = event.position()

        if self.is_panning:
            delta = event.position() - self.last_mouse
            self.last_mouse = event.position()
            self.pan += delta
            self._clamp_pan_offset()
            self.viewport.update()
            return

        w, h = self.viewport.width(), self.viewport.height()
        if self.orig_bgr is None:
            return
        iw, ih = self.orig_bgr.shape[1], self.orig_bgr.shape[0]
        scale = min(w / float(iw), h / float(ih)) * self.zoom
        dw, dh = iw * scale, ih * scale
        dx = (w - dw) / 2.0 + self.pan.x()
        dy = (h - dh) / 2.0 + self.pan.y()

        mx = event.position().x()
        my = event.position().y()
        ix = int((mx - dx) / scale)
        iy = int((my - dy) / scale)

        if self.is_lasso_active:
            self.lasso_screen_path.lineTo(event.position())
            self.lasso_pts_img.append((ix, iy))
            self.viewport.update()
            return

        if self.is_brushing:
            self.current_stroke_path.lineTo(event.position())
            is_ctrl = bool(event.modifiers() & Qt.ControlModifier)
            tool = "eraser" if is_ctrl else self.current_tool
            self._record_stroke_pixel(ix, iy, tool)
            self.viewport.update()
            return

        hit_id = -1
        if self.view_mode == "fused" and 0 <= ix < iw and 0 <= iy < ih:
            for b in self.candidate_blobs:
                bx, by, bw, bh = b["bbox"]
                if bx <= ix <= bx + bw and by <= iy <= by + bh:
                    hit_id = b["id"]
                    break

        if hit_id != self.hovered_blob_id:
            self.hovered_blob_id = hit_id

        self.viewport.update()

    def _record_stroke_pixel(self, ix: int, iy: int, tool: str):
        ih, iw = self.orig_bgr.shape[:2]
        r = int(self.brush_size)
        if self.last_img_pt is not None:
            cv2.line(self.current_stroke_mask, self.last_img_pt, (ix, iy), 255, thickness=r * 2)
        cv2.circle(self.current_stroke_mask, (ix, iy), r, 255, -1)
        self.last_img_pt = (ix, iy)

    def _viewport_mouse_release(self, event: QMouseEvent):
        if self.is_panning:
            self.is_panning = False
            self.viewport.setCursor(Qt.OpenHandCursor if (self.is_space_held or self.is_alt_held) else Qt.ArrowCursor)

        if self.is_lasso_active and event.button() == Qt.RightButton:
            self.is_lasso_active = False
            self.lasso_screen_path = QPainterPath()

            if len(self.lasso_pts_img) > 5:
                orig_h, orig_w = self.orig_bgr.shape[:2]
                poly_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                pts_arr = np.array(self.lasso_pts_img, dtype=np.int32).reshape((-1, 1, 2))
                cv2.drawContours(poly_mask, [pts_arr], -1, 255, thickness=-1)

                self._push_undo_snapshot()
                self._spawn_dissolve_particles_only_from_region(poly_mask)

                if self.shield_override:
                    self.override_brush_mask = cv2.bitwise_or(self.override_brush_mask, poly_mask)
                else:
                    self.manual_brush_mask = cv2.bitwise_or(self.manual_brush_mask, poly_mask)
                self.manual_erase_mask[poly_mask > 0] = 0

                self.is_processing_mask = True
                self.viewport.update()
                QTimer.singleShot(20, self._finalize_lasso_fusion)

            self.lasso_pts_img = []
            return

        if self.is_brushing:
            self.is_brushing = False
            self.last_img_pt = None

            is_ctrl = bool(event.modifiers() & Qt.ControlModifier)
            tool = "eraser" if is_ctrl else self.current_tool

            if cv2.countNonZero(self.current_stroke_mask) > 0:
                self.is_processing_mask = True
                self.viewport.update()
                QTimer.singleShot(20, lambda t=tool: self._execute_brush_fusion(t))
            else:
                self.current_stroke_path = QPainterPath()
                self.viewport.update()

    def _finalize_lasso_fusion(self):
        self._recompute_fusion()
        self._refresh_stats_label()
        self.is_processing_mask = False
        self.viewport.update()

    def _execute_brush_fusion(self, tool: str):
        self._spawn_dissolve_particles_only_from_region(self.current_stroke_mask)

        if tool == "eraser":
            self.manual_erase_mask = cv2.bitwise_or(self.manual_erase_mask, self.current_stroke_mask)
            self.manual_brush_mask[self.current_stroke_mask > 0] = 0
            self.override_brush_mask[self.current_stroke_mask > 0] = 0
        else:
            if self.shield_override:
                self.override_brush_mask = cv2.bitwise_or(self.override_brush_mask, self.current_stroke_mask)
            else:
                self.manual_brush_mask = cv2.bitwise_or(self.manual_brush_mask, self.current_stroke_mask)
            self.manual_erase_mask[self.current_stroke_mask > 0] = 0

        self._recompute_fusion()
        self._refresh_stats_label()

        self.current_stroke_path = QPainterPath()
        self.is_processing_mask = False
        self.viewport.update()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        if (key == Qt.Key_Z and (mods & Qt.ControlModifier) and (mods & Qt.AltModifier)) or \
           (key == Qt.Key_Z and (mods & Qt.ControlModifier) and (mods & Qt.ShiftModifier)) or \
           (key == Qt.Key_Y and (mods & Qt.ControlModifier)):
            self._redo()
            event.accept()
            return
        elif key == Qt.Key_Z and (mods & Qt.ControlModifier):
            self._undo()
            event.accept()
            return

        if key == Qt.Key_BracketLeft:
            self.brush_size = max(10, self.brush_size - 5)
            self.slider_brush_sz.setValue(self.brush_size)
            self.viewport.update()
            self._refresh_stats_label()
            event.accept()
            return
        elif key == Qt.Key_BracketRight:
            self.brush_size = min(150, self.brush_size + 5)
            self.slider_brush_sz.setValue(self.brush_size)
            self.viewport.update()
            self._refresh_stats_label()
            event.accept()
            return

        if key == Qt.Key_Up:
            self.pan += QPointF(0, 60)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()
            return
        elif key == Qt.Key_Down:
            self.pan += QPointF(0, -60)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()
            return
        elif key == Qt.Key_Left:
            self.pan += QPointF(60, 0)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()
            return
        elif key == Qt.Key_Right:
            self.pan += QPointF(-60, 0)
            self._clamp_pan_offset()
            self.viewport.update()
            event.accept()
            return

        if key == Qt.Key_Space and not event.isAutoRepeat():
            self.is_space_held = True
            if self.view_mode != "orig":
                self.prev_mode_before_hold = self.view_mode
            self._set_mode("orig")
            event.accept()
            return

        if key == Qt.Key_Alt and not event.isAutoRepeat():
            self.is_alt_held = True
            if self.view_mode != "patch":
                self.prev_mode_before_hold = self.view_mode
            self._set_mode("patch")
            event.accept()
            return

        if key == Qt.Key_1: self._set_mode("fused")
        elif key == Qt.Key_2: self._set_mode("orig")
        elif key == Qt.Key_3: self._set_mode("mask")
        elif key == Qt.Key_4: self._set_mode("patch")
        elif key == Qt.Key_5: self._set_mode("subject")
        elif key == Qt.Key_B: self._set_tool("brush")
        elif key == Qt.Key_E: self._set_tool("eraser")
        elif key == Qt.Key_W: self._toggle_override_shield()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Space and not event.isAutoRepeat():
            self.is_space_held = False
            self._set_mode(getattr(self, "prev_mode_before_hold", "fused"))
            event.accept()
            return

        if key == Qt.Key_Alt and not event.isAutoRepeat():
            self.is_alt_held = False
            self._set_mode(getattr(self, "prev_mode_before_hold", "fused"))
            event.accept()
            return

        super().keyReleaseEvent(event)

    def _save_as(self):
        default_dir = Path(self.orig_file_path).parent if self.orig_file_path else Path.home()
        default_stem = Path(self.orig_file_path).stem if self.orig_file_path else "photo"

        if self.session_mode == "cutout":
            default_name = str(default_dir / f"{default_stem}_cutout.png")
            out, _ = QFileDialog.getSaveFileName(
                self, "另存为透明底满血 PNG",
                default_name,
                "PNG 图像 (*.png);;JPEG 图像 (*.jpg)"
            )
            if out:
                # 若保存 PNG，直接写入 4 通道 BGRA；若保存 JPG，写入白底
                img_to_save = self.bgra_fused if self.bgra_fused is not None else (self.patch_aligned if self.patch_aligned is not None else self.orig_bgr)
                if imwrite_unicode(out, img_to_save, quality=96, orig_file_path=self.orig_file_path):
                    QMessageBox.information(self, "保存成功", f"透明底满血成片已保存至：\n{out}\n\n已完整保留 100% 原图物理分辨率与发丝透光细节！")
        else:
            default_name = str(default_dir / f"{default_stem}_inpainted.jpg")
            out, _ = QFileDialog.getSaveFileName(
                self, "另存为满血成片",
                default_name,
                "JPEG 图像 (*.jpg);;PNG 图像 (*.png)"
            )
            if out and imwrite_unicode(out, self.fused_bgr, quality=96, orig_file_path=self.orig_file_path):
                QMessageBox.information(self, "保存成功", f"满血成片已保存至：\n{out}\n\n已成功保留并注入原始 EXIF 拍摄参数、镜头型号与色彩配置！")

    def _on_bg_preview_changed(self):
        val = self.combo_bg_preview.currentData()
        if val == "custom_img":
            f, _ = QFileDialog.getOpenFileName(
                self, "选择自定义背景图片", "",
                "图片 (*.jpg *.jpeg *.png *.webp *.bmp);;所有文件 (*.*)"
            )
            if f and os.path.exists(f):
                bg_mat = imread_unicode(f)
                if bg_mat is not None:
                    # 自动缩放到原图同比例
                    h, w = self.orig_bgr.shape[:2]
                    resized_bg = cv2.resize(bg_mat, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    self._custom_bg_pixmap = QPixmap.fromImage(bgr_to_qimage(resized_bg))
                    self.bg_preview_mode = "custom_img"
                else:
                    self.combo_bg_preview.setCurrentIndex(0)
            else:
                self.combo_bg_preview.setCurrentIndex(0)
        else:
            self.bg_preview_mode = val

        self.viewport.update()

    def _pick_target_erase_color(self):
        c = QColorDialog.getColor(QColor(self.target_erase_color_hex), self, "选择要精确消除的背景色")
        if c.isValid():
            self.target_erase_color_hex = c.name().upper()
            self.btn_target_color.setText(self.target_erase_color_hex)
            lum = c.lightness()
            txt_col = "#000000" if lum > 140 else "#ffffff"
            self.btn_target_color.setStyleSheet(
                f"QPushButton {{ background-color: {self.target_erase_color_hex}; color: {txt_col}; "
                f"border: 1px solid #c084fc; border-radius: 4px; padding: 2px 8px; "
                f"font-family: Consolas; font-size: 11px; font-weight: bold; }}"
            )

    def _filter_mask_by_target_color(self, roi_mask: np.ndarray) -> np.ndarray:
        """在圈选/涂抹区域内，直接对【原图 orig_bgr】做平滑色差消除，绝不误伤错位发丝、绝不狗啃"""
        src = self.orig_bgr
        if src is None:
            return roi_mask

        tc = QColor(self.target_erase_color_hex)
        target_bgr = np.array([tc.blue(), tc.green(), tc.red()], dtype=np.float32)
        tol = float(self.spin_color_tol.value())

        ys, xs = np.where(roi_mask > 0)
        if len(ys) == 0:
            return roi_mask
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        sub_img = src[y1:y2, x1:x2, :3].astype(np.float32)
        sub_roi = roi_mask[y1:y2, x1:x2] > 0

        if tol <= 0.0:
            # 容差 0：100% 精确相等才杀
            match = (
                (sub_img[:, :, 0] == target_bgr[0]) &
                (sub_img[:, :, 1] == target_bgr[1]) &
                (sub_img[:, :, 2] == target_bgr[2])
            )
            soft_alpha = np.where(match, 255, 0).astype(np.uint8)
        else:
            # 平滑色差斜坡：核心色差内 100% 杀掉，边缘过渡带平滑衰减，消灭 1px 椒盐狗啃
            dist = np.linalg.norm(sub_img - target_bgr, axis=2)
            low = tol * 0.65
            high = max(low + 1.0, tol * 1.25)
            soft = np.clip((high - dist) / (high - low), 0.0, 1.0) * 255.0
            soft_u8 = soft.astype(np.uint8)

            # 3x3 开闭平滑：清除头发丝内部孤立的 1px 椒盐噪点
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core_bin = cv2.morphologyEx((soft_u8 > 128).astype(np.uint8) * 255, cv2.MORPH_OPEN, k3)
            soft_alpha = cv2.GaussianBlur(np.minimum(soft_u8, cv2.dilate(core_bin, k3)), (3, 3), 0)

        out_mask = np.zeros_like(roi_mask)
        out_mask[y1:y2, x1:x2][sub_roi] = soft_alpha[sub_roi]
        return out_mask


# ==============================================================================
# InpaintPageWidget 主工作台组件 (【内嵌堆栈】：Page 0 准备台 / Page 1 消除工作台)
# ==============================================================================
class InpaintPageWidget(QWidget):
    sig_photo_loaded = Signal(str)

    DEFAULT_AI_URL = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image_path = ""
        self.orig_bgr = None
        self.latest_patch_path = ""
        self.patch_bgr = None
        self.init_worker = None
        self.loading_dialog = None
        self.photomerge_widget = None
        self.cosplaymerge_widget = None

        self.cache_root = get_app_cache_root()
        self.download_dir = self.cache_root / "inpaint_downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.web_profile_dir = self.cache_root / "web_profile"
        self.web_profile_dir.mkdir(parents=True, exist_ok=True)

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.page_stack = QStackedWidget(self)

        self.prep_widget = QWidget()
        prep_layout = QHBoxLayout(self.prep_widget)
        prep_layout.setContentsMargins(0, 0, 0, 0)
        prep_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; width: 1px; }")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        left_layout.addWidget(QLabel("🖼️ 1. 精修高保真底图 (LR/蛋糕导出大图)"))
        self.drop_card = QFrame()
        self.drop_card.setAcceptDrops(True)
        self.drop_card.setMinimumHeight(180)
        self.drop_card.setStyleSheet("QFrame { background-color: #141418; border: 2px dashed #3f3f46; border-radius: 8px; }")
        self.drop_card.dragEnterEvent = self._on_drag_enter
        self.drop_card.dropEvent = self._on_drop

        dc_l = QVBoxLayout(self.drop_card)
        self.lbl_preview = QLabel("💡 将修好的底图拖拽扔进此处\n或直接在键盘按 Ctrl + V 粘贴截图")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("color: #71717a; font-size: 12px; line-height: 1.5;")
        dc_l.addWidget(self.lbl_preview)
        left_layout.addWidget(self.drop_card)

        h_clipboard = QHBoxLayout()
        btn_paste_clip = QPushButton("📋 粘贴剪贴板截图")
        btn_paste_clip.setToolTip("从系统剪贴板直接吃入截图 (快捷键 Ctrl+V)")
        btn_paste_clip.clicked.connect(self._paste_from_clipboard)
        h_clipboard.addWidget(btn_paste_clip)

        self.btn_copy_orig = QPushButton("📤 复制底图到剪贴板")
        self.btn_copy_orig.setToolTip("将当前底图塞进剪贴板，去右侧网页直接 Ctrl+V 发图消除")
        self.btn_copy_orig.setEnabled(False)
        self.btn_copy_orig.clicked.connect(self._copy_orig_to_clipboard)
        h_clipboard.addWidget(self.btn_copy_orig)
        left_layout.addLayout(h_clipboard)

        self.lbl_orig_info = QLabel("底图信息: 尚未载入")
        self.lbl_orig_info.setStyleSheet("color: #a1a1aa; font-family: Consolas; font-size: 11px;")
        left_layout.addWidget(self.lbl_orig_info)

        left_layout.addWidget(QLabel("⚡ 2. 网页 AI 消除补丁拦截"))
        self.status_box = QFrame()
        self.status_box.setStyleSheet("background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 8px;")
        sb_l = QVBoxLayout(self.status_box)
        self.lbl_intercept_status = QLabel("🟢 拦截器就绪：网页点击【下载】自动捕获")
        self.lbl_intercept_status.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        sb_l.addWidget(self.lbl_intercept_status)

        self.lbl_patch_file = QLabel("当前补丁: 尚未捕获")
        self.lbl_patch_file.setStyleSheet("color: #a1a1aa; font-family: Consolas; font-size: 11px;")
        left_layout.addWidget(self.status_box)

        left_layout.addWidget(QLabel("🎛️ 3. 缝合参数策略"))
        cfg_box = QFrame()
        cfg_box.setStyleSheet("background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 8px;")
        cb_l = QVBoxLayout(cfg_box)

        h_d = QHBoxLayout()
        h_d.addWidget(QLabel("覆盖吞噬量:"))
        self.slider_dilate = QSlider(Qt.Horizontal)
        self.slider_dilate.setRange(10, 100)
        self.slider_dilate.setValue(55)
        self.lbl_dilate_val = QLabel("55 px")
        self.lbl_dilate_val.setStyleSheet("color: #38bdf8; font-family: Consolas;")
        self.slider_dilate.valueChanged.connect(lambda v: self.lbl_dilate_val.setText(f"{v} px"))
        h_d.addWidget(self.slider_dilate)
        h_d.addWidget(self.lbl_dilate_val)
        cb_l.addLayout(h_d)

        h_f = QHBoxLayout()
        h_f.addWidget(QLabel("边缘羽化:"))
        self.slider_feather = QSlider(Qt.Horizontal)
        self.slider_feather.setRange(2, 30)
        self.slider_feather.setValue(15)
        self.lbl_feather_val = QLabel("15 px")
        self.lbl_feather_val.setStyleSheet("color: #38bdf8; font-family: Consolas;")
        self.slider_feather.valueChanged.connect(lambda v: self.lbl_feather_val.setText(f"{v} px"))
        h_f.addWidget(self.slider_feather)
        h_f.addWidget(self.lbl_feather_val)
        cb_l.addLayout(h_f)

        left_layout.addWidget(cfg_box)

        # 快捷提示词工具组
        left_layout.addWidget(QLabel("💬 4. 常用 AI 指令 (点击秒拷到剪贴板):"))
        prompt_box = QFrame()
        prompt_box.setStyleSheet("background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 6px;")
        pb_l = QVBoxLayout(prompt_box)
        pb_l.setSpacing(5)

        h_prompts = QHBoxLayout()
        h_prompts.setSpacing(6)

        prompts = [
            ("🚶 消除路人", "请消除画面中除前景主体人物外的所有背景路人，完美修补被遮挡的背景环境"),
            ("🧰 消除路人杂物", "请消除画面中的所有背景路人以及地面的垃圾杂物、摄影器材、警示标牌，保持主体不变"),
            ("🖤 黑白剪影抠图", "请将这张图片严格转换成纯黑白剪影蒙版：把前景主体人物（包含所有发丝、服装、配饰和手持道具的完整轮廓）全部填充为纯黑色（#000000），把人物以外的所有背景和发丝之间的缝隙全部填充为纯白色（#FFFFFF），保持原图人物姿势、构图和边缘位置100%不变"),
        ]

        for title, p_text in prompts:
            btn_p = QPushButton(title)
            btn_p.setToolTip(f"点击复制指令：\n{p_text}")
            btn_p.setStyleSheet("""
                QPushButton {
                    background-color: #202028; border: 1px solid #3f3f46; border-radius: 4px;
                    padding: 5px 4px; font-size: 11px; font-weight: bold; color: #38bdf8;
                }
                QPushButton:hover { background-color: #0284c7; color: #ffffff; }
            """)
            btn_p.clicked.connect(lambda ch=False, txt=p_text: self._copy_prompt_to_clipboard(txt))
            h_prompts.addWidget(btn_p)
        pb_l.addLayout(h_prompts)
        left_layout.addWidget(prompt_box)

        # 启动按钮行：双核驱动（魔法消除 / 一键抠图）
        h_fuse_btns = QHBoxLayout()
        h_fuse_btns.setSpacing(8)

        self.btn_fuse = QPushButton("🪄 启动魔法消除")
        self.btn_fuse.setEnabled(False)
        self.btn_fuse.setStyleSheet("""
            QPushButton {
                background-color: #0284c7; color: white; font-weight: bold; padding: 10px;
                font-size: 12px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #0369a1; }
            QPushButton:disabled { background-color: #27272a; color: #52525b; }
        """)
        self.btn_fuse.clicked.connect(lambda: self._do_fusion("inpaint"))
        h_fuse_btns.addWidget(self.btn_fuse)

        self.btn_cutout = QPushButton("✂️ 一键透明抠图")
        self.btn_cutout.setEnabled(False)
        self.btn_cutout.setToolTip("利用 AI 图推导主体轮廓，原图 100% 满血发丝抠出，输出透明底 PNG")
        self.btn_cutout.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; color: white; font-weight: bold; padding: 10px;
                font-size: 12px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #15803d; }
            QPushButton:disabled { background-color: #27272a; color: #52525b; }
        """)
        self.btn_cutout.clicked.connect(lambda: self._do_fusion("cutout"))
        h_fuse_btns.addWidget(self.btn_cutout)

        left_layout.addLayout(h_fuse_btns)

        self.btn_photomerge = QPushButton("🌌 启动二次元光学溶图工作台")
        self.btn_photomerge.setToolTip("进入独立光学溶图工作台（需手动导入底图与透明底 PNG）")
        self.btn_photomerge.setStyleSheet("""
            QPushButton {
                background-color: #6d28d9; color: white; font-weight: bold; padding: 10px;
                font-size: 13px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #5b21b6; }
        """)
        self.btn_photomerge.clicked.connect(self._open_photomerge)
        left_layout.addWidget(self.btn_photomerge)

        self.btn_cosplay_merge = QPushButton("📸 启动场照溶图工作台")
        self.btn_cosplay_merge.setToolTip("进入真人场照光学溶图与虚拟大光圈重打光工作台")
        self.btn_cosplay_merge.setStyleSheet("""
            QPushButton {
                background-color: #0369a1; color: white; font-weight: bold; padding: 10px;
                font-size: 13px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #0284c7; }
        """)
        self.btn_cosplay_merge.clicked.connect(self._open_cosplay_merge)
        left_layout.addWidget(self.btn_cosplay_merge)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        web_bar = QFrame()
        web_bar.setFixedHeight(36)
        web_bar.setStyleSheet("background-color: #141418; border-bottom: 1px solid #27272f; padding: 2px 8px;")
        wb_l = QHBoxLayout(web_bar)
        wb_l.setContentsMargins(4, 2, 4, 2)

        self.btn_web_back = QPushButton("‹")
        self.btn_web_back.setFixedSize(26, 24)
        wb_l.addWidget(self.btn_web_back)

        self.btn_web_reload = QPushButton("⟳")
        self.btn_web_reload.setFixedSize(26, 24)
        wb_l.addWidget(self.btn_web_reload)

        self.edit_web_url = QLineEdit()
        self.edit_web_url.setStyleSheet("QLineEdit { background-color: #18181e; border: 1px solid #3f3f46; border-radius: 4px; padding: 2px 8px; color: #38bdf8; font-family: Consolas; font-size: 11px; }")
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        saved_url = settings.value("inpaint_ai_url", self.DEFAULT_AI_URL)
        self.edit_web_url.setText(saved_url)
        self.edit_web_url.returnPressed.connect(self._load_url_from_edit)
        wb_l.addWidget(self.edit_web_url, stretch=1)

        btn_go = QPushButton("跳转")
        btn_go.setFixedWidth(46)
        btn_go.clicked.connect(self._load_url_from_edit)
        wb_l.addWidget(btn_go)

        right_layout.addWidget(web_bar)

        if HAS_WEBENGINE:
            self._setup_webengine(right_layout, saved_url)
        else:
            right_layout.addWidget(QLabel("❌ 未检测到 PySide6.QtWebEngineWidgets"))

        splitter.addWidget(right_panel)
        splitter.setSizes([380, 940])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        prep_layout.addWidget(splitter)
        self.page_stack.addWidget(self.prep_widget)

        self.workbench_widget = InpaintWorkbenchWidget(self)
        self.workbench_widget.sig_back_requested.connect(self._switch_to_prep)
        self.page_stack.addWidget(self.workbench_widget)

        root_layout.addWidget(self.page_stack)
        self.page_stack.setCurrentIndex(0)

    def _setup_webengine(self, parent_layout: QVBoxLayout, initial_url: str):
        self.custom_profile = QWebEngineProfile("AnicropInpaintSandbox", self)
        self.custom_profile.setPersistentStoragePath(str(self.web_profile_dir))
        self.custom_profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
        self.custom_profile.downloadRequested.connect(self._on_download_requested)

        self.web_page = QWebEnginePage(self.custom_profile, self)
        try:
            self.web_page.certificateError.connect(lambda error: error.acceptCertificate())
        except Exception:
            pass
        self.web_view = QWebEngineView(self)
        self.web_view.setPage(self.web_page)

        self.btn_web_back.clicked.connect(self.web_view.back)
        self.btn_web_reload.clicked.connect(self.web_view.reload)

        parent_layout.addWidget(self.web_view, stretch=1)
        self.web_view.setUrl(QUrl(initial_url))

    def _load_url_from_edit(self):
        if not HAS_WEBENGINE: return
        url_text = self.edit_web_url.text().strip()
        if not url_text.startswith("http://") and not url_text.startswith("https://"):
            url_text = "https://" + url_text
            self.edit_web_url.setText(url_text)
        QSettings("AnicropGeeks", "AnicropStudio").setValue("inpaint_ai_url", url_text)
        self.web_view.setUrl(QUrl(url_text))

    def _on_download_requested(self, download: QWebEngineDownloadRequest):
        orig_name = download.downloadFileName() or "patch.png"
        ext = Path(orig_name).suffix.lower() or ".png"
        target_filename = f"patch_{int(time.time() * 1000)}{ext}"
        target_path = self.download_dir / target_filename

        download.setDownloadDirectory(str(self.download_dir))
        download.setDownloadFileName(target_filename)
        download.accept()

        self.lbl_intercept_status.setText(f"⚡ 正在接收补丁流: {target_filename}...")
        self.lbl_intercept_status.setStyleSheet("color: #facc15; font-size: 11px; font-weight: bold;")

        def _on_state_changed(state):
            if state == QWebEngineDownloadRequest.DownloadState.DownloadCompleted:
                self.latest_patch_path = str(target_path)
                self.patch_bgr = imread_unicode(self.latest_patch_path)
                self.lbl_intercept_status.setText("✅ 成功拦截下载，补丁已就绪！")
                self.lbl_intercept_status.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
                self.lbl_patch_file.setText(f"📁 {target_filename} ({self.patch_bgr.shape[1]}×{self.patch_bgr.shape[0]})")
                if self.orig_bgr is not None:
                    self.btn_fuse.setEnabled(True)
                    self.btn_cutout.setEnabled(True)

        download.stateChanged.connect(_on_state_changed)

    def _on_drag_enter(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def _on_drop(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            fpath = urls[0].toLocalFile()
            if Path(fpath).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.load_original_photo(fpath)
                event.acceptProposedAction()

    def _paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        img = clipboard.image()
        if img.isNull():
            QMessageBox.information(self, "剪贴板为空", "剪贴板里没有图片数据，请先截图！")
            return

        tmp_clip_path = self.cache_root / f"clipboard_{int(time.time()*1000)}.png"
        img.save(str(tmp_clip_path), "PNG")
        self.load_original_photo(str(tmp_clip_path))

    def _copy_orig_to_clipboard(self):
        if self.orig_bgr is None:
            return
        qimg = bgr_to_qimage(self.orig_bgr)
        QApplication.clipboard().setImage(qimg)
        QMessageBox.information(self, "复制成功", "✅ 底图已复制进系统剪贴板！\n直接在右侧豆包网页输入框按【Ctrl + V】即可秒级发图消除！")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            self._paste_from_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def load_original_photo(self, file_path: str):
        img_bgr = imread_unicode(file_path)
        if img_bgr is None:
            QMessageBox.warning(self, "读取失败", "无法解码选中的底图文件！")
            return
        self.original_image_path = file_path
        self.orig_bgr = img_bgr
        h, w = img_bgr.shape[:2]
        self.lbl_orig_info.setText(f"底图: {Path(file_path).name} ({w}×{h})")

        pix = QPixmap.fromImage(bgr_to_qimage(img_bgr))
        self.lbl_preview.setPixmap(pix.scaled(self.drop_card.width() - 20, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.drop_card.setStyleSheet("QFrame { background-color: #141418; border: 2px solid #0284c7; border-radius: 8px; }")

        self.btn_copy_orig.setEnabled(True)
        if self.patch_bgr is not None:
            self.btn_fuse.setEnabled(True)
            self.btn_cutout.setEnabled(True)

        self.sig_photo_loaded.emit(file_path)

    def _open_photomerge(self):
        if self.photomerge_widget is None:
            try:
                from photomerge.merge_ui import PhotoMergePageWidget
                self.photomerge_widget = PhotoMergePageWidget(self)
                self.photomerge_widget.sig_back_requested.connect(self._switch_to_prep)
                self.page_stack.addWidget(self.photomerge_widget)
            except Exception as e:
                QMessageBox.critical(self, "溶图工作台加载失败", f"无法初始化光学溶图引擎:\n{e}")
                return

        self.page_stack.setCurrentWidget(self.photomerge_widget)
        self.photomerge_widget.setFocus()

    def _open_cosplay_merge(self):
        if self.cosplaymerge_widget is None:
            try:
                from photomerge.merge_ui import PhotoMergePageWidget
                self.cosplaymerge_widget = PhotoMergePageWidget(self, mode="cosplay")
                self.cosplaymerge_widget.sig_back_requested.connect(self._switch_to_prep)
                self.page_stack.addWidget(self.cosplaymerge_widget)
            except Exception as e:
                QMessageBox.critical(self, "场照溶图工作台加载失败", f"无法初始化场照光学溶图引擎:\n{e}")
                return

        self.page_stack.setCurrentWidget(self.cosplaymerge_widget)
        self.cosplaymerge_widget.setFocus()
        if self.original_image_path and os.path.exists(self.original_image_path) and self.cosplaymerge_widget.pipeline.bg_full_u8 is None:
            self.cosplaymerge_widget.load_cosplay_photo(self.original_image_path)

    def _switch_to_prep(self):
        self.page_stack.setCurrentIndex(0)

    def _on_init_worker_error(self, err_msg: str):
        if self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None
        self.btn_fuse.setEnabled(True)
        self.btn_cutout.setEnabled(True)
        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(False)
        QMessageBox.critical(self, "计算异常", f"缝合前处理失败:\n{err_msg}")

    def _copy_prompt_to_clipboard(self, prompt_text: str):
        clipboard = QApplication.clipboard()
        clipboard.setText(prompt_text)
        self.lbl_intercept_status.setText(f"📋 指令已复制：可在右侧网页直接 Ctrl+V 发送！")
        self.lbl_intercept_status.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")

    def _do_fusion(self, session_mode: str = "inpaint"):
        if self.orig_bgr is None or self.patch_bgr is None:
            return

        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(True)

        self.btn_fuse.setEnabled(False)
        self.btn_cutout.setEnabled(False)
        self.loading_dialog = InpaintLoadingDialog(self)

        self.init_worker = InpaintInitWorker(self.orig_bgr, self.patch_bgr, session_mode=session_mode)
        self.init_worker.sig_status.connect(self.loading_dialog.set_status)
        self.init_worker.sig_finished.connect(self._on_init_worker_finished)
        self.init_worker.sig_error.connect(self._on_init_worker_error)
        self.init_worker.start()

        self.loading_dialog.exec()

    def _on_init_worker_finished(self, data: dict):
        self.btn_fuse.setEnabled(True)
        self.btn_cutout.setEnabled(True)
        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(False)

        self.workbench_widget._incoming_base_alpha = data.get("base_alpha")
        self.workbench_widget.load_session_data(
            orig_bgr=self.orig_bgr,
            patch_aligned=data["patch_aligned"],
            subject_matte=data["subject_matte"],
            candidate_blobs=data["candidate_blobs"],
            orig_file_path=self.original_image_path,
            dilate_px=self.slider_dilate.value(),
            feather_px=self.slider_feather.value(),
            session_mode=data.get("session_mode", "inpaint")
        )
        self.page_stack.setCurrentIndex(1)
        self.workbench_widget.setFocus()

        if self.loading_dialog:
            self.loading_dialog.accept()
            self.loading_dialog = None