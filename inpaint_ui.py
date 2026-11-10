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
    QSlider, QComboBox, QFileDialog, QStackedWidget, QApplication
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


def imread_unicode(path: str) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray, quality: int = 95) -> bool:
    try:
        ext = Path(path).suffix.lower()
        if ext in ('.jpg', '.jpeg'):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif ext == '.png':
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        else:
            params = []
        res, buf = cv2.imencode(ext, img, params)
        if res:
            buf.tofile(path)
            return True
        return False
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


class InpaintInitWorker(QThread):
    sig_status = Signal(str, int)
    sig_finished = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, orig_bgr: np.ndarray, patch_bgr: np.ndarray):
        super().__init__()
        self.orig_bgr = orig_bgr
        self.patch_bgr = patch_bgr

    def run(self):
        try:
            self.sig_status.emit("🎯 正在执行 SIFT + RANSAC 亚像素图层锁死配准...", 15)
            patch_aligned, align_success = auto_align_patch(self.orig_bgr, self.patch_bgr)

            self.sig_status.emit("🤖 正在启动 MODNet 神经网络，构建模特发丝指尖结界...", 45)
            subject_matte = PortraitMatting.extract_subject_matte(self.orig_bgr, ref_size=512)

            self.sig_status.emit("🔍 正在探测全图路人消除区与影子大包围网...", 75)
            orig_h, orig_w = self.orig_bgr.shape[:2]
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
            candidate_blobs = []
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
                "candidate_blobs": candidate_blobs
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

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start(16)

        self._init_ui()

    def load_session_data(self, orig_bgr: np.ndarray, patch_aligned: np.ndarray,
                          subject_matte: np.ndarray, candidate_blobs: list,
                          orig_file_path: str, dilate_px: int, feather_px: int):
        self.orig_bgr = orig_bgr
        self.patch_aligned = patch_aligned
        self.subject_matte = subject_matte
        self.candidate_blobs = candidate_blobs
        self.orig_file_path = orig_file_path
        self.dilate_px = dilate_px
        self.feather_px = feather_px

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

    def _clear_brush_and_erase(self):
        """【已补齐核心】点击重置按钮：清空所有手动笔刷，恢复初始消除状态"""
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
        if self.orig_bgr is None or self.patch_aligned is None:
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
        tb_layout.setSpacing(8)

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

        tb_layout.addSpacing(6)

        self.btn_fused = QPushButton("✨ 快速消除 [1]")
        self.btn_orig = QPushButton("📷 消除前 [2]")
        self.btn_mask = QPushButton("🖤 精修模式 [3]")
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

        self.btn_clear_all_brush = QPushButton("🔄 重置")
        self.btn_clear_all_brush.setToolTip("重置为初始状态")
        self.btn_clear_all_brush.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_all_brush.setStyleSheet("""
            QPushButton {
                background-color: rgba(249, 115, 22, 22);
                border: 1px solid rgba(249, 115, 22, 75);
                color: #fb923c;
                border-radius: 16px;
                padding: 6px 16px;
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
        dw = min(920, max(840, self.viewport.width() - 80))
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

        painter.drawPixmap(QRectF(dx, dy, dw, dh).toRect(), target)

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
        p = Path(self.orig_file_path)
        out, _ = QFileDialog.getSaveFileName(
            self, "另存为满血成片",
            str(p.parent / f"{p.stem}_inpainted.jpg"),
            "JPEG 图像 (*.jpg);;PNG 图像 (*.png)"
        )
        if out and imwrite_unicode(out, self.fused_bgr, quality=96):
            QMessageBox.information(self, "保存成功", f"满血无损成片已保存至：\n{out}")


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
        sb_l.addWidget(self.lbl_patch_file)
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

        self.btn_fuse = QPushButton("🪄 启动 AI 魔法消除与缝合")
        self.btn_fuse.setEnabled(False)
        self.btn_fuse.setStyleSheet("""
            QPushButton {
                background-color: #0284c7; color: white; font-weight: bold; padding: 10px;
                font-size: 13px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #0369a1; }
            QPushButton:disabled { background-color: #27272a; color: #52525b; }
        """)
        self.btn_fuse.clicked.connect(self._do_fusion)
        left_layout.addWidget(self.btn_fuse)

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

        self.sig_photo_loaded.emit(file_path)

    def _do_fusion(self):
        if self.orig_bgr is None or self.patch_bgr is None:
            return

        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(True)

        self.btn_fuse.setEnabled(False)
        self.loading_dialog = InpaintLoadingDialog(self)

        self.init_worker = InpaintInitWorker(self.orig_bgr, self.patch_bgr)
        self.init_worker.sig_status.connect(self.loading_dialog.set_status)
        self.init_worker.sig_finished.connect(self._on_init_worker_finished)
        self.init_worker.sig_error.connect(self._on_init_worker_error)
        self.init_worker.start()

        self.loading_dialog.exec()

    def _on_init_worker_finished(self, data: dict):
        self.btn_fuse.setEnabled(True)
        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(False)

        self.workbench_widget.load_session_data(
            orig_bgr=self.orig_bgr,
            patch_aligned=data["patch_aligned"],
            subject_matte=data["subject_matte"],
            candidate_blobs=data["candidate_blobs"],
            orig_file_path=self.original_image_path,
            dilate_px=self.slider_dilate.value(),
            feather_px=self.slider_feather.value()
        )
        self.page_stack.setCurrentIndex(1)
        self.workbench_widget.setFocus()

        if self.loading_dialog:
            self.loading_dialog.accept()
            self.loading_dialog = None

    def _switch_to_prep(self):
        self.page_stack.setCurrentIndex(0)

    def _on_init_worker_error(self, err_msg: str):
        if self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None
        self.btn_fuse.setEnabled(True)
        win = self.window()
        if hasattr(win, "indicator") and win.indicator:
            win.indicator.set_busy(False)
        QMessageBox.critical(self, "计算异常", f"缝合前处理失败:\n{err_msg}")