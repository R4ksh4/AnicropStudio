# -*- coding: utf-8 -*-
"""
watermark_ui.py - Anicrop Studio 参数画框水印独立工作台
完全解耦的独立组件：
1. 默认药丸边框粗细固定为黄金 2px。
2. Liquid Glass 质感深度绑定透明/毛玻璃模式，彻底去开关化。
3. 增加药丸透光度 (Alpha: 10%~100%) 滑块与微调框，自由掌控悬浮玻璃质感。
4. 毛玻璃顶栏自适应抗炫光高亮显示，全套阳间画幅与 5 大独立图层系统。
"""

import os
import json
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QSlider, QFrame, QGroupBox, QComboBox, QColorDialog,
    QSpinBox, QScrollArea, QFileDialog, QSplitter, QProgressBar, QDialog,
    QMessageBox
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QThread
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QPixmap, QImage,
    QDragEnterEvent, QDropEvent, QTransform, QPainterPath
)


# ==============================================================================
# 通用组件 1: 滑块 + 输入微调框 双模联动组件 (加宽至 78px，彻底告别切字)
# ==============================================================================
class SliderInputRow(QWidget):
    valueChanged = Signal(int)

    def __init__(self, label_text: str, min_val: int, max_val: int, default_val: int, unit: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self.lbl = QLabel(label_text)
        self.lbl.setStyleSheet("color: #d4d4d8; font-size: 12px; font-weight: 500;")
        self.lbl.setFixedWidth(112)
        layout.addWidget(self.lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #27272f; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #0284c7; border-radius: 2px; }
            QSlider::handle:horizontal { width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; background: #38bdf8; }
        """)
        layout.addWidget(self.slider)

        self.spin = QSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)
        if unit:
            self.spin.setSuffix(f" {unit}")
        self.spin.setFixedWidth(78)
        self.spin.setStyleSheet("""
            QSpinBox {
                background-color: #141418;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding-left: 4px;
                padding-right: 20px;
                color: #38bdf8;
                font-weight: bold;
                font-size: 11px;
                min-height: 24px;
            }
            QSpinBox:focus { border: 1px solid #38bdf8; }
            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 17px;
                background: #202028;
                border-left: 1px solid #3f3f46;
                border-bottom: 1px solid #27272f;
            }
            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 17px;
                background: #202028;
                border-left: 1px solid #3f3f46;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #0284c7; }
            QSpinBox::up-arrow {
                width: 0; height: 0;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-bottom: 4px solid #38bdf8;
            }
            QSpinBox::down-arrow {
                width: 0; height: 0;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 4px solid #38bdf8;
            }
        """)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.valueChanged.emit)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, val: int):
        self.slider.setValue(val)


# ==============================================================================
# 通用组件 2: 色块 + 调色盘 + Hex 输入框 + 常用预设色
# ==============================================================================
class ColorPickerRow(QWidget):
    colorChanged = Signal(str)

    PRESET_COLORS = [
        ("#FFFFFF", "纯白"),
        ("#F4F4F5", "画廊白"),
        ("#F7F4EE", "复古米"),
        ("#D4D4D8", "细灰边"),
        ("#121214", "哑光黑"),
        ("#000000", "纯黑"),
        ("#0C4A6E", "暗海蓝"),
    ]

    def __init__(self, label_text: str, default_hex: str = "#FFFFFF", parent=None):
        super().__init__(parent)
        self.current_hex = default_hex.upper()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color: #d4d4d8; font-size: 12px; font-weight: 500;")
        lbl.setFixedWidth(115)
        top_row.addWidget(lbl)

        self.btn_swatch = QPushButton()
        self.btn_swatch.setFixedSize(26, 22)
        self.btn_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_swatch.clicked.connect(self._open_palette)
        top_row.addWidget(self.btn_swatch)

        self.edit_hex = QLineEdit(self.current_hex)
        self.edit_hex.setFixedWidth(80)
        self.edit_hex.setStyleSheet("background-color: #141418; border: 1px solid #3f3f46; border-radius: 4px; padding: 3px 6px; color: #ffffff; font-family: Consolas; font-size: 11px;")
        self.edit_hex.textChanged.connect(self._on_hex_edited)
        top_row.addWidget(self.edit_hex)
        top_row.addStretch()
        layout.addLayout(top_row)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(121, 0, 0, 0)
        preset_row.setSpacing(5)
        for hex_code, tip in self.PRESET_COLORS:
            btn = QPushButton()
            btn.setFixedSize(16, 16)
            btn.setToolTip(f"{tip} ({hex_code})")
            btn.setStyleSheet(f"background-color: {hex_code}; border: 1px solid #3f3f46; border-radius: 3px;")
            btn.clicked.connect(lambda ch=False, c=hex_code: self.setColor(c))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        self._refresh_swatch()

    def _refresh_swatch(self):
        self.btn_swatch.setStyleSheet(f"background-color: {self.current_hex}; border: 2px solid #52525b; border-radius: 4px;")

    def _open_palette(self):
        c = QColorDialog.getColor(QColor(self.current_hex), self, "选择颜色")
        if c.isValid():
            self.setColor(c.name().upper())

    def _on_hex_edited(self, text: str):
        t = text.strip()
        if len(t) == 7 and t.startswith("#"):
            c = QColor(t)
            if c.isValid():
                self.current_hex = t.upper()
                self._refresh_swatch()
                self.colorChanged.emit(self.current_hex)

    def setColor(self, hex_code: str):
        self.current_hex = hex_code.upper()
        self.edit_hex.blockSignals(True)
        self.edit_hex.setText(self.current_hex)
        self.edit_hex.blockSignals(False)
        self._refresh_swatch()
        self.colorChanged.emit(self.current_hex)

    def colorHex(self) -> str:
        return self.current_hex


# ==============================================================================
# 独立的所见即所得实时大图交互预览视口
# ==============================================================================
class WatermarkPreviewViewer(QFrame):
    sig_image_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("QFrame { background-color: #050507; border: 1px solid #27272f; border-radius: 6px; }")

        self.preview_image = QImage()
        self.has_photo = False

    def setImage(self, qimg: QImage):
        self.preview_image = qimg
        self.has_photo = not qimg.isNull()
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = Path(path).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".webp", ".cr3", ".cr2", ".arw", ".nef", ".dng"}:
                self.sig_image_dropped.emit(path)
                event.acceptProposedAction()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#09090b"))

        if not self.has_photo or self.preview_image.isNull():
            painter.setPen(QPen(QColor("#3f3f46"), 1.5, Qt.DashLine))
            drop_rect = self.rect().adjusted(40, 40, -40, -40)
            painter.drawRoundedRect(drop_rect, 10, 10)

            painter.setPen(QColor("#71717a"))
            painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
            painter.drawText(drop_rect.adjusted(0, -20, 0, 0), Qt.AlignCenter, "🖼️ 将照片拖拽扔到此视口预览")

            painter.setFont(QFont("Segoe UI", 11))
            painter.setPen(QColor("#52525b"))
            painter.drawText(drop_rect.adjusted(0, 30, 0, 0), Qt.AlignCenter, "支持 JPG / PNG / 佳能 CR3 / 索尼 ARW / 尼康 NEF\n或直接在左侧加载图片")
            return

        img_sz = self.preview_image.size()
        scale = min((w - 32) / img_sz.width(), (h - 32) / img_sz.height())
        draw_w = int(img_sz.width() * scale)
        draw_h = int(img_sz.height() * scale)
        draw_x = (w - draw_w) // 2
        draw_y = (h - draw_h) // 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 190))
        painter.drawRect(draw_x + 6, draw_y + 6, draw_w, draw_h)

        painter.drawImage(QRectF(draw_x, draw_y, draw_w, draw_h), self.preview_image)


# ==============================================================================
# 核心渲染引擎 (Liquid Glass 磨砂质感 + 高清毛玻璃顶栏抗混淆)
# ==============================================================================
class WatermarkEngine:
    _LOGO_CACHE = {}
    PREVIEW_BASELINE_DIM = 1600.0

    @staticmethod
    def detect_brand(camera_str: str) -> str:
        s = camera_str.upper()
        if "CANON" in s or "EOS" in s:
            return "CANON"
        if "SONY" in s or "ILCE" in s:
            return "SONY"
        if "NIKON" in s:
            return "NIKON"
        if "FUJIFILM" in s or "FUJI" in s or "GFX" in s:
            return "FUJIFILM"
        if "LEICA" in s:
            return "LEICA"
        if "HASSELBLAD" in s:
            return "HASSELBLAD"
        if "DJI" in s or "POCKET" in s:
            return "DJI"
        if "APPLE" in s or "IPHONE" in s:
            return "APPLE"
        return "GENERIC"

    @classmethod
    def get_brand_logo_pixmap(cls, brand: str, custom_path: str = "") -> QPixmap | None:
        if custom_path and os.path.exists(custom_path):
            if custom_path in cls._LOGO_CACHE:
                return cls._LOGO_CACHE[custom_path]
            pix = QPixmap(custom_path)
            if not pix.isNull():
                cls._LOGO_CACHE[custom_path] = pix
                return pix

        if brand in cls._LOGO_CACHE:
            return cls._LOGO_CACHE[brand]

        base_dir = Path(__file__).resolve().parent
        logo_dir = base_dir / "assets" / "logos"

        candidates = [
            logo_dir / f"{brand.lower()}.png",
            logo_dir / f"{brand.lower()}_logo.png",
            logo_dir / f"{brand.lower()}.svg",
        ]

        for p in candidates:
            if p.exists():
                pix = QPixmap(str(p))
                if not pix.isNull():
                    cls._LOGO_CACHE[brand] = pix
                    return pix

        cls._LOGO_CACHE[brand] = None
        return None

    @classmethod
    def draw_brand_logo(cls, painter: QPainter, brand: str, rect: QRectF, is_dark: bool, custom_path: str = "") -> int:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        pix = cls.get_brand_logo_pixmap(brand, custom_path)
        if pix and not pix.isNull():
            scaled = pix.scaled(
                int(rect.width()), int(rect.height()),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            draw_x = rect.left() + (rect.width() - scaled.width()) / 2.0
            draw_y = rect.top() + (rect.height() - scaled.height()) / 2.0
            painter.drawPixmap(int(draw_x), int(draw_y), scaled)
            painter.restore()
            return scaled.width()

        badge_h = rect.height()
        drawn_width = 0

        if brand == "LEICA":
            r = min(rect.width(), badge_h) * 0.42
            cx = rect.center().x()
            cy = rect.center().y()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#DC2626"))
            painter.drawEllipse(QPointF(cx, cy), r, r)
            f = QFont("Georgia")
            f.setPixelSize(max(8, int(r * 1.1)))
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(QRectF(cx - r, cy - r, r * 2, r * 2), Qt.AlignCenter, "L")
            drawn_width = int(r * 2)

        elif brand == "CANON":
            f = QFont("Arial Black")
            f.setPixelSize(max(10, int(badge_h * 0.70)))
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor("#DC2626") if is_dark else QColor("#B91C1C"))
            fm = painter.fontMetrics()
            txt = "Canon"
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, txt)
            drawn_width = fm.horizontalAdvance(txt)

        elif brand == "SONY":
            f = QFont("Georgia")
            f.setPixelSize(max(10, int(badge_h * 0.65)))
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor("#FFFFFF") if is_dark else QColor("#111827"))
            fm = painter.fontMetrics()
            txt = "SONY"
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, txt)
            drawn_width = fm.horizontalAdvance(txt)

        elif brand == "NIKON":
            f = QFont("Arial")
            f.setPixelSize(max(10, int(badge_h * 0.70)))
            f.setBold(True)
            f.setItalic(True)
            painter.setFont(f)
            painter.setPen(QColor("#FACC15"))
            fm = painter.fontMetrics()
            txt = "Nikon"
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, txt)
            drawn_width = fm.horizontalAdvance(txt)

        elif brand == "FUJIFILM":
            f = QFont("Arial Black")
            f.setPixelSize(max(9, int(badge_h * 0.58)))
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QColor("#10B981") if is_dark else QColor("#047857"))
            fm = painter.fontMetrics()
            txt = "FUJIFILM"
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignLeft, txt)
            drawn_width = fm.horizontalAdvance(txt)

        painter.restore()
        return drawn_width

    @staticmethod
    def draw_drop_shadow(painter: QPainter, rect: QRectF, radius: float, shadow_blur: int = 35):
        if shadow_blur <= 0:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        layers = 6
        for i in range(layers, 0, -1):
            spread = (shadow_blur / float(layers)) * i
            alpha = int(24 * (1.0 - i / float(layers + 1)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, alpha))
            shadow_rect = rect.adjusted(-spread * 0.6, spread * 0.2, spread * 0.6, spread * 1.1)
            painter.drawRoundedRect(shadow_rect, radius + spread * 0.3, radius + spread * 0.3)

        painter.restore()

    @staticmethod
    def generate_blurred_background(source_qimg: QImage, target_w: int, target_h: int, blur_strength: int, mask_alpha: float) -> QImage:
        img_rgb = source_qimg.convertToFormat(QImage.Format_RGB888)
        w, h = img_rgb.width(), img_rgb.height()
        bpl = img_rgb.bytesPerLine()
        buf = np.frombuffer(img_rgb.bits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)

        small_w = max(32, target_w // 8)
        small_h = max(32, target_h // 8)
        resized = cv2.resize(buf, (small_w, small_h), interpolation=cv2.INTER_AREA)

        ksize = int(blur_strength)
        if ksize % 2 == 0: ksize += 1
        ksize = max(3, ksize)
        blurred = cv2.GaussianBlur(resized, (ksize, ksize), 0)

        bg_full = cv2.resize(blurred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if mask_alpha > 0.001:
            bg_full = (bg_full.astype(np.float32) * (1.0 - mask_alpha)).astype(np.uint8)

        h_out, w_out = bg_full.shape[:2]
        res_qimg = QImage(bg_full.data, w_out, h_out, w_out * 3, QImage.Format_RGB888).copy()
        return res_qimg.convertToFormat(QImage.Format_ARGB32_Premultiplied)

    @classmethod
    def render(cls, source_qimg: QImage, exif_data: dict, cfg: dict) -> QImage:
        """核心渲染引擎：全局动态相对比例缩放，确保大图导出与预览 100% 绝对一致"""
        if source_qimg.isNull():
            return QImage()

        cur_img = source_qimg
        img_rot = cfg.get("image_rotation", 0)
        if img_rot != 0:
            cur_img = cur_img.transformed(QTransform().rotate(img_rot))

        orig_w, orig_h = cur_img.width(), cur_img.height()
        ref_dim = min(orig_w, orig_h)

        scale_k = max(0.2, ref_dim / cls.PREVIEW_BASELINE_DIM)
        frame_mode = cfg.get("frame_mode", "bottom_bar")
        style_layout = cfg.get("style_layout", "standard")

        # 提前无条件声明顶部展签高度
        top_header_h = max(24, int(orig_h * 0.08)) if style_layout == "capsule_pills" else 0

        bar_ratio = cfg.get("bar_height_ratio", 12) / 100.0
        bar_h = int(orig_h * bar_ratio)

        pad_h_ratio = cfg.get("padding_h_ratio", 0) / 100.0
        pad_v_ratio = cfg.get("padding_v_ratio", 0) / 100.0

        if frame_mode == "surround":
            pad_px_h = int(orig_w * pad_h_ratio)
            pad_px_v = int(orig_h * pad_v_ratio)
            base_content_w = orig_w + pad_px_h * 2
            base_content_h = orig_h + pad_px_v * 2 + bar_h + top_header_h
            base_img_x = pad_px_h
            base_img_y = pad_px_v + top_header_h
            bar_start_y = pad_px_v + top_header_h + orig_h
        else:
            pad_px_h = 0
            pad_px_v = 0
            base_content_w = orig_w
            base_content_h = orig_h + bar_h + top_header_h
            base_img_x = 0
            base_img_y = top_header_h
            bar_start_y = orig_h + top_header_h

        corner_radius = int(cfg.get("corner_radius", 0) * scale_k)
        shadow_blur = int(cfg.get("shadow_blur", 0) * scale_k)

        # 比例适配
        ratio_mode = cfg.get("ratio_mode", "free")
        canvas_w = base_content_w
        canvas_h = base_content_h
        extra_x = 0
        extra_y = 0

        if ratio_mode not in ("free", "native"):
            ratio_map = {
                "9:20": 9.0 / 20.0, "9:19.5": 9.0 / 19.5, "9:16": 9.0 / 16.0,
                "4:5": 4.0 / 5.0, "3:4": 3.0 / 4.0, "2:3": 2.0 / 3.0,
                "3:2": 3.0 / 2.0, "4:3": 4.0 / 3.0, "16:9": 16.0 / 9.0,
                "16:10": 16.0 / 10.0, "21:9": 21.0 / 9.0, "1:1": 1.0,
            }
            target_r = ratio_map.get(ratio_mode, canvas_w / float(canvas_h))
            cur_r = canvas_w / float(canvas_h)
            if cur_r > target_r:
                new_h = int(canvas_w / target_r)
                extra_y = (new_h - canvas_h) // 2
                canvas_h = new_h
            else:
                new_w = int(canvas_h * target_r)
                extra_x = (new_w - canvas_w) // 2
                canvas_w = new_w

        img_x = base_img_x + extra_x
        img_y = base_img_y + extra_y
        bar_top = bar_start_y + extra_y

        bar_x = img_x if frame_mode == "surround" else 0
        bar_w = orig_w if frame_mode == "surround" else canvas_w

        bg_type = cfg.get("bg_type", "solid")
        bg_hex = cfg.get("bg_color", "#FFFFFF")
        text_hex = cfg.get("text_color", "#121214")
        is_dark_bg = (QColor(bg_hex).lightness() < 128) if bg_type == "solid" else True

        if bg_type == "transparent":
            canvas = QImage(canvas_w, canvas_h, QImage.Format_ARGB32_Premultiplied)
            canvas.fill(Qt.transparent)
        elif bg_type == "blur":
            blur_k = cfg.get("blur_strength", 35)
            mask_a = cfg.get("mask_alpha", 20) / 100.0
            canvas = cls.generate_blurred_background(cur_img, canvas_w, canvas_h, blur_k, mask_a)
        else:
            canvas = QImage(canvas_w, canvas_h, QImage.Format_ARGB32_Premultiplied)
            canvas.fill(QColor(bg_hex))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        card_rect = QRectF(img_x, img_y, orig_w, orig_h)
        if frame_mode == "surround" and shadow_blur > 0:
            cls.draw_drop_shadow(painter, card_rect, corner_radius, shadow_blur)

        # 绘制原图卡片
        if corner_radius > 0:
            painter.save()
            path = QPainterPath()
            path.addRoundedRect(card_rect, corner_radius, corner_radius)
            painter.setClipPath(path)
            painter.drawImage(card_rect, cur_img)
            painter.restore()
        else:
            painter.drawImage(card_rect, cur_img)

        # 字段与元素渲染
        if bar_h > 0 or top_header_h > 0:
            margin_x = max(20, int(ref_dim * 0.035))

            cam_str = exif_data.get("camera", "Camera")
            lens_str = exif_data.get("lens", "Lens")
            focal_str = exif_data.get("focal", "--")
            ap_str = exif_data.get("aperture", "--")
            shutter_str = exif_data.get("shutter", "--")
            iso_str = exif_data.get("iso", "--")
            date_str = exif_data.get("date", "--")
            brand = cls.detect_brand(cam_str)

            # 毛玻璃背景下顶栏文字智能转为高亮亮白并带轻微微阴影，防止暗部隐形
            if bg_type == "blur":
                t_col = QColor("#FFFFFF")
                sub_col = QColor("#E4E4E7")
            else:
                t_col = QColor(text_hex)
                sub_col = t_col.lighter(150) if t_col.lightness() < 128 else t_col.darker(150)

            # ==========================================================
            # 分支 A: 小红书胶囊药丸风 (深度绑定 Liquid Glass 磨砂高光质感)
            # ==========================================================
            if style_layout == "capsule_pills":
                # 1. 顶部展签: Logo + 机身名 (带毛玻璃防隐形投影)
                if top_header_h > 0:
                    badge_h = int(top_header_h * 0.55)
                    badge_w = int(badge_h * 2.4)
                    custom_p = cfg.get("logo_custom_path", "")

                    f_head = QFont("Segoe UI")
                    f_head.setPixelSize(max(11, int(ref_dim * 0.022)))
                    f_head.setBold(True)
                    painter.setFont(f_head)

                    cam_txt = f"|   {cam_str}"
                    fm = painter.fontMetrics()
                    cam_txt_w = fm.horizontalAdvance(cam_txt)

                    total_header_w = badge_w + 14 + cam_txt_w
                    start_hx = (canvas_w - total_header_w) / 2.0
                    header_y = max(6.0, (img_y - top_header_h) / 2.0)

                    logo_rect = QRectF(start_hx, header_y + (top_header_h - badge_h) / 2.0, badge_w, badge_h)
                    drawn_w = cls.draw_brand_logo(painter, brand, logo_rect, is_dark_bg, custom_p)

                    actual_txt_x = start_hx + (drawn_w if drawn_w > 0 else badge_w) + 12
                    txt_rect = QRectF(actual_txt_x, header_y, cam_txt_w + 50, top_header_h)

                    if bg_type == "blur":
                        painter.setPen(QColor(0, 0, 0, 160))
                        painter.drawText(txt_rect.adjusted(1, 1, 1, 1), Qt.AlignVCenter | Qt.AlignLeft, cam_txt)

                    painter.setPen(t_col)
                    painter.drawText(txt_rect, Qt.AlignVCenter | Qt.AlignLeft, cam_txt)

                # 2. 底部 4 颗独立药丸 (默认 2px 边框 + 绑定 Liquid Glass 液体磨砂与 Alpha 透明度)
                if bar_h > 0:
                    foc_num = focal_str.replace("mm", "").strip() or "--"
                    ap_num = ap_str.replace("f/", "").strip() or "--"
                    iso_num = iso_str.replace("ISO ", "").replace("ISO", "").strip() or "--"
                    sh_num = shutter_str.replace("s", "").strip() or "--"

                    pills = [
                        (foc_num, "mm"),
                        (ap_num, "F"),
                        (iso_num, "ISO"),
                        (sh_num, "S")
                    ]

                    pill_scale = cfg.get("pill_size_scale", 100) / 100.0
                    base_pill_h = min(int(bar_h * 0.52), int(ref_dim * 0.075))
                    pill_h = max(18, int(base_pill_h * pill_scale))
                    pill_w = int(pill_h * 1.85)
                    pill_gap = int(pill_w * 0.22)
                    total_w = 4 * pill_w + 3 * pill_gap
                    start_x = (canvas_w - total_w) / 2.0
                    pill_y = bar_top + (bar_h - pill_h) / 2.0

                    # 核心绑定：默认 2px 边框，透光度从配置读取
                    border_w = max(0, cfg.get("pill_border_width", 2))
                    border_hex = cfg.get("pill_border_color", "#D4D4D8")
                    pill_bg_hex = cfg.get("pill_bg_color", "#FFFFFF")
                    pill_txt_hex = cfg.get("pill_text_color", "#18181B")
                    pill_alpha_pct = cfg.get("pill_alpha", 75) / 100.0

                    # 自动深度绑定 Liquid Glass 模式：只要是毛玻璃/透明底色或透光度 < 1.0 自动触发
                    is_liquid_mode = (bg_type in ("blur", "transparent")) or (pill_alpha_pct < 0.99)

                    for idx, (val, unit) in enumerate(pills):
                        px = start_x + idx * (pill_w + pill_gap)
                        pr = QRectF(px, pill_y, pill_w, pill_h)

                        painter.save()
                        if is_liquid_mode:
                            # 1. 悬浮微发光外阴影
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QColor(0, 0, 0, int(45 * (1.0 - pill_alpha_pct * 0.3))))
                            painter.drawRoundedRect(pr.adjusted(-1, 2, 1, 4), 7, 7)

                            # 2. 液体半透明透光底色 (由 pill_alpha 动态控制)
                            pill_base_col = QColor(pill_bg_hex)
                            pill_base_col.setAlpha(int(255 * pill_alpha_pct))
                            painter.setBrush(QBrush(pill_base_col))

                            # 3. 苹果标志性 1px/2px 物理高光微边缘
                            if border_w > 0:
                                if bg_type == "blur":
                                    highlight_col = QColor(255, 255, 255, min(255, int(200 * pill_alpha_pct + 55)))
                                else:
                                    highlight_col = QColor(border_hex)
                                painter.setPen(QPen(highlight_col, float(border_w)))
                            else:
                                painter.setPen(Qt.NoPen)
                            painter.drawRoundedRect(pr, 6, 6)
                        else:
                            # 常规纯色实心卡片
                            painter.setPen(QPen(QColor(border_hex), border_w) if border_w > 0 else Qt.NoPen)
                            painter.setBrush(QBrush(QColor(pill_bg_hex)))
                            painter.drawRoundedRect(pr, 6, 6)

                        # 药丸数值 (大字加粗)
                        f_val = QFont("Segoe UI")
                        f_val.setPixelSize(max(8, int(pill_h * 0.38)))
                        f_val.setBold(True)
                        painter.setFont(f_val)
                        painter.setPen(QColor(pill_txt_hex))
                        painter.drawText(QRectF(px, pill_y + 2, pill_w, pill_h * 0.52), Qt.AlignCenter, val)

                        # 药丸单位
                        f_unit = QFont("Segoe UI")
                        f_unit.setPixelSize(max(6, int(pill_h * 0.25)))
                        painter.setFont(f_unit)
                        painter.setPen(QColor("#71717A") if not is_liquid_mode else QColor("#4B5563"))
                        painter.drawText(QRectF(px, pill_y + pill_h * 0.50, pill_w, pill_h * 0.42), Qt.AlignCenter, unit)
                        painter.restore()

            # ==========================================================
            # 分支 B: 5 大独立图层系统 (适用于你的 4 套定制预设)
            # ==========================================================
            else:
                if bar_h > 0:
                    logo_rendered_w = 0
                    if cfg.get("logo_show", True):
                        logo_size_pct = cfg.get("logo_size", 50) / 100.0
                        logo_h = max(16, int(bar_h * 0.65 * logo_size_pct))
                        logo_w = int(logo_h * 2.5)

                        logo_align = cfg.get("logo_align", "left")
                        off_x = int(cfg.get("logo_offset_x", 0) * scale_k)
                        off_y = int(cfg.get("logo_offset_y", 0) * scale_k)

                        if logo_align == "center":
                            base_lx = bar_x + (bar_w - logo_w) / 2.0
                        elif logo_align == "right":
                            base_lx = bar_x + bar_w - margin_x - logo_w
                        else:
                            base_lx = bar_x + margin_x

                        logo_rect = QRectF(base_lx + off_x, bar_top + (bar_h - logo_h) / 2.0 + off_y, logo_w, logo_h)
                        custom_p = cfg.get("logo_custom_path", "")
                        logo_rendered_w = cls.draw_brand_logo(painter, brand, logo_rect, is_dark_bg, custom_p)

                    def _render_text_layer(layer_key: str, default_str: str, default_y_ratio: float):
                        if not cfg.get(f"{layer_key}_show", True):
                            return

                        raw_tpl = cfg.get(f"{layer_key}_text", default_str)
                        text = (raw_tpl.replace("{brand}", brand)
                                       .replace("{camera}", cam_str)
                                       .replace("{lens}", lens_str)
                                       .replace("{focal}", focal_str)
                                       .replace("{aperture}", ap_str)
                                       .replace("{shutter}", shutter_str)
                                       .replace("{iso}", iso_str)
                                       .replace("{date}", date_str)
                                       .replace("|", "  |  "))
                        if not text.strip():
                            return

                        align = cfg.get(f"{layer_key}_align", "left")
                        font_fam = cfg.get(f"{layer_key}_font", "Segoe UI")
                        font_size_pt = cfg.get(f"{layer_key}_size", 14)
                        bold = cfg.get(f"{layer_key}_bold", False)

                        off_x = int(cfg.get(f"{layer_key}_offset_x", 0) * scale_k)
                        off_y = int(cfg.get(f"{layer_key}_offset_y", 0) * scale_k)

                        base_y = bar_top + bar_h * default_y_ratio + off_y

                        f = QFont(font_fam)
                        calc_px = max(8, int(ref_dim * 0.02 * (font_size_pt / 14.0)))
                        f.setPixelSize(calc_px)
                        f.setBold(bold)
                        painter.setFont(f)

                        col = QColor(text_hex)
                        if bg_type == "blur":
                            col = QColor("#FFFFFF")
                        if layer_key in ("lens", "date"):
                            col = col.lighter(150) if col.lightness() < 128 else col.darker(150)
                        painter.setPen(col)

                        fm = painter.fontMetrics()
                        tw = fm.horizontalAdvance(text)

                        left_start_margin = margin_x
                        if align == "left" and cfg.get("logo_show", True) and cfg.get("logo_align", "left") == "left":
                            left_start_margin += max(logo_rendered_w, int(margin_x * 0.5)) + 12

                        if align == "center":
                            draw_x = bar_x + (bar_w - tw) / 2.0 + off_x
                        elif align == "right":
                            draw_x = bar_x + bar_w - margin_x - tw + off_x
                        else:
                            draw_x = bar_x + left_start_margin + off_x

                        painter.drawText(QPointF(draw_x, base_y), text)

                    _render_text_layer("camera", "{camera}", 0.44)
                    _render_text_layer("lens", "{lens}", 0.78)
                    _render_text_layer("params", "{focal}  {aperture}  {shutter}  {iso}", 0.44)
                    _render_text_layer("date", "{date}", 0.78)

        painter.end()

        canvas_rot = cfg.get("canvas_rotation", 0)
        if canvas_rot != 0:
            canvas = canvas.transformed(QTransform().rotate(canvas_rot))

        return canvas


# ==============================================================================
# 后台批量加框工作线程
# ==============================================================================
class BatchWatermarkWorker(QThread):
    sig_progress = Signal(int, int, str)
    sig_finished = Signal(dict)

    def __init__(self, src_dir: Path, out_dir: Path, cfg: dict):
        super().__init__()
        self.src_dir = src_dir
        self.out_dir = out_dir
        self.cfg = cfg
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def run(self):
        t0 = time.time()
        from album import VALID_IMAGE_EXTS, RAW_IMAGE_EXTS, load_photo_qimage, parse_photo_exif

        valid_exts = VALID_IMAGE_EXTS | RAW_IMAGE_EXTS
        files = [f for f in self.src_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        total = len(files)
        success = 0
        failed = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)

        for idx, f in enumerate(files, 1):
            if self._is_canceled:
                break
            self.sig_progress.emit(idx, total, f.name)
            try:
                img = load_photo_qimage(str(f))
                if img.isNull():
                    failed += 1
                    continue
                ex = parse_photo_exif(str(f))
                rendered = WatermarkEngine.render(img, ex, self.cfg)
                out_path = self.out_dir / (f.stem + "_framed.jpg")
                rendered.save(str(out_path), "JPG", 95)
                success += 1
            except Exception:
                failed += 1

        self.sig_finished.emit({
            "total": total,
            "success": success,
            "failed": failed,
            "elapsed": time.time() - t0,
            "out_dir": str(self.out_dir)
        })


# ==============================================================================
# 主组件: 参数画框水印独立工作台
# ==============================================================================
class WatermarkPageWidget(QWidget):
    OFFICIAL_PRESETS = {
        "classic_white_bar": {
            "name": "⭐ 经典白底 (双侧对齐)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "bottom_bar",
            "style_layout": "standard",
            "ratio_mode": "free",
            "bar_height_ratio": 12,
            "padding_h_ratio": 0,
            "padding_v_ratio": 0,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "solid",
            "bg_color": "#FFFFFF",
            "text_color": "#121214",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": True,
            "logo_align": "left",
            "logo_custom_path": "",
            "logo_size": 75,
            "logo_offset_x": 0,
            "logo_offset_y": 0,
            "camera_show": True,
            "camera_text": "{camera}",
            "camera_align": "left",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 18,
            "camera_offset_x": 0,
            "camera_offset_y": 12,
            "lens_show": True,
            "lens_text": "{lens}",
            "lens_align": "left",
            "lens_font": "Segoe UI",
            "lens_bold": False,
            "lens_size": 11,
            "lens_offset_x": 0,
            "lens_offset_y": -12,
            "params_show": True,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "right",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 16,
            "params_offset_x": 0,
            "params_offset_y": 12,
            "date_show": True,
            "date_text": "{date}",
            "date_align": "right",
            "date_font": "Segoe UI",
            "date_bold": False,
            "date_size": 10,
            "date_offset_x": 0,
            "date_offset_y": -13
        },
        "pill_white": {
            "name": "💊 药丸纯白 (小红书展签胶囊)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "surround",
            "style_layout": "capsule_pills",
            "ratio_mode": "free",
            "bar_height_ratio": 14,
            "padding_h_ratio": 5,
            "padding_v_ratio": 5,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "solid",
            "bg_color": "#FFFFFF",
            "text_color": "#18181B",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": False,
            "logo_align": "center",
            "logo_custom_path": "",
            "logo_size": 75,
            "logo_offset_x": 0,
            "logo_offset_y": 0,
            "pill_alpha": 100,
            "pill_size_scale": 100,
            "pill_border_width": 2,
            "pill_border_color": "#D4D4D8",
            "pill_bg_color": "#FFFFFF",
            "pill_text_color": "#18181B",
            "camera_show": False,
            "camera_text": "{brand}  |  {camera}",
            "camera_align": "center",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 18,
            "camera_offset_x": 0,
            "camera_offset_y": 0,
            "lens_show": False,
            "lens_text": "{lens}",
            "lens_align": "center",
            "lens_font": "Segoe UI",
            "lens_bold": False,
            "lens_size": 11,
            "lens_offset_x": 0,
            "lens_offset_y": 0,
            "params_show": False,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "center",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 16,
            "params_offset_x": 0,
            "params_offset_y": 0,
            "date_show": False,
            "date_text": "{date}",
            "date_align": "center",
            "date_font": "Segoe UI",
            "date_bold": False,
            "date_size": 10,
            "date_offset_x": 0,
            "date_offset_y": 0
        },
        "pill_blur": {
            "name": "🥛 药丸透明 (毛玻璃液体胶囊)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "surround",
            "style_layout": "capsule_pills",
            "ratio_mode": "free",
            "bar_height_ratio": 14,
            "padding_h_ratio": 5,
            "padding_v_ratio": 5,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "blur",
            "bg_color": "#FFFFFF",
            "text_color": "#18181B",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": False,
            "logo_align": "center",
            "logo_custom_path": "",
            "logo_size": 75,
            "logo_offset_x": 0,
            "logo_offset_y": 0,
            "pill_alpha": 29,
            "pill_size_scale": 100,
            "pill_border_width": 2,
            "pill_border_color": "#D4D4D8",
            "pill_bg_color": "#FFFFFF",
            "pill_text_color": "#18181B",
            "camera_show": False,
            "camera_text": "{brand}  |  {camera}",
            "camera_align": "center",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 18,
            "camera_offset_x": 0,
            "camera_offset_y": 0,
            "lens_show": False,
            "lens_text": "{lens}",
            "lens_align": "center",
            "lens_font": "Segoe UI",
            "lens_bold": False,
            "lens_size": 11,
            "lens_offset_x": 0,
            "lens_offset_y": 0,
            "params_show": False,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "center",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 16,
            "params_offset_x": 0,
            "params_offset_y": 0,
            "date_show": False,
            "date_text": "{date}",
            "date_align": "center",
            "date_font": "Segoe UI",
            "date_bold": False,
            "date_size": 10,
            "date_offset_x": 0,
            "date_offset_y": 0
        },
        "poster_white_2_3": {
            "name": "🖼️ 竖屏海报 (2:3 官方展台)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "surround",
            "style_layout": "standard",
            "ratio_mode": "2:3",
            "bar_height_ratio": 13,
            "padding_h_ratio": 4,
            "padding_v_ratio": 4,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "solid",
            "bg_color": "#FFFFFF",
            "text_color": "#000000",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": True,
            "logo_align": "center",
            "logo_custom_path": "",
            "logo_size": 81,
            "logo_offset_x": 0,
            "logo_offset_y": -36,
            "camera_show": True,
            "camera_text": "{camera}",
            "camera_align": "center",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 29,
            "camera_offset_x": 0,
            "camera_offset_y": 105,
            "lens_show": True,
            "lens_text": "{lens}",
            "lens_align": "center",
            "lens_font": "Segoe UI",
            "lens_bold": True,
            "lens_size": 19,
            "lens_offset_x": 0,
            "lens_offset_y": 74,
            "params_show": False,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "right",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 15,
            "params_offset_x": 37,
            "params_offset_y": -4,
            "date_show": False,
            "date_text": "{date}",
            "date_align": "right",
            "date_font": "Segoe UI",
            "date_bold": True,
            "date_size": 14,
            "date_offset_x": 32,
            "date_offset_y": 2
        },
        "blur_frame_3_2": {
            "name": "📸 透明画框 (3:2 弥散毛玻璃)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "surround",
            "style_layout": "standard",
            "ratio_mode": "3:2",
            "bar_height_ratio": 13,
            "padding_h_ratio": 4,
            "padding_v_ratio": 4,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "blur",
            "bg_color": "#FFFFFF",
            "text_color": "#FFFFFF",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": True,
            "logo_align": "center",
            "logo_custom_path": "",
            "logo_size": 81,
            "logo_offset_x": 0,
            "logo_offset_y": -11,
            "camera_show": True,
            "camera_text": "{camera}",
            "camera_align": "center",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 21,
            "camera_offset_x": 0,
            "camera_offset_y": 92,
            "lens_show": True,
            "lens_text": "{lens}",
            "lens_align": "left",
            "lens_font": "Segoe UI",
            "lens_bold": False,
            "lens_size": 12,
            "lens_offset_x": 0,
            "lens_offset_y": 0,
            "params_show": True,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "left",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 15,
            "params_offset_x": 0,
            "params_offset_y": 20,
            "date_show": True,
            "date_text": "{date}",
            "date_align": "right",
            "date_font": "Segoe UI",
            "date_bold": True,
            "date_size": 13,
            "date_offset_x": 0,
            "date_offset_y": 0
        },
        "blur_minimal_2_3": {
            "name": "📱 透明竖屏 (2:3 锁屏流)",
            "image_rotation": 0,
            "canvas_rotation": 0,
            "frame_mode": "surround",
            "style_layout": "standard",
            "ratio_mode": "2:3",
            "bar_height_ratio": 13,
            "padding_h_ratio": 4,
            "padding_v_ratio": 4,
            "corner_radius": 0,
            "shadow_blur": 0,
            "bg_type": "blur",
            "bg_color": "#FFFFFF",
            "text_color": "#FFFFFF",
            "blur_strength": 35,
            "mask_alpha": 20,
            "logo_show": True,
            "logo_align": "center",
            "logo_custom_path": "",
            "logo_size": 80,
            "logo_offset_x": 0,
            "logo_offset_y": -8,
            "camera_show": True,
            "camera_text": "{camera}",
            "camera_align": "center",
            "camera_font": "Segoe UI",
            "camera_bold": True,
            "camera_size": 20,
            "camera_offset_x": 0,
            "camera_offset_y": 124,
            "lens_show": False,
            "lens_text": "{lens}",
            "lens_align": "center",
            "lens_font": "Segoe UI",
            "lens_bold": True,
            "lens_size": 14,
            "lens_offset_x": 0,
            "lens_offset_y": 0,
            "params_show": False,
            "params_text": "{focal}  {aperture}  {shutter}  {iso}",
            "params_align": "left",
            "params_font": "Segoe UI",
            "params_bold": True,
            "params_size": 15,
            "params_offset_x": 0,
            "params_offset_y": 0,
            "date_show": False,
            "date_text": "{date}",
            "date_align": "right",
            "date_font": "Segoe UI",
            "date_bold": True,
            "date_size": 12,
            "date_offset_x": 0,
            "date_offset_y": 0
        }
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_photo_path = ""
        self.source_image = QImage()
        self.image_rotation = 0
        self.canvas_rotation = 0
        self.worker = None

        self.cached_exif = {
            "camera": "Canon EOS R50",
            "lens": "RF50mm F1.8 STM",
            "focal": "50mm",
            "aperture": "f/1.8",
            "shutter": "1/250s",
            "iso": "ISO 100",
            "date": "2026.07.11 13:36",
        }

        self._init_ui()
        self.apply_config(self.OFFICIAL_PRESETS["classic_white_bar"])

    def _init_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; width: 1px; }")

        control_deck = QWidget()
        deck_layout = QVBoxLayout(control_deck)
        deck_layout.setContentsMargins(8, 8, 8, 8)
        deck_layout.setSpacing(10)

        # ---------------- 1. 模板管理 ----------------
        grp_preset = QGroupBox("📁 1. 模板预设与参数存档")
        p_l = QVBoxLayout(grp_preset)
        p_l.setSpacing(6)

        h_tpl = QHBoxLayout()
        h_tpl.setSpacing(6)
        self.combo_presets = QComboBox()
        for k, v in self.OFFICIAL_PRESETS.items():
            self.combo_presets.addItem(v["name"], k)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        h_tpl.addWidget(self.combo_presets, stretch=1)

        self.btn_reset_preset = QPushButton("↩️ 重置")
        self.btn_reset_preset.setFixedWidth(68)
        self.btn_reset_preset.setToolTip("恢复为当前预设的默认值")
        self.btn_reset_preset.clicked.connect(self._reset_current_preset)
        h_tpl.addWidget(self.btn_reset_preset)
        p_l.addLayout(h_tpl)

        h_actions = QHBoxLayout()
        h_actions.setSpacing(6)
        self.btn_export_preset = QPushButton("📤 导出预设")
        self.btn_export_preset.setToolTip("将当前调好的全部参数导出为独立 JSON 文件")
        self.btn_export_preset.clicked.connect(self._export_preset)
        h_actions.addWidget(self.btn_export_preset)

        self.btn_import_preset = QPushButton("📥 导入预设")
        self.btn_import_preset.setToolTip("加载之前导出的 JSON 预设文件并即时生效")
        self.btn_import_preset.clicked.connect(self._import_preset)
        h_actions.addWidget(self.btn_import_preset)
        p_l.addLayout(h_actions)
        deck_layout.addWidget(grp_preset)

        # ---------------- 2. 画布与几何形态 ----------------
        grp_geo = QGroupBox("📐 2. 画布形态、画幅比例与旋转")
        g_l = QVBoxLayout(grp_geo)
        g_l.setSpacing(8)

        h_rot = QHBoxLayout()
        h_rot.addWidget(QLabel("旋转控制:"))
        self.btn_rot_img = QPushButton("🔄 原图旋转")
        self.btn_rot_img.clicked.connect(self._rotate_image_step)
        h_rot.addWidget(self.btn_rot_img)
        self.btn_rot_frame = QPushButton("🔄 画框旋转")
        self.btn_rot_frame.clicked.connect(self._rotate_canvas_step)
        h_rot.addWidget(self.btn_rot_frame)
        g_l.addLayout(h_rot)

        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("骨架形态:"))
        self.combo_frame_mode = QComboBox()
        self.combo_frame_mode.addItem("仅底部外延参数栏 (底栏型)", "bottom_bar")
        self.combo_frame_mode.addItem("四周画布包围 (悬浮画心型)", "surround")
        h_mode.addWidget(self.combo_frame_mode, stretch=1)
        g_l.addLayout(h_mode)

        h_ratio = QHBoxLayout()
        h_ratio.addWidget(QLabel("画面画幅:"))
        self.combo_ratio = QComboBox()
        self.combo_ratio.addItem("自由比例 (长宽随留白展开)", "free")
        self.combo_ratio.addItem("保持原图比例 (仅底边等比外延)", "native")
        self.combo_ratio.addItem("3:2 (相机原生 135 全幅横屏)", "3:2")
        self.combo_ratio.addItem("4:3 (标准数码横画幅)", "4:3")
        self.combo_ratio.addItem("16:9 (经典宽屏横画幅)", "16:9")
        self.combo_ratio.addItem("16:10 (影视平板横屏)", "16:10")
        self.combo_ratio.addItem("21:9 (电影超宽荧幕)", "21:9")
        self.combo_ratio.addItem("2:3 (相机原生竖构图)", "2:3")
        self.combo_ratio.addItem("3:4 (经典人像竖屏)", "3:4")
        self.combo_ratio.addItem("4:5 (小红书竖屏黄金流)", "4:5")
        self.combo_ratio.addItem("9:16 (手机全屏壁纸)", "9:16")
        self.combo_ratio.addItem("9:20 (现代全面屏超长壁纸)", "9:20")
        self.combo_ratio.addItem("1:1 (正方形画廊)", "1:1")
        h_ratio.addWidget(self.combo_ratio, stretch=1)
        g_l.addLayout(h_ratio)

        self.row_bar_height = SliderInputRow("底边高度比例:", 0, 80, 12, "%")
        g_l.addWidget(self.row_bar_height)

        self.row_padding_h = SliderInputRow("左右留白边距:", 0, 60, 0, "%")
        g_l.addWidget(self.row_padding_h)

        self.row_padding_v = SliderInputRow("上下留白边距:", 0, 60, 0, "%")
        g_l.addWidget(self.row_padding_v)

        self.row_radius = SliderInputRow("原图边角圆角:", 0, 150, 0, "px")
        g_l.addWidget(self.row_radius)

        self.row_shadow = SliderInputRow("悬浮弥散外阴影:", 0, 80, 0, "px")
        g_l.addWidget(self.row_shadow)

        deck_layout.addWidget(grp_geo)

        # ---------------- 3. 底色与材质 ----------------
        grp_bg = QGroupBox("🎨 3. 底色与材质")
        bg_l = QVBoxLayout(grp_bg)
        bg_l.setSpacing(8)

        h_bg_type = QHBoxLayout()
        h_bg_type.addWidget(QLabel("底色类型:"))
        self.combo_bg_type = QComboBox()
        self.combo_bg_type.addItem("纯色背景 (调色盘/Hex)", "solid")
        self.combo_bg_type.addItem("毛玻璃背景 (提取原图自适应模糊)", "blur")
        self.combo_bg_type.addItem("纯透明背景 (导出 PNG 备用)", "transparent")
        h_bg_type.addWidget(self.combo_bg_type, stretch=1)
        bg_l.addLayout(h_bg_type)

        self.color_bg = ColorPickerRow("背景基准色:", "#FFFFFF")
        bg_l.addWidget(self.color_bg)

        self.color_text = ColorPickerRow("全局文字主色:", "#121214")
        bg_l.addWidget(self.color_text)

        self.row_blur = SliderInputRow("毛玻璃模糊强度:", 5, 120, 35)
        bg_l.addWidget(self.row_blur)

        self.row_mask_alpha = SliderInputRow("模糊底图压暗遮罩:", 0, 80, 20, "%")
        bg_l.addWidget(self.row_mask_alpha)

        deck_layout.addWidget(grp_bg)

        # ---------------- 4. 品牌 Logo 独立定制面板 ----------------
        grp_logo = QGroupBox("🏷️ 4. 品牌 Logo 定制 (支持导入本地 PNG/SVG)")
        l_layout = QVBoxLayout(grp_logo)
        l_layout.setSpacing(8)

        h_l_top = QHBoxLayout()
        self.cb_logo_show = QCheckBox("显示品牌 Logo")
        self.cb_logo_show.setChecked(True)
        h_l_top.addWidget(self.cb_logo_show)

        h_l_top.addWidget(QLabel("吸附对齐:"))
        self.combo_logo_align = QComboBox()
        self.combo_logo_align.addItem("靠左对齐 (Left)", "left")
        self.combo_logo_align.addItem("居中对齐 (Center)", "center")
        self.combo_logo_align.addItem("靠右对齐 (Right)", "right")
        h_l_top.addWidget(self.combo_logo_align)
        l_layout.addLayout(h_l_top)

        h_custom_logo = QHBoxLayout()
        self.edit_logo_custom = QLineEdit()
        self.edit_logo_custom.setPlaceholderText("留空则自动按机身匹配，或点击浏览自选...")
        h_custom_logo.addWidget(self.edit_logo_custom)
        btn_browse_logo = QPushButton("浏览...")
        btn_browse_logo.clicked.connect(self._browse_custom_logo)
        h_custom_logo.addWidget(btn_browse_logo)
        btn_clear_logo = QPushButton("清除")
        btn_clear_logo.clicked.connect(lambda: self.edit_logo_custom.setText(""))
        h_custom_logo.addWidget(btn_clear_logo)
        l_layout.addLayout(h_custom_logo)

        self.row_logo_size = SliderInputRow("图标尺寸大小:", 10, 100, 75, "%")
        l_layout.addWidget(self.row_logo_size)

        self.row_logo_off_x = SliderInputRow("水平位置偏移:", -200, 200, 0, "px")
        l_layout.addWidget(self.row_logo_off_x)

        self.row_logo_off_y = SliderInputRow("垂直位置偏移:", -100, 100, 0, "px")
        l_layout.addWidget(self.row_logo_off_y)

        deck_layout.addWidget(grp_logo)

        # ---------------- 辅助函数：构造 4 大文本元素控制面板 ----------------
        def _build_text_layer_box(title: str, default_text: str, default_align: str, default_size: int, default_bold: bool):
            grp = QGroupBox(title)
            v = QVBoxLayout(grp)
            v.setSpacing(7)

            h1 = QHBoxLayout()
            cb = QCheckBox("显示")
            cb.setChecked(True)
            h1.addWidget(cb)
            edit = QLineEdit(default_text)
            edit.setStyleSheet("background-color: #141418; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 6px; color: #ffffff; font-size: 11px;")
            h1.addWidget(edit, stretch=1)
            v.addLayout(h1)

            h2 = QHBoxLayout()
            h2.addWidget(QLabel("吸附:"))
            combo_align = QComboBox()
            combo_align.addItem("左", "left")
            combo_align.addItem("中", "center")
            combo_align.addItem("右", "right")
            idx = combo_align.findData(default_align)
            if idx >= 0: combo_align.setCurrentIndex(idx)
            h2.addWidget(combo_align)

            h2.addWidget(QLabel("字体:"))
            combo_font = QComboBox()
            combo_font.addItem("Segoe UI", "Segoe UI")
            combo_font.addItem("Consolas", "Consolas")
            combo_font.addItem("Arial", "Arial")
            combo_font.addItem("Microsoft YaHei", "Microsoft YaHei")
            h2.addWidget(combo_font, stretch=1)

            cb_bold = QCheckBox("加粗")
            cb_bold.setChecked(default_bold)
            h2.addWidget(cb_bold)
            v.addLayout(h2)

            row_size = SliderInputRow("字体大小:", 8, 50, default_size, "pt")
            v.addWidget(row_size)

            row_ox = SliderInputRow("水平偏移 (X):", -300, 300, 0, "px")
            v.addWidget(row_ox)

            row_oy = SliderInputRow("垂直偏移 (Y):", -150, 150, 0, "px")
            v.addWidget(row_oy)

            deck_layout.addWidget(grp)
            return {
                "cb": cb, "edit": edit, "align": combo_align, "font": combo_font,
                "bold": cb_bold, "size": row_size, "ox": row_ox, "oy": row_oy
            }

        self.ui_camera = _build_text_layer_box("📷 5. 相机型号 (Camera)", "{camera}", "left", 18, True)
        self.ui_lens = _build_text_layer_box("🔍 6. 镜头型号 (Lens)", "{lens}", "left", 11, False)
        self.ui_params = _build_text_layer_box("⚡ 7. 拍摄参数 (Params)", "{focal}  {aperture}  {shutter}  {iso}", "right", 16, True)
        self.ui_date = _build_text_layer_box("📅 8. 拍摄日期 (Date / 自定义格式)", "{date}", "right", 10, False)

        # ---------------- 9. 胶囊药丸与 Liquid Glass 质感面板 ----------------
        grp_pills = QGroupBox("💊 9. 胶囊药丸与 Liquid Glass 质感定制")
        pill_v = QVBoxLayout(grp_pills)
        pill_v.setSpacing(7)

        # 核心改动：用透光度滑块取代生硬开关，自动深度绑定毛玻璃/透明模式！
        self.row_pill_alpha = SliderInputRow("药丸透光度 (Alpha):", 10, 100, 75, "%")
        pill_v.addWidget(self.row_pill_alpha)

        self.row_pill_scale = SliderInputRow("药丸整体缩放:", 50, 180, 100, "%")
        pill_v.addWidget(self.row_pill_scale)

        # 默认 2px 黄金边框粗细
        self.row_pill_border_w = SliderInputRow("药丸边框粗细:", 0, 6, 2, "px")
        pill_v.addWidget(self.row_pill_border_w)

        self.color_pill_border = ColorPickerRow("药丸边框基准色:", "#D4D4D8")
        pill_v.addWidget(self.color_pill_border)

        self.color_pill_bg = ColorPickerRow("药丸卡片底色:", "#FFFFFF")
        pill_v.addWidget(self.color_pill_bg)

        self.color_pill_txt = ColorPickerRow("药丸文字主色:", "#18181B")
        pill_v.addWidget(self.color_pill_txt)

        hint_pill = QLabel("💡 提示：底色选毛玻璃或透明度<100%时，自动激活苹果 Liquid Glass 磨砂高光质感。")
        hint_pill.setStyleSheet("color: #71717a; font-size: 11px;")
        pill_v.addWidget(hint_pill)

        deck_layout.addWidget(grp_pills)

        # ---------------- 10. 动作与批量导出 ----------------
        grp_act = QGroupBox("⚡ 10. 渲染交付")
        act_l = QVBoxLayout(grp_act)
        act_l.setSpacing(8)

        self.btn_load_file = QPushButton("📂 打开单张本地照片...")
        self.btn_load_file.clicked.connect(self._browse_single_photo)
        act_l.addWidget(self.btn_load_file)

        self.btn_save_single = QPushButton("💾 保存当前画框大图 (原生满血分辨率)")
        self.btn_save_single.setObjectName("PrimaryBtn")
        self.btn_save_single.setStyleSheet("QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 8px; font-size: 13px; } QPushButton:hover { background-color: #0369a1; }")
        self.btn_save_single.clicked.connect(self._save_current_rendered)
        act_l.addWidget(self.btn_save_single)

        self.btn_batch = QPushButton("📦 批量文件夹加框输出...")
        self.btn_batch.clicked.connect(self._open_batch_dialog)
        act_l.addWidget(self.btn_batch)

        deck_layout.addWidget(grp_act)
        deck_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(420)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #141418; }")
        scroll_area.setWidget(control_deck)
        splitter.addWidget(scroll_area)

        # ======================================================================
        # 右侧: 所见即所得大图视口
        # ======================================================================
        self.viewer = WatermarkPreviewViewer()
        self.viewer.sig_image_dropped.connect(self.load_photo)
        splitter.addWidget(self.viewer)

        splitter.setSizes([480, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter)
        self._bind_events()

    def _bind_events(self):
        triggers = [
            self.row_bar_height.valueChanged, self.row_padding_h.valueChanged, self.row_padding_v.valueChanged,
            self.row_radius.valueChanged, self.row_shadow.valueChanged,
            self.row_blur.valueChanged, self.row_mask_alpha.valueChanged,
            self.color_bg.colorChanged, self.color_text.colorChanged,
            self.combo_frame_mode.currentIndexChanged, self.combo_ratio.currentIndexChanged,
            self.combo_bg_type.currentIndexChanged,
            self.cb_logo_show.toggled, self.combo_logo_align.currentIndexChanged,
            self.edit_logo_custom.textChanged, self.row_logo_size.valueChanged,
            self.row_logo_off_x.valueChanged, self.row_logo_off_y.valueChanged,
            # 药丸与透光度触发器
            self.row_pill_alpha.valueChanged, self.row_pill_scale.valueChanged, self.row_pill_border_w.valueChanged,
            self.color_pill_border.colorChanged, self.color_pill_bg.colorChanged, self.color_pill_txt.colorChanged
        ]

        for layer in (self.ui_camera, self.ui_lens, self.ui_params, self.ui_date):
            triggers.extend([
                layer["cb"].toggled, layer["edit"].textChanged,
                layer["align"].currentIndexChanged, layer["font"].currentIndexChanged,
                layer["bold"].toggled, layer["size"].valueChanged,
                layer["ox"].valueChanged, layer["oy"].valueChanged
            ])

        for sig in triggers:
            sig.connect(self._render_preview)

    def _rotate_image_step(self):
        self.image_rotation = (self.image_rotation + 90) % 360
        self._render_preview()

    def _rotate_canvas_step(self):
        self.canvas_rotation = (self.canvas_rotation + 90) % 360
        self._render_preview()

    def _browse_custom_logo(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择自定义 Logo 图片", "", "图片文件 (*.png *.jpg *.jpeg *.svg);;所有文件 (*.*)")
        if f:
            self.edit_logo_custom.setText(f)

    def _browse_single_photo(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "选择用于加框预览的照片", "",
            "图片文件 (*.jpg *.jpeg *.png *.webp *.cr3 *.arw *.nef *.dng);;所有文件 (*.*)"
        )
        if f:
            self.load_photo(f)

    def load_photo(self, file_path: str):
        self.current_photo_path = file_path
        p = Path(file_path)
        if not p.exists():
            return

        from album import load_photo_qimage, parse_photo_exif
        img = load_photo_qimage(file_path)
        if not img.isNull():
            self.source_image = img
            ex = parse_photo_exif(file_path)
            if ex:
                self.cached_exif.update(ex)
            self.apply_config(self.OFFICIAL_PRESETS["classic_white_bar"])

    def export_config(self) -> dict:
        cfg = {
            "image_rotation": self.image_rotation,
            "canvas_rotation": self.canvas_rotation,
            "frame_mode": self.combo_frame_mode.currentData(),
            "style_layout": getattr(self, "current_style_layout", "standard"),
            "ratio_mode": self.combo_ratio.currentData(),
            "bar_height_ratio": self.row_bar_height.value(),
            "padding_h_ratio": self.row_padding_h.value(),
            "padding_v_ratio": self.row_padding_v.value(),
            "corner_radius": self.row_radius.value(),
            "shadow_blur": self.row_shadow.value(),
            "bg_type": self.combo_bg_type.currentData(),
            "bg_color": self.color_bg.colorHex(),
            "text_color": self.color_text.colorHex(),
            "blur_strength": self.row_blur.value(),
            "mask_alpha": self.row_mask_alpha.value(),
            "logo_show": self.cb_logo_show.isChecked(),
            "logo_align": self.combo_logo_align.currentData(),
            "logo_custom_path": self.edit_logo_custom.text().strip(),
            "logo_size": self.row_logo_size.value(),
            "logo_offset_x": self.row_logo_off_x.value(),
            "logo_offset_y": self.row_logo_off_y.value(),
            # 药丸配置
            "pill_alpha": self.row_pill_alpha.value(),
            "pill_size_scale": self.row_pill_scale.value(),
            "pill_border_width": self.row_pill_border_w.value(),
            "pill_border_color": self.color_pill_border.colorHex(),
            "pill_bg_color": self.color_pill_bg.colorHex(),
            "pill_text_color": self.color_pill_txt.colorHex(),
        }

        mapping = [("camera", self.ui_camera), ("lens", self.ui_lens),
                   ("params", self.ui_params), ("date", self.ui_date)]
        for key, ui in mapping:
            cfg[f"{key}_show"] = ui["cb"].isChecked()
            cfg[f"{key}_text"] = ui["edit"].text()
            cfg[f"{key}_align"] = ui["align"].currentData()
            cfg[f"{key}_font"] = ui["font"].currentData()
            cfg[f"{key}_bold"] = ui["bold"].isChecked()
            cfg[f"{key}_size"] = ui["size"].value()
            cfg[f"{key}_offset_x"] = ui["ox"].value()
            cfg[f"{key}_offset_y"] = ui["oy"].value()

        return cfg

    def apply_config(self, cfg: dict):
        self.blockSignals(True)
        self.image_rotation = cfg.get("image_rotation", 0)
        self.canvas_rotation = cfg.get("canvas_rotation", 0)
        self.current_style_layout = cfg.get("style_layout", "standard")

        idx = self.combo_frame_mode.findData(cfg.get("frame_mode", "bottom_bar"))
        if idx >= 0: self.combo_frame_mode.setCurrentIndex(idx)

        idx = self.combo_ratio.findData(cfg.get("ratio_mode", "free"))
        if idx >= 0: self.combo_ratio.setCurrentIndex(idx)

        self.row_bar_height.setValue(cfg.get("bar_height_ratio", 12))
        self.row_padding_h.setValue(cfg.get("padding_h_ratio", 0))
        self.row_padding_v.setValue(cfg.get("padding_v_ratio", 0))
        self.row_radius.setValue(cfg.get("corner_radius", 0))
        self.row_shadow.setValue(cfg.get("shadow_blur", 0))

        idx = self.combo_bg_type.findData(cfg.get("bg_type", "solid"))
        if idx >= 0: self.combo_bg_type.setCurrentIndex(idx)

        self.color_bg.setColor(cfg.get("bg_color", "#FFFFFF"))
        self.color_text.setColor(cfg.get("text_color", "#121214"))
        self.row_blur.setValue(cfg.get("blur_strength", 35))
        self.row_mask_alpha.setValue(cfg.get("mask_alpha", 20))

        self.cb_logo_show.setChecked(cfg.get("logo_show", True))
        idx = self.combo_logo_align.findData(cfg.get("logo_align", "left"))
        if idx >= 0: self.combo_logo_align.setCurrentIndex(idx)
        self.edit_logo_custom.setText(cfg.get("logo_custom_path", ""))
        self.row_logo_size.setValue(cfg.get("logo_size", 75))
        self.row_logo_off_x.setValue(cfg.get("logo_offset_x", 0))
        self.row_logo_off_y.setValue(cfg.get("logo_offset_y", 0))

        # 药丸配置恢复 (默认 2px 边框与 75% 透光度)
        self.row_pill_alpha.setValue(cfg.get("pill_alpha", 75))
        self.row_pill_scale.setValue(cfg.get("pill_size_scale", 100))
        self.row_pill_border_w.setValue(cfg.get("pill_border_width", 2))
        self.color_pill_border.setColor(cfg.get("pill_border_color", "#D4D4D8"))
        self.color_pill_bg.setColor(cfg.get("pill_bg_color", "#FFFFFF"))
        self.color_pill_txt.setColor(cfg.get("pill_text_color", "#18181B"))

        mapping = [("camera", self.ui_camera), ("lens", self.ui_lens),
                   ("params", self.ui_params), ("date", self.ui_date)]
        for key, ui in mapping:
            ui["cb"].setChecked(cfg.get(f"{key}_show", True))
            ui["edit"].setText(cfg.get(f"{key}_text", ""))
            idx = ui["align"].findData(cfg.get(f"{key}_align", "left"))
            if idx >= 0: ui["align"].setCurrentIndex(idx)
            idx = ui["font"].findData(cfg.get(f"{key}_font", "Segoe UI"))
            if idx >= 0: ui["font"].setCurrentIndex(idx)
            ui["bold"].setChecked(cfg.get(f"{key}_bold", False))
            ui["size"].setValue(cfg.get(f"{key}_size", 14))
            ui["ox"].setValue(cfg.get(f"{key}_offset_x", 0))
            ui["oy"].setValue(cfg.get(f"{key}_offset_y", 0))

        self.blockSignals(False)
        self._render_preview()

    def _on_preset_selected(self):
        k = self.combo_presets.currentData()
        if k in self.OFFICIAL_PRESETS:
            self.apply_config(self.OFFICIAL_PRESETS[k])
        elif hasattr(self, "_custom_presets") and k in self._custom_presets:
            self.apply_config(self._custom_presets[k])

    def _reset_current_preset(self):
        k = self.combo_presets.currentData()
        if k in self.OFFICIAL_PRESETS:
            self.apply_config(self.OFFICIAL_PRESETS[k])
        elif hasattr(self, "_custom_presets") and k in self._custom_presets:
            self.apply_config(self._custom_presets[k])
        else:
            self.apply_config(self.OFFICIAL_PRESETS["classic_white_bar"])

    def _export_preset(self):
        k = self.combo_presets.currentData()
        default_name = f"preset_{k}.json" if k else "watermark_preset.json"
        f, _ = QFileDialog.getSaveFileName(self, "导出当前预设配置", default_name, "JSON 预设文件 (*.json)")
        if f:
            cfg = self.export_config()
            try:
                Path(f).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
                QMessageBox.information(self, "导出成功", f"画框预设已成功导出至：\n{f}\n\n下次可直接点击「导入预设」随时调用！")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出预设异常：\n{e}")

    def _import_preset(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入外部画框预设", "", "JSON 预设文件 (*.json);;所有文件 (*.*)")
        if f:
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    stem = Path(f).stem
                    custom_key = f"custom_{stem}_{int(time.time())}"
                    if not hasattr(self, "_custom_presets"):
                        self._custom_presets = {}
                    self._custom_presets[custom_key] = data
                    self.combo_presets.addItem(f"📄 {stem} (已导入)", custom_key)
                    self.combo_presets.setCurrentIndex(self.combo_presets.count() - 1)
                    self.apply_config(data)
                    QMessageBox.information(self, "导入成功", f"预设「{stem}」已载入并即时应用！")
                else:
                    QMessageBox.warning(self, "格式错误", "无效的预设配置文件！")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"无法解析预设文件：\n{e}")

    def _render_preview(self):
        if self.source_image.isNull():
            self.viewer.setImage(QImage())
            return

        preview_img = self.source_image
        max_d = max(preview_img.width(), preview_img.height())
        if max_d > 1600:
            scale = 1600.0 / max_d
            preview_img = preview_img.scaled(
                int(preview_img.width() * scale), int(preview_img.height() * scale),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        cfg = self.export_config()
        rendered = WatermarkEngine.render(preview_img, self.cached_exif, cfg)
        self.viewer.setImage(rendered)

    def _save_current_rendered(self):
        if self.source_image.isNull():
            QMessageBox.warning(self, "未载入照片", "请先打开或拖拽一张照片到视口！")
            return

        out, _ = QFileDialog.getSaveFileName(
            self, "保存原生全分辨率画框成片",
            f"{Path(self.current_photo_path).stem}_framed.jpg" if self.current_photo_path else "framed_output.jpg",
            "JPEG 照片 (*.jpg);;PNG 图像 (*.png)"
        )
        if out:
            cfg = self.export_config()
            full_res = WatermarkEngine.render(self.source_image, self.cached_exif, cfg)
            fmt = "PNG" if out.lower().endswith(".png") else "JPG"
            full_res.save(out, fmt, 95)
            QMessageBox.information(self, "保存完成", f"高清满血成片已保存至：\n{out}")

    def _open_batch_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("📦 批量文件夹加框自动化交付")
        dlg.resize(460, 260)
        dlg.setStyleSheet("""
            QDialog { background-color: #18181b; color: #f4f4f5; }
            QLineEdit { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 4px; padding: 6px; color: #fff; }
            QPushButton { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 4px; padding: 6px 14px; font-weight: bold; }
            QPushButton#ActionBtn { background-color: #0284c7; color: white; border: none; padding: 8px 20px; }
        """)
        v = QVBoxLayout(dlg)
        v.setSpacing(12)

        v.addWidget(QLabel("📂 1. 原图输入文件夹 (支持包含 JPG / CR3 / ARW / NEF):"))
        h1 = QHBoxLayout()
        edit_src = QLineEdit()
        h1.addWidget(edit_src)
        btn_s = QPushButton("浏览...")
        btn_s.clicked.connect(lambda: edit_src.setText(QFileDialog.getExistingDirectory(dlg, "选择输入目录")))
        h1.addWidget(btn_s)
        v.addLayout(h1)

        v.addWidget(QLabel("📁 2. 加框成品输出目录:"))
        h2 = QHBoxLayout()
        edit_out = QLineEdit()
        h2.addWidget(edit_out)
        btn_o = QPushButton("浏览...")
        btn_o.clicked.connect(lambda: edit_out.setText(QFileDialog.getExistingDirectory(dlg, "选择输出目录")))
        h2.addWidget(btn_o)
        v.addLayout(h2)

        v.addStretch()

        btns = QHBoxLayout()
        btns.addStretch()
        btn_run = QPushButton("⚡ 开始批量加框")
        btn_run.setObjectName("ActionBtn")
        btns.addWidget(btn_run)
        v.addLayout(btns)

        def _start_run():
            s_dir = edit_src.text().strip()
            o_dir = edit_out.text().strip()
            if not s_dir or not os.path.exists(s_dir):
                QMessageBox.warning(dlg, "路径无效", "请指定有效的输入目录！")
                return
            if not o_dir:
                QMessageBox.warning(dlg, "路径无效", "请指定输出目录！")
                return
            dlg.accept()

            prog = QDialog(self)
            prog.setWindowTitle("正在执行批量加框...")
            prog.resize(420, 110)
            pv = QVBoxLayout(prog)
            lbl_st = QLabel("正在初始化渲染管线...")
            pv.addWidget(lbl_st)
            pbar = QProgressBar()
            pv.addWidget(pbar)
            prog.show()

            self.worker = BatchWatermarkWorker(Path(s_dir), Path(o_dir), self.export_config())
            self.worker.sig_progress.connect(lambda cur, tot, name: (
                pbar.setValue(int(cur / tot * 100)),
                lbl_st.setText(f"加框渲染中 [{cur}/{tot}]: {name}")
            ))
            self.worker.sig_finished.connect(lambda res: (
                prog.close(),
                QMessageBox.information(self, "批量完成", f"批量处理完成！\n成功: {res['success']} 张\n失败: {res['failed']} 张\n耗时: {res['elapsed']:.2f} 秒\n输出目录: {res['out_dir']}")
            ))
            self.worker.start()

        btn_run.clicked.connect(_start_run)
        dlg.exec()