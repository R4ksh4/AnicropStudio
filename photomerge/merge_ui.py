import math
import os
import time
import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .pipeline import MergeParams, PhotoMergePipeline, imwrite_unicode


ACS_QSS = """
QWidget#PhotoMergePage {
    background-color: #0d0d11;
    color: #e4e4e7;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
}
QWidget {
    background-color: transparent;
    color: #e4e4e7;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
}
QWidget#RightSidebar {
    background-color: #0d0d11;
}
QFrame#Card {
    background-color: #18181e;
    border: 1px solid #27272f;
    border-radius: 8px;
}
QLabel#CardTitle {
    color: #a1a1aa;
    font-weight: bold;
    font-size: 11px;
}
QPushButton {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 12px;
    color: #e4e4e7;
}
QPushButton:hover {
    background-color: #323238;
    border-color: #38bdf8;
    color: #ffffff;
}
QComboBox {
    background-color: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    padding: 4px 10px;
    color: #e4e4e7;
}
QComboBox:hover {
    border-color: #38bdf8;
}
QComboBox QAbstractItemView {
    background-color: #18181e;
    border: 1px solid #3f3f46;
    selection-background-color: #0284c7;
    color: #e4e4e7;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #27272f;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 2px;
}
QCheckBox {
    spacing: 6px;
    color: #d4d4d8;
}
QScrollArea {
    border: none;
}
"""


class PreviewWorker(QThread):
    sig_done = Signal(object, str, float, float, object)
    sig_err = Signal(str)

    def __init__(self, pipeline: PhotoMergePipeline, params: MergeParams, quality: str, view_mode: str):
        super().__init__()
        self.pipeline = pipeline
        self.params = params
        self.quality = quality
        self.view_mode = view_mode

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            if self.view_mode == "depth":
                d_u8 = (self.pipeline.depth_proxy * 255.0).astype(np.uint8)
                out_u8 = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)
            elif self.view_mode == "raw":
                out_u8 = self.pipeline.render_raw_overlay(self.params)
            else:
                out_u8 = self.pipeline.render(self.params, quality=self.quality)
            dt = (time.perf_counter() - t0) * 1000.0
            self.sig_done.emit(
                out_u8,
                self.quality,
                dt,
                self.pipeline.auto_azimuth_deg,
                self.pipeline.auto_light_color_rgb,
            )
        except Exception as e:
            self.sig_err.emit(str(e))


class ExportWorker(QThread):
    sig_progress = Signal(int, str)
    sig_finished = Signal(str, int, int, float)
    sig_error = Signal(str)

    def __init__(self, pipeline: PhotoMergePipeline, params: MergeParams, save_path: str):
        super().__init__()
        self.pipeline = pipeline
        self.params = params
        self.save_path = save_path

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            full_u8 = self.pipeline.render(
                self.params,
                quality="full",
                progress_cb=lambda pct, msg: self.sig_progress.emit(pct, msg),
            )
            self.sig_progress.emit(98, "正在写入磁盘并注入原片 EXIF/ICC 元数据...")
            imwrite_unicode(self.save_path, full_u8, quality=96, orig_file_path=self.pipeline.bg_path)
            dt = time.perf_counter() - t0
            h, w = full_u8.shape[:2]
            self.sig_progress.emit(100, "导出完成！")
            self.sig_finished.emit(self.save_path, w, h, dt)
        except Exception as e:
            self.sig_error.emit(str(e))


class ExportProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        self.setFixedSize(460, 220)

        self.angle = 0
        self.elapsed_sec = 0.0
        self.status_text = "正在启动全分辨率无损渲染管线..."
        self.progress_pct = 5

        self.timer_rot = QTimer(self)
        self.timer_rot.timeout.connect(self._on_tick)
        self.timer_rot.start(16)

    def _on_tick(self) -> None:
        self.angle = (self.angle + 5) % 360
        self.elapsed_sec += 0.016
        self.update()

    def set_status(self, pct: int, text: str) -> None:
        self.progress_pct = max(self.progress_pct, pct)
        self.status_text = text
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        p.setBrush(QColor("#141418"))
        p.setPen(QPen(QColor("#0284c7"), 1.8))
        p.drawRoundedRect(rect, 10, 10)

        cx, cy = self.width() / 2.0, 62.0
        radius = 32.0
        p.save()
        p.translate(cx, cy)
        p.rotate(self.angle)
        grad = QConicalGradient(0, 0, 0)
        grad.setColorAt(0.0, QColor("#38bdf8"))
        grad.setColorAt(0.5, QColor("#0284c7"))
        grad.setColorAt(1.0, QColor(2, 132, 199, 0))
        p.setPen(QPen(QBrush(grad), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(-radius, -radius, radius * 2, radius * 2), 0, 360 * 16)
        p.restore()

        p.setBrush(QColor("#38bdf8"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 4.5, 4.5)

        p.setFont(QFont("Microsoft YaHei UI", 11, QFont.Weight.Bold))
        p.setPen(QColor("#f4f4f5"))
        p.drawText(QRectF(16, 108, self.width() - 32, 24), Qt.AlignmentFlag.AlignCenter, "⚡ 正在导出全分辨率无损成片")

        p.setFont(QFont("Microsoft YaHei UI", 9))
        p.setPen(QColor("#38bdf8"))
        p.drawText(QRectF(20, 134, self.width() - 40, 20), Qt.AlignmentFlag.AlignCenter, self.status_text)

        bar_x, bar_y, bar_w, bar_h = 36, 164, self.width() - 72, 8
        p.setBrush(QColor("#27272a"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 4, 4)

        fill_w = (bar_w * self.progress_pct) / 100.0
        p.setBrush(QColor("#0284c7"))
        p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 4, 4)

        p.setFont(QFont("Consolas", 9))
        p.setPen(QColor("#a1a1aa"))
        p.drawText(
            QRectF(36, 180, self.width() - 72, 20),
            Qt.AlignmentFlag.AlignCenter,
            f"{self.progress_pct}% | 已耗时: {self.elapsed_sec:.1f} 秒",
        )


class AppleGlassDock(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        r = rect.height() / 2.0

        grad = QLinearGradient(rect.left(), rect.top(), rect.left(), rect.bottom())
        grad.setColorAt(0.0, QColor(36, 36, 48, 225))
        grad.setColorAt(1.0, QColor(18, 18, 26, 240))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1.0))
        painter.drawRoundedRect(QRectF(rect), r, r)

        hi = QRectF(rect).adjusted(r, 1, -r, 0)
        hi_grad = QLinearGradient(hi.left(), hi.top(), hi.right(), hi.top())
        hi_grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        hi_grad.setColorAt(0.5, QColor(255, 255, 255, 75))
        hi_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setPen(QPen(QBrush(hi_grad), 1.0))
        painter.drawLine(QPointF(hi.left(), rect.top() + 1.5), QPointF(hi.right(), rect.top() + 1.5))


class LightCompassWidget(QWidget):
    angleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(88, 88)
        self._azimuth_deg = -60.0
        self._dragging = False
        self.setToolTip("拖动光点调整主光源方向")

    def set_azimuth(self, deg: float) -> None:
        self._azimuth_deg = float(deg)
        self.update()

    def _update_from_pos(self, pos) -> None:
        cx, cy = self.width() / 2.0, self.height() / 2.0
        dx, dy = pos.x() - cx, pos.y() - cy
        if abs(dx) < 2 and abs(dy) < 2:
            return
        deg = math.degrees(math.atan2(dy, dx))
        self._azimuth_deg = deg
        self.update()
        self.angleChanged.emit(deg)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_pos(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._update_from_pos(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = min(cx, cy) - 9.0

        p.setBrush(QBrush(QColor(18, 18, 24, 210)))
        p.setPen(QPen(QColor(63, 63, 70), 1.5))
        p.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        p.setPen(QPen(QColor(50, 50, 58), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(int(cx - r + 4), int(cy), int(cx + r - 4), int(cy))
        p.drawLine(int(cx), int(cy - r + 4), int(cx), int(cy + r - 4))

        rad = math.radians(self._azimuth_deg)
        px = cx + math.cos(rad) * (r - 2)
        py = cy + math.sin(rad) * (r - 2)

        p.setPen(QPen(QColor(56, 189, 248), 2.0))
        p.drawLine(int(cx), int(cy), int(px), int(py))
        p.setBrush(QBrush(QColor(250, 204, 21)))
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawEllipse(int(px - 5), int(py - 5), 10, 10)


class CanvasView(QGraphicsView):
    characterMoved = Signal(float, float)
    characterDragFinished = Signal()
    characterScaled = Signal(float)
    brushPainted = Signal(float, float, float, float, float, bool, float, float, str)
    brushFinished = Signal()
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QBrush(QColor("#050507")))

        self._canon_w = 1
        self._canon_h = 1
        self._pos_x = 0.5
        self._pos_y = 0.69
        self._scale = 0.62
        self._char_aspect = 1.0

        self.tool_mode = "move"
        self.brush_radius_norm = 0.055
        self.brush_hardness = 0.50
        self.brush_opacity = 0.65
        self.brush_mode = "blur"  # blur | sharp | rim_erase
        self._mouse_vp = QPointF(-100, -100)

        self._dragging_char = False
        self._brushing = False
        self._panning = False
        self._last_mouse = None

    def set_canonical_size(self, w: int, h: int) -> None:
        self._canon_w = max(1, w)
        self._canon_h = max(1, h)
        self._scene.setSceneRect(0, 0, self._canon_w, self._canon_h)

    def set_image(self, img_bgr_u8: np.ndarray, reset_view: bool = False) -> None:
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        if self._canon_w <= 1:
            self.set_canonical_size(w, h)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self._pixmap_item.setScale(self._canon_w / float(w))
        self._scene.setSceneRect(0, 0, self._canon_w, self._canon_h)
        if reset_view:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_view(self) -> None:
        if self._canon_w > 1:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def sync_char_transform(self, pos_x: float, pos_y: float, scale: float, char_aspect: float) -> None:
        self._pos_x = pos_x
        self._pos_y = pos_y
        self._scale = scale
        self._char_aspect = char_aspect

    def _scene_to_char_uv(self, scene_pt: QPointF) -> tuple[float, float]:
        target_h = self._canon_h * self._scale
        target_w = target_h * self._char_aspect
        cx, cy = self._pos_x * self._canon_w, self._pos_y * self._canon_h
        x1, y1 = cx - target_w / 2.0, cy - target_h / 2.0
        u = (scene_pt.x() - x1) / max(1.0, target_w)
        v = (scene_pt.y() - y1) / max(1.0, target_h)
        return u, v

    def _scene_to_bg_uv(self, scene_pt: QPointF) -> tuple[float, float]:
        return scene_pt.x() / max(1.0, float(self._canon_w)), scene_pt.y() / max(1.0, float(self._canon_h))

    def _emit_brush_at(self, scene_pt: QPointF, is_erase: bool) -> None:
        u, v = self._scene_to_char_uv(scene_pt)
        u_bg, v_bg = self._scene_to_bg_uv(scene_pt)
        self.brushPainted.emit(
            u,
            v,
            u_bg,
            v_bg,
            self.brush_radius_norm,
            is_erase,
            self.brush_hardness,
            self.brush_opacity,
            self.brush_mode,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._last_mouse = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            if self.tool_mode == "brush":
                self._brushing = True
                is_erase = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self._emit_brush_at(scene_pt, is_erase)
            else:
                self._dragging_char = True
                self._last_mouse = scene_pt
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.viewport().update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_vp = event.position()
        if self.tool_mode == "brush":
            self.viewport().update()

        if self._panning and self._last_mouse is not None:
            delta = event.position() - self._last_mouse
            self._last_mouse = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self._brushing:
            scene_pt = self.mapToScene(event.position().toPoint())
            is_erase = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._emit_brush_at(scene_pt, is_erase)
            return
        if self._dragging_char and self._last_mouse is not None and self._canon_w > 1:
            curr_scene = self.mapToScene(event.position().toPoint())
            dx = (curr_scene.x() - self._last_mouse.x()) / self._canon_w
            dy = (curr_scene.y() - self._last_mouse.y()) / self._canon_h
            self._last_mouse = curr_scene
            self._pos_x = max(-0.2, min(1.2, self._pos_x + dx))
            self._pos_y = max(-0.2, min(1.2, self._pos_y + dy))
            self.viewport().update()
            self.characterMoved.emit(self._pos_x, self._pos_y)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        was_char = self._dragging_char
        was_brush = self._brushing
        self._dragging_char = False
        self._brushing = False
        self._panning = False
        self.setCursor(Qt.CursorShape.CrossCursor if self.tool_mode == "brush" else Qt.CursorShape.ArrowCursor)
        self.viewport().update()
        if was_char:
            self.characterDragFinished.emit()
        if was_brush:
            self.brushFinished.emit()
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self._dragging_char and self._canon_w > 1:
            target_h = self._canon_h * self._scale
            target_w = target_h * self._char_aspect
            cx, cy = self._pos_x * self._canon_w, self._pos_y * self._canon_h
            box = QRectF(cx - target_w / 2.0, cy - target_h / 2.0, target_w, target_h)
            painter.setPen(QPen(QColor("#38bdf8"), max(2.0, 2.0 / max(0.1, self.transform().m11())), Qt.PenStyle.DashLine))
            painter.drawRect(box)

        if self.tool_mode == "brush" and self._mouse_vp.x() >= 0:
            painter.resetTransform()
            if self.brush_mode == "sharp":
                ref_dim = max(self._canon_w, self._canon_h)
                col = QColor("#4ade80")
            else:
                target_h_scene = self._canon_h * self._scale
                target_w_scene = target_h_scene * self._char_aspect
                ref_dim = max(target_w_scene, target_h_scene)
                col = QColor("#facc15") if self.brush_mode == "rim_erase" else QColor("#38bdf8")

            r_vp = max(4.0, self.brush_radius_norm * ref_dim * self.transform().m11())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(col, 1.8))
            painter.setBrush(QColor(col.red(), col.green(), col.blue(), int(20 + self.brush_opacity * 35)))
            painter.drawEllipse(self._mouse_vp, r_vp, r_vp)

            # 内圈虚线指示当前画笔硬度范围
            r_hard = max(1.5, r_vp * self.brush_hardness)
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._mouse_vp, r_hard, r_hard)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = 0.03 if delta > 0 else -0.03
            self._scale = max(0.10, min(1.50, self._scale + step))
            self.characterScaled.emit(self._scale)
        else:
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self.scale(factor, factor)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)


class CompactSlider(QWidget):
    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float = 0.01,
        suffix: str = "",
        formatter=None,
    ):
        super().__init__()
        self._scale = 1.0 / step
        self._suffix = suffix
        self._formatter = formatter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        top = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #d4d4d8;")
        self._val_lbl = QLabel()
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._val_lbl.setStyleSheet("color: #38bdf8; font-family: Consolas; font-size: 11px;")
        top.addWidget(lbl)
        top.addWidget(self._val_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(round(min_val * self._scale)), int(round(max_val * self._scale)))
        self._slider.valueChanged.connect(self._on_changed)
        self._slider.sliderReleased.connect(self.sliderReleased.emit)

        layout.addLayout(top)
        layout.addWidget(self._slider)
        self.set_value(default, emit=False)

    def _on_changed(self, iv: int) -> None:
        val = iv / self._scale
        self._update_label(val)
        self.valueChanged.emit(val)

    def _update_label(self, val: float) -> None:
        if self._formatter is not None:
            self._val_lbl.setText(self._formatter(val))
        elif self._scale <= 1.0:
            self._val_lbl.setText(f"{int(round(val))}{self._suffix}")
        else:
            self._val_lbl.setText(f"{val:.2f}{self._suffix}")

    
    def value(self) -> float:
        return self._slider.value() / self._scale

    def set_value(self, val: float, emit: bool = False) -> None:
        if not emit:
            self._slider.blockSignals(True)
        self._slider.setValue(int(round(val * self._scale)))
        self._update_label(val)
        if not emit:
            self._slider.blockSignals(False)


class PhotoMergePageWidget(QWidget):
    sig_back_requested = Signal()

    def __init__(self, parent=None, mode: str = "anime"):
        super().__init__(parent)
        self.work_mode = mode if mode in ("anime", "cosplay") else "anime"
        self.canvas = None
        self.dock = None
        self.brush_dock = None
        self.setObjectName("PhotoMergePage")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.params = MergeParams(work_mode=self.work_mode)
        self.pipeline = PhotoMergePipeline()

        self._view_mode = "merged"
        self._prev_mode_before_space = "merged"
        self._pending_render = None
        self._preview_worker = None
        self._export_worker = None
        self._export_dlg = None
        self._first_view = True

        self._fast_timer = QTimer(self)
        self._fast_timer.setSingleShot(True)
        self._fast_timer.setInterval(50)
        self._fast_timer.timeout.connect(lambda: self._enqueue_render("fast"))

        self._hq_timer = QTimer(self)
        self._hq_timer.setSingleShot(True)
        self._hq_timer.setInterval(250)
        self._hq_timer.timeout.connect(lambda: self._enqueue_render("proxy"))

        self.setStyleSheet(ACS_QSS)
        self._init_ui()
        self._apply_work_mode_defaults(self.work_mode, render=False)
        if self.work_mode == "anime":
            self._try_load_samples()
        

    def _init_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; width: 1px; }")
        root_l = QVBoxLayout(self)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.setSpacing(0)
        root_l.addWidget(splitter)

        left_wrap = QWidget()
        left_layout = QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setFixedHeight(38)
        top_bar.setStyleSheet("background-color: #121216; border-bottom: 1px solid #27272f; padding: 0 10px;")
        tb_l = QHBoxLayout(top_bar)
        tb_l.setContentsMargins(6, 0, 6, 0)
        tb_l.setSpacing(8)

        self.btn_back = QPushButton("‹ 返回输入")
        self.btn_back.setToolTip("返回 AI 魔法消除&溶图准备台")
        self.btn_back.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        tb_l.addWidget(self.btn_back)
    
        self.btn_work_mode = QPushButton("📸 场照溶图模式" if self.work_mode == "cosplay" else "🌌 二次元溶图模式")
        self.btn_work_mode.setToolTip("点击在「二次元溶图」与「场照光学溶图」之间切换默认参数策略")
        self.btn_work_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_work_mode.setStyleSheet(
            "QPushButton { background-color: #1e293b; border: 1px solid #38bdf8; border-radius: 4px; color: #38bdf8; font-weight: bold; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #0284c7; color: #ffffff; }"
        )
        self.btn_work_mode.clicked.connect(self._toggle_work_mode)
        tb_l.addWidget(self.btn_work_mode)

        self.btn_mode_merged = QPushButton("✨ 溶图成片 [1]")
        self.btn_mode_raw = QPushButton("📷 直贴对比 [2 / 空格]")
        self.btn_mode_depth = QPushButton("🧊 深度透视 [3]")
        for b in (self.btn_mode_merged, self.btn_mode_raw, self.btn_mode_depth):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            tb_l.addWidget(b)
        tb_l.addStretch()
        hint_top = "就绪 | 场照模式：拖入单张原片即可自动抠人重塑" if self.work_mode == "cosplay" else "就绪 | 拖入底图与透明PNG即可开始"
        self.lbl_stats = QLabel(hint_top)
        self.lbl_stats.setStyleSheet("color: #38bdf8; font-family: Consolas; font-size: 11px;")
        tb_l.addWidget(self.lbl_stats)

        self.btn_mode_merged.clicked.connect(lambda: self._set_view_mode("merged"))
        self.btn_mode_raw.clicked.connect(lambda: self._set_view_mode("raw"))
        self.btn_mode_depth.clicked.connect(lambda: self._set_view_mode("depth"))
        left_layout.addWidget(top_bar)

        self.canvas = CanvasView()
        self.canvas.installEventFilter(self)
        self.canvas.characterMoved.connect(self._on_canvas_move)
        self.canvas.characterDragFinished.connect(lambda: self._hq_timer.start())
        self.canvas.characterScaled.connect(self._on_canvas_scale)
        self.canvas.brushPainted.connect(self._on_brush_paint)
        self.canvas.brushFinished.connect(lambda: self._hq_timer.start())
        self.canvas.filesDropped.connect(self._on_files_dropped)
        left_layout.addWidget(self.canvas, stretch=1)

        # 画笔专属悬浮设置条（选中 [B] 画笔时自动在 Dock 上方弹出）
        self.brush_dock = AppleGlassDock(self.canvas)
        self.brush_dock.setVisible(False)
        bd_l = QHBoxLayout(self.brush_dock)
        bd_l.setContentsMargins(14, 4, 14, 4)
        bd_l.setSpacing(10)

        bd_l.addWidget(QLabel("笔刷:"))
        self.combo_brush_mode = QComboBox()
        self.combo_brush_mode.addItem("🖌️ 模糊柔化", "blur")
        self.combo_brush_mode.addItem("🔍 清晰度恢复", "sharp")
        self.combo_brush_mode.addItem("✨ 轮廓光擦除", "rim_erase")
        self.combo_brush_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_brush_mode.currentIndexChanged.connect(self._on_brush_cfg_changed)
        bd_l.addWidget(self.combo_brush_mode)

        bd_l.addWidget(QLabel("大小:"))
        self.sl_brush_sz = QSlider(Qt.Orientation.Horizontal)
        self.sl_brush_sz.setRange(15, 200)
        self.sl_brush_sz.setValue(55)
        self.sl_brush_sz.setFixedWidth(85)
        self.sl_brush_sz.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sl_brush_sz.valueChanged.connect(self._on_brush_cfg_changed)
        bd_l.addWidget(self.sl_brush_sz)

        bd_l.addWidget(QLabel("硬度:"))
        self.sl_brush_hard = QSlider(Qt.Orientation.Horizontal)
        self.sl_brush_hard.setRange(0, 100)
        self.sl_brush_hard.setValue(50)
        self.sl_brush_hard.setFixedWidth(85)
        self.sl_brush_hard.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sl_brush_hard.valueChanged.connect(self._on_brush_cfg_changed)
        bd_l.addWidget(self.sl_brush_hard)

        bd_l.addWidget(QLabel("透明度:"))
        self.sl_brush_op = QSlider(Qt.Orientation.Horizontal)
        self.sl_brush_op.setRange(5, 100)
        self.sl_brush_op.setValue(65)
        self.sl_brush_op.setFixedWidth(85)
        self.sl_brush_op.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sl_brush_op.valueChanged.connect(self._on_brush_cfg_changed)
        bd_l.addWidget(self.sl_brush_op)

        self.dock = AppleGlassDock(self.canvas)
        self.dock.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 14);
                border: 1px solid rgba(255, 255, 255, 22);
                border-radius: 16px;
                color: #e4e4e7;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 32);
                color: #ffffff;
            }
        """)
        dock_l = QHBoxLayout(self.dock)
        dock_l.setContentsMargins(14, 6, 14, 6)
        dock_l.setSpacing(8)

        self.btn_tool_move = QPushButton("✋ 移动 [V]")
        self.btn_tool_move.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_tool_move.clicked.connect(lambda: self._set_tool_mode("move"))

        self.btn_tool_brush = QPushButton("🖌️ 笔刷 [B]")
        self.btn_tool_brush.setToolTip("局部柔化 / 清晰度恢复 / 轮廓光擦除；Ctrl+左键 反向擦除；[ / ] 调大小")
        self.btn_tool_brush.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_tool_brush.clicked.connect(lambda: self._set_tool_mode("brush"))

        btn_clear_brush = QPushButton("🧹 清空笔刷")
        btn_clear_brush.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_clear_brush.clicked.connect(self._clear_brush)

        btn_snap = QPushButton("🧲 贴齐底边")
        btn_snap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_snap.clicked.connect(self._snap_to_bottom)

        btn_reset = QPushButton("🔄 全部重置")
        btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_reset.clicked.connect(self._reset_params)

        btn_export = QPushButton("💾 导出成片")
        btn_export.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                color: white;
                border-radius: 16px;
                padding: 6px 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        btn_export.clicked.connect(self._export_full_res)

        for b in (self.btn_tool_move, self.btn_tool_brush, btn_clear_brush, btn_snap, btn_reset, btn_export):
            dock_l.addWidget(b)

        splitter.addWidget(left_wrap)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(370)
        scroll.setObjectName("RightSidebar")

        right_panel = QWidget()
        right_panel.setObjectName("RightSidebar")
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(10)

        # 卡片 1: 素材与主光源
        card_src = QFrame()
        card_src.setObjectName("Card")
        c1 = QVBoxLayout(card_src)
        c1.setContentsMargins(12, 10, 12, 10)
        c1.setSpacing(8)
        t1 = QLabel("🖼️ 1. 素材载入与主光源")
        t1.setObjectName("CardTitle")
        c1.addWidget(t1)

        btn_row = QHBoxLayout()
        btn_bg = QPushButton("导入底图")
        btn_bg.clicked.connect(self._pick_bg)
        btn_ch = QPushButton("导入立绘 (PNG)")
        btn_ch.clicked.connect(self._pick_char)
        btn_row.addWidget(btn_bg)
        btn_row.addWidget(btn_ch)
        c1.addLayout(btn_row)

        self.btn_auto_cutout = QPushButton("📸 单图场照自动抠人并重合")
        self.btn_auto_cutout.setToolTip("直接对当前底图自动抠出人物并 1:1 原位重合")
        self.btn_auto_cutout.setStyleSheet("QPushButton { background-color: #0c4a6e; border: 1px solid #0284c7; color: #38bdf8; font-weight: bold; } QPushButton:hover { background-color: #0284c7; color: #ffffff; }")
        self.btn_auto_cutout.clicked.connect(self._auto_cutout_current_bg)
        c1.addWidget(self.btn_auto_cutout)

        light_row = QHBoxLayout()
        self.compass = LightCompassWidget()
        self.compass.angleChanged.connect(self._on_compass_changed)
        light_ctrl = QVBoxLayout()
        light_ctrl.setSpacing(5)
        self.combo_light_mode = QComboBox()
        self.combo_light_mode.addItems(["自动识别", "自定义", "绑定光效"])
        self.combo_light_mode.currentTextChanged.connect(lambda: self._enqueue_render("proxy"))
        self.lbl_az = QLabel("方位角: -60°")
        self.lbl_az.setStyleSheet("color: #38bdf8; font-family: Consolas; font-size: 11px;")

        color_row = QHBoxLayout()
        self.cb_auto_color = QCheckBox("自动光色")
        self.cb_auto_color.setChecked(True)
        self.cb_auto_color.toggled.connect(lambda: self._enqueue_render("proxy"))
        self.btn_light_color = QPushButton()
        self.btn_light_color.setFixedSize(44, 22)
        self.btn_light_color.setToolTip("点击自定义主光源勾边颜色")
        self.btn_light_color.clicked.connect(self._pick_light_color)
        self._set_swatch(self.btn_light_color, self.params.custom_light_color_rgb)
        color_row.addWidget(self.cb_auto_color)
        color_row.addWidget(self.btn_light_color)
        color_row.addStretch()

        light_ctrl.addWidget(self.combo_light_mode)
        light_ctrl.addWidget(self.lbl_az)
        light_ctrl.addLayout(color_row)
        self._light_color_presets = {
            "日落暖金 (推荐)": (255, 195, 110),
            "浓烈夕阳": (255, 150, 55),
            "古街灯笼": (255, 135, 40),
            "清晨微暖": (255, 228, 185),
            "清冷月光": (165, 205, 255),
            "赛博青蓝": (80, 230, 255),
            "霓虹洋红": (255, 95, 190),
        }
        self.combo_color_preset = QComboBox()
        self.combo_color_preset.addItems(["光色预设...", *self._light_color_presets.keys()])
        self.combo_color_preset.currentTextChanged.connect(self._on_color_preset_changed)
        light_ctrl.addWidget(self.combo_color_preset)
        light_ctrl.addStretch()
        light_row.addWidget(self.compass)
        light_row.addLayout(light_ctrl)
        c1.addLayout(light_row)
        self.sl_az = CompactSlider("光源角度", -180.0, 180.0, self.params.azimuth_deg, 1.0, "°")
        self.sl_az.valueChanged.connect(self._on_az_slider_changed)
        self.sl_az.sliderReleased.connect(lambda: self._hq_timer.start())
        c1.addWidget(self.sl_az)
        rl.addWidget(card_src)

        # 卡片 2: 光学溶图与光影
        card_merge = QFrame()
        card_merge.setObjectName("Card")
        c2 = QVBoxLayout(card_merge)
        c2.setContentsMargins(12, 10, 12, 10)
        c2.setSpacing(3)
        t2 = QLabel("✨ 2. 光学溶图与光影")
        t2.setObjectName("CardTitle")
        c2.addWidget(t2)

        self.sl_scale = CompactSlider("角色大小", 0.10, 2.50, self.params.scale, 0.01)
        self.sl_bg_ev = CompactSlider("背景曝光", -2.50, 2.50, self.params.bg_exposure, 0.05, formatter=lambda v: f"{v:+.2f} EV")
        self.sl_ch_ev = CompactSlider("角色曝光", -2.50, 2.50, self.params.char_exposure, 0.05, formatter=lambda v: f"{v:+.2f} EV")
        sensor_row = QHBoxLayout()
        sensor_row.addWidget(QLabel("相机画幅"))
        self.combo_sensor = QComboBox()
        self.combo_sensor.addItems(["半画幅 (APS-C)", "全画幅", "手机"])
        self.combo_sensor.setCurrentText(self.params.sensor_type)
        self.combo_sensor.currentTextChanged.connect(lambda: self._enqueue_render("proxy"))
        sensor_row.addWidget(self.combo_sensor, stretch=1)
        c2.addLayout(sensor_row)

        self.sl_bokeh = CompactSlider(
            "虚拟光圈",
            0.0,
            1.0,
            self.params.aperture_strength,
            0.01,
            formatter=self._format_bokeh_fstop,
        )
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("边缘引擎"))
        self.combo_edge_mode = QComboBox()
        self.combo_edge_mode.addItem("真人物理边缘", "real")
        self.combo_edge_mode.addItem("二次元赛璐璐边缘", "anime")
        self.combo_edge_mode.currentIndexChanged.connect(self._on_edge_shadow_mode_changed)
        edge_row.addWidget(self.combo_edge_mode, stretch=1)
        c2.addLayout(edge_row)

        sh_row = QHBoxLayout()
        sh_row.addWidget(QLabel("阴影引擎"))
        self.combo_shadow_mode = QComboBox()
        self.combo_shadow_mode.addItem("真人接触压暗", "real")
        self.combo_shadow_mode.addItem("二次元平地投影", "anime")
        self.combo_shadow_mode.currentIndexChanged.connect(self._on_edge_shadow_mode_changed)
        sh_row.addWidget(self.combo_shadow_mode, stretch=1)
        c2.addLayout(sh_row)

        self.sl_dof_tol = CompactSlider("焦平面清晰带", 0.0, 0.35, self.params.dof_tolerance, 0.01)
        self.sl_edge_choke = CompactSlider("边缘微收/扩展", -3.0, 3.0, self.params.edge_choke, 0.2, " px")
        self.sl_feather = CompactSlider("角色边缘羽化", 0.0, 16.0, self.params.edge_feather, 0.2, " px")
        self.sl_rim = CompactSlider("主光源发光勾边", 0.0, 4.00, self.params.rim_strength, 0.02)
        self.sl_light_wrap = CompactSlider("环境溢色包边", 0.0, 1.50, self.params.light_wrap_strength, 0.02)
        self.sl_25d = CompactSlider("2.5D 立体明暗", 0.0, 1.50, self.params.shading_25d_strength, 0.02)
        self.sl_tone = CompactSlider("环境光色融合", 0.0, 1.20, self.params.brightness_match, 0.02)
        self.sl_shadow = CompactSlider("地面阴影浓度", 0.0, 1.50, self.params.shadow_opacity, 0.02)

        for s in (
            self.sl_scale,
            self.sl_bg_ev,
            self.sl_ch_ev,
            self.sl_bokeh,
            self.sl_dof_tol,
            self.sl_edge_choke,
            self.sl_feather,
            self.sl_rim,
            self.sl_light_wrap,
            self.sl_25d,
            self.sl_tone,
            self.sl_shadow,
        ):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            c2.addWidget(s)

        # 真人物理边缘细调子组（切回二次元边缘时自动折叠）
        self.real_edge_box = QWidget()
        re_l = QVBoxLayout(self.real_edge_box)
        re_l.setContentsMargins(0, 2, 0, 0)
        re_l.setSpacing(3)
        self.sl_rim_tex = CompactSlider("轮廓纹理贴合", 0.0, 1.0, self.params.rim_texture_weight, 0.02)
        self.sl_rim_falloff = CompactSlider("轮廓衰减柔化", 0.0, 1.0, self.params.rim_falloff, 0.02)
        self.sl_hair_trans = CompactSlider("发丝物理透光", 0.0, 1.0, self.params.hair_transmission, 0.02)
        self.sl_normal_mix = CompactSlider("衣褶立体混合", 0.0, 1.0, self.params.normal_detail_mix, 0.02)
        for s in (self.sl_rim_tex, self.sl_rim_falloff, self.sl_hair_trans, self.sl_normal_mix):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            re_l.addWidget(s)
        c2.addWidget(self.real_edge_box)
        rl.addWidget(card_merge)

        # 卡片 3: 3 种独立模拟光效 (各自独立参数面板)
        card_fx = QFrame()
        card_fx.setObjectName("Card")
        cfx = QVBoxLayout(card_fx)
        cfx.setContentsMargins(12, 10, 12, 10)
        cfx.setSpacing(6)
        tfx = QLabel("☀️ 3. 氛围光效模拟")
        tfx.setObjectName("CardTitle")
        cfx.addWidget(tfx)

        fx_top = QHBoxLayout()
        self.cb_fx_enable = QCheckBox("启用光效")
        self.cb_fx_enable.setChecked(self.params.light_fx_enabled)
        self.cb_fx_enable.toggled.connect(lambda: self._enqueue_render("proxy"))
        self.combo_fx_type = QComboBox()
        self.combo_fx_type.addItems(["丁达尔光", "聚光灯", "光晕"])
        self.combo_fx_type.currentIndexChanged.connect(self._on_fx_type_changed)
        fx_top.addWidget(self.cb_fx_enable)
        fx_top.addWidget(self.combo_fx_type, stretch=1)

        self.cb_face_protect = QCheckBox("🛡️ 人脸保护")
        self.cb_face_protect.setChecked(self.params.face_protect_enabled)
        self.cb_face_protect.setToolTip("保护面部五官：物理驱散人脸上的灰尘杂质，并柔化减弱面部强光晕")
        self.cb_face_protect.toggled.connect(lambda: self._enqueue_render("proxy"))
        fx_top.addWidget(self.cb_face_protect)

        cfx.addLayout(fx_top)

        fx_layer_row = QHBoxLayout()
        fx_layer_row.addWidget(QLabel("光效图层"))
        self.combo_fx_layer = QComboBox()
        self.combo_fx_layer.addItem("人后体积逆光 (边缘微溢)", "人后体积逆光")
        self.combo_fx_layer.addItem("置于人物后方 (纯人后)", "置于人物后方")
        self.combo_fx_layer.addItem("置于人物前方 (全局覆盖)", "置于人物前方")
        self.combo_fx_layer.currentIndexChanged.connect(self._on_fx_layer_changed)
        fx_layer_row.addWidget(self.combo_fx_layer, stretch=1)
        cfx.addLayout(fx_layer_row)

        self.sl_fx_occ = CompactSlider("光束被人体遮挡", 0.0, 1.0, self.params.fx_occlusion_strength, 0.02)
        self.sl_fx_occ.valueChanged.connect(self._on_slider_moving)
        self.sl_fx_occ.sliderReleased.connect(lambda: self._hq_timer.start())
        cfx.addWidget(self.sl_fx_occ)

        self.sl_fx_spill = CompactSlider("光束漫过人体", 0.0, 1.0, self.params.fx_edge_spill, 0.02)
        self.sl_fx_spill.valueChanged.connect(self._on_slider_moving)
        self.sl_fx_spill.sliderReleased.connect(lambda: self._hq_timer.start())
        cfx.addWidget(self.sl_fx_spill)

        self.sl_fx_x = CompactSlider("发光中心 X 轴", -0.30, 1.30, self.params.fx_pos_x, 0.01)
        self.sl_fx_y = CompactSlider("发光中心 Y 轴", -0.30, 1.30, self.params.fx_pos_y, 0.01)
        self.sl_fx_aim = CompactSlider("光效照射角度", -180.0, 180.0, self.params.fx_aim_deg, 1.0, "°")
        for s in (self.sl_fx_x, self.sl_fx_y, self.sl_fx_aim):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            cfx.addWidget(s)

        self.fx_stack = QStackedWidget()

        # Page 0: 丁达尔光专属参数
        p_tyn = QWidget()
        l_tyn = QVBoxLayout(p_tyn)
        l_tyn.setContentsMargins(0, 0, 0, 0)
        l_tyn.setSpacing(3)
        self.sl_tyn_str = CompactSlider("光线强度", 0.0, 2.50, self.params.tyndall_strength, 0.02)
        self.sl_tyn_cnt = CompactSlider("光束数量", 0.20, 4.00, self.params.tyndall_count, 0.05)
        self.sl_tyn_soft = CompactSlider("边缘柔化", 0.0, 1.50, self.params.tyndall_softness, 0.02)
        self.sl_tyn_len = CompactSlider("光线长度", 0.15, 2.50, self.params.tyndall_length, 0.02)
        for s in (self.sl_tyn_str, self.sl_tyn_cnt, self.sl_tyn_soft, self.sl_tyn_len):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            l_tyn.addWidget(s)
        l_tyn.addStretch()
        self.fx_stack.addWidget(p_tyn)

        # Page 1: 聚光灯专属参数
        p_spot = QWidget()
        l_spot = QVBoxLayout(p_spot)
        l_spot.setContentsMargins(0, 0, 0, 0)
        l_spot.setSpacing(3)
        self.sl_spot_str = CompactSlider("光束强度", 0.0, 3.00, self.params.spot_strength, 0.02)
        self.sl_spot_wid = CompactSlider("光锥宽度", 0.05, 1.50, self.params.spot_width, 0.02)
        self.sl_spot_soft = CompactSlider("边缘柔化", 0.02, 1.50, self.params.spot_softness, 0.02)
        self.sl_spot_dust = CompactSlider("空气灰尘", 0.0, 2.00, self.params.spot_dust, 0.02)
        self.sl_spot_tip = CompactSlider("入射尖端隐藏", 0.0, 2.00, self.params.spot_tip_hide, 0.02)
        for s in (self.sl_spot_str, self.sl_spot_wid, self.sl_spot_soft, self.sl_spot_dust, self.sl_spot_tip):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            l_spot.addWidget(s)
        l_spot.addStretch()
        self.fx_stack.addWidget(p_spot)

        # Page 2: 光晕与太阳光斑专属参数
        p_halo = QWidget()
        l_halo = QVBoxLayout(p_halo)
        l_halo.setContentsMargins(0, 0, 0, 0)
        l_halo.setSpacing(3)
        self.sl_halo_str = CompactSlider("光晕强度", 0.0, 3.00, self.params.halo_strength, 0.02)
        self.sl_halo_rad = CompactSlider("光晕半径", 0.20, 3.00, self.params.halo_radius, 0.02)
        self.sl_halo_wid = CompactSlider("光晕宽度", 0.15, 2.00, self.params.halo_width, 0.02)
        self.sl_halo_core = CompactSlider("入射光核强度", 0.0, 2.00, self.params.halo_core_strength, 0.02)
        self.sl_halo_hide = CompactSlider("光晕反向隐藏", 0.0, 1.50, self.params.halo_arc_hide, 0.02)

        halo_col_row = QHBoxLayout()
        halo_col_row.addWidget(QLabel("光晕过渡色"))
        self.btn_halo_in = QPushButton("内")
        self.btn_halo_in.setFixedSize(44, 22)
        self.btn_halo_in.clicked.connect(lambda: self._pick_fx_color("halo_in"))
        self.btn_halo_out = QPushButton("外")
        self.btn_halo_out.setFixedSize(44, 22)
        self.btn_halo_out.clicked.connect(lambda: self._pick_fx_color("halo_out"))
        self._set_swatch(self.btn_halo_in, self.params.halo_inner_rgb)
        self._set_swatch(self.btn_halo_out, self.params.halo_outer_rgb)
        halo_col_row.addStretch()
        halo_col_row.addWidget(self.btn_halo_in)
        halo_col_row.addWidget(QLabel("→"))
        halo_col_row.addWidget(self.btn_halo_out)

        self.sl_flare_str = CompactSlider("光斑强度", 0.0, 3.00, self.params.flare_strength, 0.02)
        self.sl_flare_sz = CompactSlider("光斑大小", 0.20, 3.50, self.params.flare_size, 0.02)
        self.sl_flare_ang = CompactSlider("光斑角度", -180.0, 180.0, self.params.flare_angle, 1.0, "°")

        flare_meta_row = QHBoxLayout()
        flare_meta_row.addWidget(QLabel("光斑形状"))
        self.combo_flare_shape = QComboBox()
        self.combo_flare_shape.addItems(["圆形散景", "六边形光圈", "八边形光圈"])
        self.combo_flare_shape.setCurrentText(self.params.flare_shape)
        self.combo_flare_shape.currentTextChanged.connect(lambda: self._enqueue_render("proxy"))
        self.btn_flare_col = QPushButton()
        self.btn_flare_col.setFixedSize(40, 22)
        self.btn_flare_col.setToolTip("光斑颜色")
        self.btn_flare_col.clicked.connect(lambda: self._pick_fx_color("flare"))
        self._set_swatch(self.btn_flare_col, self.params.flare_color_rgb)
        flare_meta_row.addWidget(self.combo_flare_shape, stretch=1)
        flare_meta_row.addWidget(self.btn_flare_col)

        for s in (self.sl_halo_str, self.sl_halo_rad, self.sl_halo_wid, self.sl_halo_core, self.sl_halo_hide):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            l_halo.addWidget(s)
        l_halo.addLayout(halo_col_row)
        for s in (self.sl_flare_str, self.sl_flare_sz, self.sl_flare_ang):
            s.valueChanged.connect(self._on_slider_moving)
            s.sliderReleased.connect(lambda: self._hq_timer.start())
            l_halo.addWidget(s)
        l_halo.addLayout(flare_meta_row)
        self.fx_stack.addWidget(p_halo)

        cfx.addWidget(self.fx_stack)
        rl.addWidget(card_fx)

        # 卡片 4: 空间遮挡控制
        card_occ = QFrame()
        card_occ.setObjectName("Card")
        c3 = QVBoxLayout(card_occ)
        c3.setContentsMargins(12, 10, 12, 10)
        c3.setSpacing(4)
        t3 = QLabel("🛡️ 4. 空间遮挡控制")
        t3.setObjectName("CardTitle")
        c3.addWidget(t3)

        self.cb_occ = QCheckBox("开启深度自动前景遮挡")
        self.cb_occ.setChecked(self.params.occlusion_enabled)
        self.cb_occ.toggled.connect(lambda: self._enqueue_render("proxy"))
        self.sl_occ_bias = CompactSlider("遮挡前后阈值微调", -0.08, 0.12, self.params.occlusion_bias, 0.005)
        self.sl_occ_bias.valueChanged.connect(self._on_slider_moving)
        self.sl_occ_bias.sliderReleased.connect(lambda: self._hq_timer.start())
        c3.addWidget(self.cb_occ)
        c3.addWidget(self.sl_occ_bias)
        rl.addWidget(card_occ)

        rl.addStretch()

        hint = QLabel("操作：V移动(左键拖角色) / B模糊笔刷 | Ctrl+滚轮缩角色 | 滚轮缩画布 | 空格对比")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #71717a; font-size: 11px;")
        rl.addWidget(hint)

        scroll.setWidget(right_panel)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)


        self._update_mode_buttons()
        self._set_tool_mode("move")

    def _position_dock(self) -> None:
        if not hasattr(self, "canvas") or not hasattr(self, "dock") or self.canvas is None or self.dock is None:
            return
        cw, ch = self.canvas.width(), self.canvas.height()
        if cw <= 0 or ch <= 0:
            return
        hint_w = self.dock.sizeHint().width()
        dw = min(cw - 24, max(hint_w + 16, 720))
        dh = 48
        dx = max(12, (cw - dw) // 2)
        dy = max(12, ch - dh - 18)
        self.dock.setGeometry(dx, dy, dw, dh)
        if hasattr(self, "brush_dock") and self.brush_dock is not None:
            bw = min(cw - 24, max(self.brush_dock.sizeHint().width() + 24, 620))
            bh = 42
            self.brush_dock.setGeometry(max(12, (cw - bw) // 2), max(12, dy - bh - 8), bw, bh)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas and event.type() == event.Type.Resize:
            self._position_dock()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "canvas") and hasattr(self, "dock"):
            self._position_dock()

    def _format_bokeh_fstop(self, t: float) -> str:
        f_val = self.pipeline.get_target_fnumber(t)
        f_str = f"f{f_val:.2f}" if f_val < 1.0 else f"f{f_val:.1f}"
        if t <= 0.01:
            src_tag = "原片EXIF" if self.pipeline.bg_has_exif_f else "默认"
            return f"{f_str} ({src_tag})"
        return f_str

    def _set_swatch(self, btn: QPushButton, rgb: tuple[int, int, int]) -> None:
        r, g, b = rgb
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt_col = "#111111" if lum > 150 else "#f4f4f5"
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {txt_col}; border: 1px solid #e4e4e7; border-radius: 5px; font-weight: bold; padding: 0;"
        )

    def _pick_light_color(self) -> None:
        r, g, b = self.params.custom_light_color_rgb
        col = QColorDialog.getColor(QColor(r, g, b), self, "选择主光源勾边颜色")
        if col.isValid():
            self.params.custom_light_color_rgb = (col.red(), col.green(), col.blue())
            self._set_swatch(self.btn_light_color, self.params.custom_light_color_rgb)
            self.cb_auto_color.blockSignals(True)
            self.cb_auto_color.setChecked(False)
            self.cb_auto_color.blockSignals(False)
            self._enqueue_render("proxy")
            self.combo_color_preset.blockSignals(True)
            self.combo_color_preset.setCurrentIndex(0)
            self.combo_color_preset.blockSignals(False)

    def _on_color_preset_changed(self, name: str) -> None:
        rgb = self._light_color_presets.get(name)
        if rgb is None:
            return
        self.params.custom_light_color_rgb = rgb
        self._set_swatch(self.btn_light_color, rgb)
        self.cb_auto_color.blockSignals(True)
        self.cb_auto_color.setChecked(False)
        self.cb_auto_color.blockSignals(False)
        self._enqueue_render("proxy")

    def _pick_fx_color(self, target: str) -> None:
        if target == "halo_in":
            curr = self.params.halo_inner_rgb
            title = "选择光晕内圈颜色"
        elif target == "halo_out":
            curr = self.params.halo_outer_rgb
            title = "选择光晕外圈过渡色"
        else:
            curr = self.params.flare_color_rgb
            title = "选择太阳光斑颜色"

        col = QColorDialog.getColor(QColor(*curr), self, title)
        if col.isValid():
            rgb = (col.red(), col.green(), col.blue())
            if target == "halo_in":
                self.params.halo_inner_rgb = rgb
                self._set_swatch(self.btn_halo_in, rgb)
            elif target == "halo_out":
                self.params.halo_outer_rgb = rgb
                self._set_swatch(self.btn_halo_out, rgb)
            else:
                self.params.flare_color_rgb = rgb
                self._set_swatch(self.btn_flare_col, rgb)
            if not self.cb_fx_enable.isChecked():
                self.cb_fx_enable.setChecked(True)
            else:
                self._enqueue_render("proxy")

    def _on_fx_type_changed(self, idx: int) -> None:
        self.fx_stack.setCurrentIndex(idx)
        if not self.cb_fx_enable.isChecked():
            self.cb_fx_enable.setChecked(True)
        else:
            self._enqueue_render("proxy")

    def _on_fx_layer_changed(self) -> None:
        layer = str(self.combo_fx_layer.currentData() or "置于人物前方")
        self.sl_fx_occ.setVisible(layer in ("置于人物后方", "人后体积逆光"))
        self.sl_fx_spill.setVisible(layer == "人后体积逆光")
        self._enqueue_render("proxy")

    def _set_tool_mode(self, mode: str) -> None:
        self.canvas.tool_mode = mode
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if mode == "brush" else Qt.CursorShape.ArrowCursor)
        act_style = "background-color: #0284c7; border: 1px solid #38bdf8; color: white; border-radius: 16px; padding: 6px 14px; font-weight: bold;"
        norm_style = "background-color: rgba(255, 255, 255, 14); border: 1px solid rgba(255, 255, 255, 22); color: #e4e4e7; border-radius: 16px; padding: 6px 14px; font-weight: bold;"
        self.btn_tool_move.setStyleSheet(act_style if mode == "move" else norm_style)
        self.btn_tool_brush.setStyleSheet(act_style if mode == "brush" else norm_style)
        if hasattr(self, "brush_dock") and self.brush_dock is not None:
            self.brush_dock.setVisible(mode == "brush")
            self._position_dock()
        self.canvas.viewport().update()

    def _on_brush_cfg_changed(self) -> None:
        self.canvas.brush_mode = str(self.combo_brush_mode.currentData() or "blur")
        self.canvas.brush_radius_norm = self.sl_brush_sz.value() / 1000.0
        self.canvas.brush_hardness = self.sl_brush_hard.value() / 100.0
        self.canvas.brush_opacity = self.sl_brush_op.value() / 100.0
        self.canvas.viewport().update()

    def _on_edge_shadow_mode_changed(self) -> None:
        e_mode = str(self.combo_edge_mode.currentData() or "anime")
        if hasattr(self, "real_edge_box"):
            self.real_edge_box.setVisible(e_mode == "real")
        self._enqueue_render("proxy")

    def _toggle_work_mode(self) -> None:
        next_mode = "anime" if self.work_mode == "cosplay" else "cosplay"
        self._apply_work_mode_defaults(next_mode, render=True)

    def _apply_work_mode_defaults(self, mode: str, render: bool = True) -> None:
        self.work_mode = mode
        self.params.work_mode = mode
        if hasattr(self, "btn_work_mode"):
            self.btn_work_mode.setText("📸 场照溶图模式" if mode == "cosplay" else "🌌 二次元溶图模式")

        self.combo_edge_mode.blockSignals(True)
        self.combo_shadow_mode.blockSignals(True)
        if mode == "cosplay":
            self.combo_edge_mode.setCurrentIndex(0)    # real
            self.combo_shadow_mode.setCurrentIndex(0)  # real
            self.sl_rim.set_value(0.65)
            self.sl_light_wrap.set_value(0.35)
            self.sl_25d.set_value(0.32)
            self.sl_tone.set_value(0.25)
            self.sl_shadow.set_value(0.45)
            self.sl_feather.set_value(0.6)
            self.sl_dof_tol.set_value(0.08)
            self.real_edge_box.setVisible(True)
            self.combo_fx_layer.blockSignals(True)
            self.combo_fx_layer.setCurrentIndex(0)  # 人后体积逆光
            self.combo_fx_layer.blockSignals(False)
            self.sl_fx_spill.setVisible(True)
        else:
            self.combo_edge_mode.setCurrentIndex(1)    # anime
            self.combo_shadow_mode.setCurrentIndex(1)  # anime
            self.sl_rim.set_value(0.90)
            self.sl_light_wrap.set_value(0.55)
            self.sl_25d.set_value(0.45)
            self.sl_tone.set_value(0.70)
            self.sl_shadow.set_value(0.65)
            self.sl_feather.set_value(1.5)
            self.sl_dof_tol.set_value(0.06)
            self.real_edge_box.setVisible(False)
            self.combo_fx_layer.blockSignals(True)
            self.combo_fx_layer.setCurrentIndex(2)  # 置于人物前方
            self.combo_fx_layer.blockSignals(False)
            self.sl_fx_spill.setVisible(False)
        self.combo_edge_mode.blockSignals(False)
        self.combo_shadow_mode.blockSignals(False)

        if render and self.pipeline.bg_proxy_u8 is not None and self.pipeline.char_full_bgra is not None:
            self._enqueue_render("proxy")

    def _update_mode_buttons(self) -> None:
        act = "background-color: #0284c7; border-color: #38bdf8; color: #ffffff; font-weight: bold;"
        norm = "background-color: #1f1f24; border-color: #3f3f46; color: #a1a1aa;"
        self.btn_mode_merged.setStyleSheet(act if self._view_mode == "merged" else norm)
        self.btn_mode_raw.setStyleSheet(act if self._view_mode == "raw" else norm)
        self.btn_mode_depth.setStyleSheet(act if self._view_mode == "depth" else norm)

    def _set_view_mode(self, mode: str) -> None:
        if self._view_mode == mode:
            return
        self._view_mode = mode
        self._update_mode_buttons()
        self._enqueue_render("proxy")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._prev_mode_before_space = self._view_mode
            self._set_view_mode("raw")
            event.accept()
            return
        if key == Qt.Key.Key_1:
            self._set_view_mode("merged")
        elif key == Qt.Key.Key_2:
            self._set_view_mode("raw")
        elif key == Qt.Key.Key_3:
            self._set_view_mode("depth")
        elif key == Qt.Key.Key_V:
            self._set_tool_mode("move")
        elif key == Qt.Key.Key_B:
            self._set_tool_mode("brush")
        elif key == Qt.Key.Key_BracketLeft:
            self.sl_brush_sz.setValue(max(15, self.sl_brush_sz.value() - 10))
        elif key == Qt.Key.Key_BracketRight:
            self.sl_brush_sz.setValue(min(200, self.sl_brush_sz.value() + 10))
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._set_view_mode(self._prev_mode_before_space)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _sync_canonical_bg_size(self) -> None:
        if self.pipeline.bg_proxy_u8 is not None:
            ph, pw = self.pipeline.bg_proxy_u8.shape[:2]
            self.canvas.set_canonical_size(pw, ph)

    def _try_load_samples(self) -> None:
        if not os.path.isdir("samples"):
            return
        bg_cands = [f for f in os.listdir("samples") if f.lower().startswith("bg")]
        ch_cands = [f for f in os.listdir("samples") if f.lower().startswith("char")]
        if bg_cands:
            self.pipeline.load_background(os.path.join("samples", bg_cands[0]))
            self._sync_canonical_bg_size()
        if ch_cands:
            self.pipeline.load_character(os.path.join("samples", ch_cands[0]))
        if bg_cands and ch_cands:
            self._enqueue_render("proxy")

    def load_cosplay_photo(self, path: str) -> None:
        """供外部一键送入场照原片：载入底图 -> 自动抠人 -> 1:1 绝对重合"""
        if not path or not os.path.exists(path):
            return
        self._apply_work_mode_defaults("cosplay", render=False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.pipeline.load_background(path)
            self._sync_canonical_bg_size()
            self.pipeline.auto_extract_character_from_bg()
            self.params.is_same_bg_aligned = True
            self.params.pos_x, self.params.pos_y = 0.50, 0.50
            self.sl_scale.set_value(1.00, emit=False)
        finally:
            QApplication.restoreOverrideCursor()
        self._first_view = True
        self._enqueue_render("proxy")

    def _auto_cutout_current_bg(self) -> None:
        if self.pipeline.bg_full_u8 is None:
            QMessageBox.information(self, "提示", "请先导入或拖入一张底图！")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.pipeline.auto_extract_character_from_bg()
            self.params.is_same_bg_aligned = True
            self.params.pos_x, self.params.pos_y = 0.50, 0.50
            self.sl_scale.set_value(1.00, emit=False)
        finally:
            QApplication.restoreOverrideCursor()
        self._enqueue_render("proxy")

    def _apply_auto_alignment_if_cosplay(self) -> None:
        if self.work_mode != "cosplay":
            self.params.is_same_bg_aligned = False
            return
        if self.pipeline.bg_proxy_u8 is None or self.pipeline.char_full_bgra is None:
            return
        is_same, px, py, sc = self.pipeline.detect_same_bg_alignment()
        self.params.is_same_bg_aligned = is_same
        self.params.pos_x = px
        self.params.pos_y = py
        self.sl_scale.set_value(sc, emit=False)
        self.combo_shadow_mode.blockSignals(True)
        self.combo_shadow_mode.setCurrentIndex(0 if is_same else 1)
        self.combo_shadow_mode.blockSignals(False)

    def _pick_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择现实底图", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.pipeline.load_background(path)
                self._sync_canonical_bg_size()
                if self.work_mode == "cosplay" and self.pipeline.char_full_bgra is None:
                    self.pipeline.auto_extract_character_from_bg()
                self._apply_auto_alignment_if_cosplay()
            finally:
                QApplication.restoreOverrideCursor()
            self._first_view = True
            self._enqueue_render("proxy")

    def _pick_char(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择透明底角色立绘", "", "PNG Images (*.png *.webp)")
        if path:
            self.pipeline.load_character(path)
            self._apply_auto_alignment_if_cosplay()
            self._enqueue_render("proxy")

    def _on_files_dropped(self, paths: list) -> None:
        got_bg = False
        got_char = False
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in (".png", ".webp"):
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4 and np.any(img[:, :, 3] < 250):
                    self.pipeline.load_character(p)
                    got_char = True
                    continue
            if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self.pipeline.load_background(p)
                    self._sync_canonical_bg_size()
                    got_bg = True
                finally:
                    QApplication.restoreOverrideCursor()
                self._first_view = True

        if self.work_mode == "cosplay" and got_bg and not got_char:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.pipeline.auto_extract_character_from_bg()
            finally:
                QApplication.restoreOverrideCursor()
        self._apply_auto_alignment_if_cosplay()
        self._enqueue_render("proxy")

    def _snap_to_bottom(self) -> None:
        self.params.pos_y = 1.0 - self.sl_scale.value() / 2.0
        self._enqueue_render("proxy")

    def _clear_brush(self) -> None:
        self.pipeline.reset_fade_mask()
        self._enqueue_render("proxy")

    def _reset_params(self) -> None:
        d = MergeParams()
        self.params.pos_x, self.params.pos_y = d.pos_x, d.pos_y
        self.params.custom_light_color_rgb = d.custom_light_color_rgb
        self.params.halo_inner_rgb = d.halo_inner_rgb
        self.params.halo_outer_rgb = d.halo_outer_rgb
        self.params.flare_color_rgb = d.flare_color_rgb

        self.sl_scale.set_value(d.scale)
        self.sl_bg_ev.set_value(d.bg_exposure)
        self.sl_ch_ev.set_value(d.char_exposure)
        self.sl_bokeh.set_value(d.aperture_strength)
        self.combo_sensor.blockSignals(True)
        self.combo_sensor.setCurrentText(d.sensor_type)
        self.combo_sensor.blockSignals(False)
        self.sl_feather.set_value(d.edge_feather)
        self.sl_rim.set_value(d.rim_strength)
        self.sl_25d.set_value(d.shading_25d_strength)
        self.sl_tone.set_value(d.brightness_match)
        self.sl_shadow.set_value(d.shadow_opacity)
        self.sl_occ_bias.set_value(d.occlusion_bias)



        
        self.cb_fx_enable.setChecked(d.light_fx_enabled)
        self.cb_face_protect.setChecked(d.face_protect_enabled)
        self.sl_fx_x.set_value(d.fx_pos_x)
        self.sl_fx_y.set_value(d.fx_pos_y)
        self.sl_fx_aim.set_value(d.fx_aim_deg)
        self.combo_fx_type.blockSignals(True)
        self.combo_fx_type.setCurrentText(d.light_fx_type)
        self.fx_stack.setCurrentIndex(0)
        self.combo_fx_type.blockSignals(False)

        self.sl_tyn_str.set_value(d.tyndall_strength)
        self.sl_tyn_cnt.set_value(d.tyndall_count)
        self.sl_tyn_soft.set_value(d.tyndall_softness)
        self.sl_tyn_len.set_value(d.tyndall_length)

        self.sl_spot_str.set_value(d.spot_strength)
        self.sl_spot_wid.set_value(d.spot_width)
        self.sl_spot_soft.set_value(d.spot_softness)
        self.sl_spot_dust.set_value(d.spot_dust)
        self.sl_spot_tip.set_value(d.spot_tip_hide)

        self.sl_halo_str.set_value(d.halo_strength)
        self.sl_halo_rad.set_value(d.halo_radius)
        self.sl_halo_core.set_value(d.halo_core_strength)
        self.sl_halo_hide.set_value(d.halo_arc_hide)
        self.sl_halo_wid.set_value(d.halo_width)
        self.sl_flare_str.set_value(d.flare_strength)
        self.sl_flare_ang.set_value(d.flare_angle)
        self.combo_flare_shape.blockSignals(True)
        self.combo_flare_shape.setCurrentText(d.flare_shape)
        self.combo_flare_shape.blockSignals(False)

        self._set_swatch(self.btn_halo_in, d.halo_inner_rgb)
        self._set_swatch(self.btn_halo_out, d.halo_outer_rgb)
        self._set_swatch(self.btn_flare_col, d.flare_color_rgb)

        self.combo_light_mode.blockSignals(True)
        self.combo_light_mode.setCurrentText(d.light_dir_mode)
        self.combo_light_mode.blockSignals(False)
        self.cb_auto_color.setChecked(True)
        self.cb_occ.setChecked(d.occlusion_enabled)
        self.pipeline.reset_fade_mask()
        self._enqueue_render("proxy")

    def _trigger_fast_throttled(self) -> None:
        if not self._fast_timer.isActive():
            self._fast_timer.start()

    def _on_canvas_move(self, px: float, py: float) -> None:
        self.params.pos_x = px
        self.params.pos_y = py
        self._trigger_fast_throttled()

    def _on_canvas_scale(self, sc: float) -> None:
        self.sl_scale.set_value(sc, emit=False)
        self._trigger_fast_throttled()
        self._hq_timer.start()

    def _on_brush_paint(
        self,
        u: float,
        v: float,
        u_bg: float,
        v_bg: float,
        r_norm: float,
        is_erase: bool,
        hardness: float,
        opacity: float,
        brush_mode: str,
    ) -> None:
        self.pipeline.paint_fade_mask(
            u,
            v,
            r_norm,
            erase=is_erase,
            hardness=hardness,
            opacity=opacity,
            brush_mode=brush_mode,
            u_bg=u_bg,
            v_bg=v_bg,
        )
        self._trigger_fast_throttled()

    def _on_compass_changed(self, deg: float) -> None:
        self.combo_light_mode.blockSignals(True)
        self.combo_light_mode.setCurrentText("自定义")
        self.combo_light_mode.blockSignals(False)
        self.params.azimuth_deg = deg
        self.sl_az.set_value(deg, emit=False)
        self.lbl_az.setText(f"方位角: {deg:.0f}°")
        self._trigger_fast_throttled()
        self._hq_timer.start()

    def _on_az_slider_changed(self, deg: float) -> None:
        self.combo_light_mode.blockSignals(True)
        self.combo_light_mode.setCurrentText("自定义")
        self.combo_light_mode.blockSignals(False)
        self.params.azimuth_deg = deg
        self.compass.set_azimuth(deg)
        self.lbl_az.setText(f"方位角: {deg:.0f}°")
        self._trigger_fast_throttled()

    def _on_slider_moving(self) -> None:
        self._trigger_fast_throttled()
        self._hq_timer.start()

    def _collect_params(self) -> MergeParams:
        tone = self.sl_tone.value()
        rim = self.sl_rim.value()
        return MergeParams(
            pos_x=self.params.pos_x,
            pos_y=self.params.pos_y,
            scale=self.sl_scale.value(),
            light_dir_mode=self.combo_light_mode.currentText(),
            azimuth_deg=self.params.azimuth_deg,
            elevation_deg=35.0,
            auto_light_color=self.cb_auto_color.isChecked(),
            custom_light_color_rgb=self.params.custom_light_color_rgb,
            bg_exposure=self.sl_bg_ev.value(),
            char_exposure=self.sl_ch_ev.value(),
            light_fx_enabled=self.cb_fx_enable.isChecked(),
            light_fx_type=self.combo_fx_type.currentText(),
            face_protect_enabled=self.cb_face_protect.isChecked(),
            light_fx_layer=str(self.combo_fx_layer.currentData() or "置于人物前方"),
            fx_edge_spill=self.sl_fx_spill.value(),
            fx_occlusion_strength=self.sl_fx_occ.value(),
            fx_pos_x=self.sl_fx_x.value(),
            fx_pos_y=self.sl_fx_y.value(),
            fx_aim_deg=self.sl_fx_aim.value(),
            tyndall_strength=self.sl_tyn_str.value(),
            tyndall_count=self.sl_tyn_cnt.value(),
            tyndall_softness=self.sl_tyn_soft.value(),
            tyndall_length=self.sl_tyn_len.value(),
            spot_strength=self.sl_spot_str.value(),
            spot_width=self.sl_spot_wid.value(),
            spot_softness=self.sl_spot_soft.value(),
            spot_dust=self.sl_spot_dust.value(),
            spot_tip_hide=self.sl_spot_tip.value(),
            halo_strength=self.sl_halo_str.value(),
            halo_radius=self.sl_halo_rad.value(),
            halo_core_strength=self.sl_halo_core.value(),
            halo_arc_hide=self.sl_halo_hide.value(),
            halo_width=self.sl_halo_wid.value(),
            halo_inner_rgb=self.params.halo_inner_rgb,
            halo_outer_rgb=self.params.halo_outer_rgb,
            flare_strength=self.sl_flare_str.value(),
            flare_size=self.sl_flare_sz.value(),
            flare_angle=self.sl_flare_ang.value(),
            flare_shape=self.combo_flare_shape.currentText(),
            flare_color_rgb=self.params.flare_color_rgb,
            aperture_strength=self.sl_bokeh.value(),
            sensor_type=self.combo_sensor.currentText(),
            edge_feather=self.sl_feather.value(),
            rim_strength=rim,
            light_wrap_strength=self.sl_light_wrap.value(),
            shading_25d_strength=self.sl_25d.value(),
            brightness_match=tone,
            color_temp_match=tone * 0.65,
            shadow_lift=tone * 0.68,
            shadow_opacity=self.sl_shadow.value(),
            shadow_length=0.85,
            blur_match_strength=0.20,
            noise_match_strength=0.20,
            occlusion_enabled=self.cb_occ.isChecked(),
            occlusion_bias=self.sl_occ_bias.value(),
            work_mode=self.work_mode,
            edge_mode=str(self.combo_edge_mode.currentData() or "anime"),
            shadow_mode=str(self.combo_shadow_mode.currentData() or "anime"),
            edge_choke=self.sl_edge_choke.value(),
            rim_texture_weight=self.sl_rim_tex.value(),
            rim_falloff=self.sl_rim_falloff.value(),
            hair_transmission=self.sl_hair_trans.value(),
            normal_detail_mix=self.sl_normal_mix.value(),
            dof_tolerance=self.sl_dof_tol.value(),
            is_same_bg_aligned=self.params.is_same_bg_aligned,
        )

    def _enqueue_render(self, quality: str) -> None:
        if self.pipeline.bg_proxy_u8 is None or self.pipeline.char_full_bgra is None:
            return
        self.params = self._collect_params()
        self._pending_render = (self.params, quality)
        self._kick_preview_worker()

    def _kick_preview_worker(self) -> None:
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        if self._pending_render is None:
            return
        params, quality = self._pending_render
        self._pending_render = None

        self._preview_worker = PreviewWorker(self.pipeline, params, quality, self._view_mode)
        self._preview_worker.sig_done.connect(self._on_preview_done)
        self._preview_worker.sig_err.connect(lambda err: self.lbl_stats.setText(f"渲染异常: {err}"))
        self._preview_worker.start()

    def _on_preview_done(self, out_u8: np.ndarray, quality: str, dt: float, auto_az: float, auto_rgb: tuple) -> None:
        if self.params.light_dir_mode in ("自动识别", "绑定光效"):
            self.params.azimuth_deg = auto_az
            self.compass.set_azimuth(auto_az)
            self.sl_az.set_value(auto_az, emit=False)
            tag_l = "自动" if self.params.light_dir_mode == "自动识别" else "已绑光效"
            self.lbl_az.setText(f"方位角: {auto_az:.0f}° ({tag_l})")
        if self.params.auto_light_color:
            self.params.custom_light_color_rgb = auto_rgb
            self._set_swatch(self.btn_light_color, auto_rgb)

        ch_h, ch_w = self.pipeline.char_full_bgra.shape[:2]
        self.canvas.sync_char_transform(self.params.pos_x, self.params.pos_y, self.params.scale, ch_w / float(ch_h))
        self.canvas.set_image(out_u8, reset_view=self._first_view)
        self._first_view = False

        tag = "⚡拖拽预览" if quality == "fast" else "✨高清预览"
        self.sl_bokeh.set_value(self.sl_bokeh.value(), emit=False)
        self.lbl_stats.setText(f"{tag} {dt:.0f}ms | {self.pipeline.depth_engine.active_providers[0]}")

        if self._pending_render is not None:
            self._kick_preview_worker()

    def _export_full_res(self) -> None:
        if self.pipeline.bg_full_u8 is None or self.pipeline.char_full_bgra is None:
            QMessageBox.warning(self, "提示", "请先导入底图和角色立绘！")
            return
        os.makedirs("output", exist_ok=True)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出全分辨率成片",
            os.path.join("output", "photomerge_export.jpg"),
            "JPEG (*.jpg);;PNG (*.png)",
        )
        if not save_path:
            return

        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.wait()

        params = self._collect_params()
        self._export_dlg = ExportProgressDialog(self)
        self._export_worker = ExportWorker(self.pipeline, params, save_path)
        self._export_worker.sig_progress.connect(self._export_dlg.set_status)
        self._export_worker.sig_finished.connect(self._on_export_finished)
        self._export_worker.sig_error.connect(self._on_export_error)
        self._export_worker.start()
        self._export_dlg.exec()

    def _on_export_finished(self, save_path: str, w: int, h: int, dt: float) -> None:
        if self._export_dlg:
            self._export_dlg.accept()
            self._export_dlg = None
        QMessageBox.information(
            self,
            "导出成功",
            f"满血成片已保存：\n{save_path}\n分辨率：{w} × {h}\n耗时：{dt:.2f} 秒\n（若为 JPEG 已无损注入原片 EXIF/ICC）",
        )

    def _on_export_error(self, err: str) -> None:
        if self._export_dlg:
            self._export_dlg.reject()
            self._export_dlg = None
        QMessageBox.critical(self, "导出失败", str(err))