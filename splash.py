# -*- coding: utf-8 -*-
"""
splash.py - Anicrop Studio 开机自检动画

和主窗口同尺寸的窗口，居中显示。
- 纯黑底 + BIOS 风格白色等宽字体自检流
- 8 秒，可跳过（任意键 / 点击）
- 结束后直接隐藏自己（无白闪）
- emit sig_finished，让 main.py 显示主窗口

用法:
    splash = SplashScreen(width=1320, height=880)
    splash.sig_finished.connect(show_main_window)
    splash.show()
"""
from boot_beep import play_boot_beep
import sys
import time
from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QKeyEvent, QMouseEvent




# ==================== 自检日志内容 ====================
# (文本, 下一行出现的间隔秒数)
BOOT_LINES = [
    ("Anicrop Studio Kernel Beta 2.0.0", 0.15),
    ("Copyright (c) 2026 R4ksh4. All rights reserved.", 0.10),
    ("", 0.08),
    ("[    0.000000] Booting Anicrop Studio...", 0.12),
    ("[    0.012843] CPU: Photo Processing Engine @ 4 cores", 0.15),
    ("[    0.028451] Memory: 8192MB available", 0.12),
    ("[    0.041209] Initializing image pipeline...", 0.15),
    ("[    0.053872] Loading RAW decoder (CR3/ARW/NEF/DNG) ... [OK]", 0.35),
    ("[    0.067431] Loading EXIF parser ................ [OK]", 0.35),
    ("[    0.081546] Loading SCRFD-10G face detector ..... [OK]", 0.45),
    ("[    0.098213] Mounting thumbnail cache ............ [OK]", 0.35),
    ("[    0.114789] Initializing UI subsystem ........... [OK]", 0.35),
    ("[    0.131002] Restoring last session .............. [OK]", 0.35),
    ("[    0.148564] Anicrop Studio ready.", 0.30),
    ("", 0.10),
    ("R4ksh4_ChooseAgent.exe", 0.20),
    ("5.100.98.75", 0.15),
]
# ====================================================


class SplashScreen(QWidget):
    sig_finished = Signal()

    def __init__(self, parent=None, width: int = 1320, height: int = 880):
        super().__init__(parent)
        # 无边框 + 置顶，但不全屏
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.SplashScreen
        )
        self.setStyleSheet("background-color: #000000;")
        self.resize(width, height)

        self._line_index = -1
        self._line_timer = 0.0
        self._elapsed = 0.0
        self._finished_flag = False
        self._skip_requested = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        # 滴一声
        import threading
        QTimer.singleShot(80, lambda: play_boot_beep(freq=2000, beep_ms=100, gap_ms=20, repeat=2,volume=0.8))


    def show_centered(self):
        """屏幕居中显示"""
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)
        self.show()
        self.setFocus()
        self.raise_()
        self.activateWindow()

    def _skip(self):
        if self._finished_flag:
            return
        self._skip_requested = True
        self._finalize()

    def _finalize(self):
        if self._finished_flag:
            return
        self._finished_flag = True
        self._timer.stop()
        self.hide()
        self.sig_finished.emit()

    def _tick(self):
        dt = 0.016
        self._elapsed += dt

        if self._skip_requested:
            return

        if self._line_index < len(BOOT_LINES) - 1:
            self._line_timer += dt
            next_line = BOOT_LINES[self._line_index + 1]
            interval = next_line[1]
            if self._line_timer >= interval:
                self._line_timer = 0.0
                self._line_index += 1
                self.update()
        else:
            # 全部刷完，稍等 0.5 秒
            if self._elapsed > 8.0:
                self._finalize()

    def keyPressEvent(self, event: QKeyEvent):
        self._skip()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        self._skip()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor("#000000"))

        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.Monospace)
        painter.setFont(font)
        painter.setPen(QColor("#d8d8d8"))

        line_height = painter.fontMetrics().height() + 2
        margin_x = 40
        margin_y = 40

        y = margin_y
        for i in range(min(self._line_index + 1, len(BOOT_LINES))):
            text = BOOT_LINES[i][0]
            if text:
                painter.drawText(margin_x, y, text)
            y += line_height