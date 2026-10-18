# -*- coding: utf-8 -*-
"""
splash_overlay.py - Anicrop Studio 开机自检遮罩层

只做两件事：
  1. 显示 8 秒 BIOS 自检流
  2. 结束后隐藏自己（不融化、不淡出，直接消失）
"""
from boot_beep import play_boot_beep
import time
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QFont




BOOT_LINES = [
    ("Anicrop Studio Kernel Beta 1.0.1", 0.15),
    ("Copyright (c) 2026 R4ksh4. All rights reserved.", 0.10),
    ("", 0.08),
    ("[    0.000000] Booting Anicrop Studio...", 0.12),
    ("[    0.012843] Runtime: Python 3.11.17", 0.15),
    ("[    0.028451] UI Framework: PySide6 6.11.2", 0.12),
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
    ("Design By R4ksh4", 0.15),
]


class SplashOverlay(QWidget):
    sig_finished = Signal()

    DURATION = 5.0

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: transparent;")
        self.setFocusPolicy(Qt.StrongFocus)

        self._t0 = time.time()
        self._finished = False
        self._skip_requested = False
        self._line_index = -1
        self._line_timer = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        import threading
        QTimer.singleShot(80, lambda: play_boot_beep(freq=2000, beep_ms=100, gap_ms=20, repeat=2,volume=0.8))

    def start(self):
        self._t0 = time.time()
        self.raise_()
        self.show()

    def skip(self):
        if self._finished:
            return
        self._skip_requested = True
        self._finalize()

    def _tick(self):
        if self._finished:
            return
        elapsed = time.time() - self._t0

        if self._skip_requested:
            self._finalize()
            return

        if elapsed >= self.DURATION:
            self._finalize()
            return

        self._update_boot_lines()
        self.update()

    def _update_boot_lines(self):
        while self._line_index < len(BOOT_LINES) - 1:
            self._line_timer += 0.016
            next_interval = BOOT_LINES[self._line_index + 1][1]
            if self._line_timer >= next_interval:
                self._line_timer = 0.0
                self._line_index += 1
            else:
                break

    def _finalize(self):
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self.hide()
        self.sig_finished.emit()

    def keyPressEvent(self, event):
        self.skip()

    def mousePressEvent(self, event):
        self.skip()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 255))

        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.Monospace)
        painter.setFont(font)
        painter.setPen(QColor("#d8d8d8"))

        line_h = painter.fontMetrics().height() + 2
        mx, my = 40, 40
        y = my
        for i in range(min(self._line_index + 1, len(BOOT_LINES))):
            text = BOOT_LINES[i][0]
            if text:
                painter.drawText(mx, y, text)
            y += line_h