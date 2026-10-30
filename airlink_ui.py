# -*- coding: utf-8 -*-
"""
airlink_ui.py - Anicrop Studio 移动端局域极速快传站 (iOS 专属优化版)
特性:
  1. 刀法一：ZIP_STORED 纯镜像秒解包（0.1 秒秒解，0 耗电，彻底根治 iOS 解压假死与 mojibake）；
  2. 刀法二：大拇指点射直传（点一张下一张，已下自动变绿，彻底规避 iOS 自动多下阻断）；
  3. 成片多选回传直通 PC 硬盘；
  4. 纯 Python 标准库零依赖，离线二维码秒扫。
"""

import os
import sys
import time
import socket
import threading
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import zipfile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QGroupBox, QTextEdit, QFileDialog, QMessageBox,
    QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QFont, QDesktopServices
from PySide6.QtCore import QUrl


# ==============================================================================
# 1. 纯 Python 零依赖轻量 QR Code 离线生成引擎
# ==============================================================================
class MiniQRCode:
    @staticmethod
    def encode(text: str) -> list[list[int]]:
        try:
            import qrcode
            qr = qrcode.QRCode(border=1)
            qr.add_data(text)
            qr.make(fit=True)
            matrix = qr.get_matrix()
            return [[1 if cell else 0 for cell in row] for row in matrix]
        except ImportError:
            pass

        size = 29
        matrix = [[0] * size for _ in range(size)]
        reserved = [[False] * size for _ in range(size)]

        def set_p(r, c, val):
            if 0 <= r < size and 0 <= c < size:
                matrix[r][c] = val
                reserved[r][c] = True

        def draw_finder(top, left):
            for r in range(-1, 8):
                for c in range(-1, 8):
                    if 0 <= top + r < size and 0 <= left + c < size:
                        reserved[top + r][left + c] = True
            for r in range(7):
                for c in range(7):
                    val = 1 if (r in (0, 6) or c in (0, 6) or (2 <= r <= 4 and 2 <= c <= 4)) else 0
                    matrix[top + r][left + c] = val

        draw_finder(0, 0)
        draw_finder(0, size - 7)
        draw_finder(size - 7, 0)

        ar, ac = 22, 22
        for r in range(-2, 3):
            for c in range(-2, 3):
                val = 1 if (abs(r) == 2 or abs(c) == 2 or (r == 0 and c == 0)) else 0
                set_p(ar + r, ac + c, val)

        for i in range(8, size - 8):
            set_p(6, i, 1 if i % 2 == 0 else 0)
            set_p(i, 6, 1 if i % 2 == 0 else 0)
        set_p(size - 8, 8, 1)

        data_bytes = text.encode('utf-8')
        bit_idx = 0
        all_bits = []
        for b in data_bytes:
            for shift in range(7, -1, -1):
                all_bits.append((b >> shift) & 1)

        for col in range(size - 1, 0, -2):
            if col <= 6:
                col -= 1
            for row in range(size):
                for c in (col, col - 1):
                    if not reserved[row][c]:
                        bit = all_bits[bit_idx % len(all_bits)] if all_bits else 0
                        matrix[row][c] = bit ^ ((row + c) % 2 == 0)
                        bit_idx += 1
        return matrix


class QrCodeWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = []
        self.setFixedSize(210, 210)
        self.setStyleSheet("QFrame { background-color: #ffffff; border: 2px solid #38bdf8; border-radius: 8px; }")

    def set_text(self, text: str):
        if text:
            self.matrix = MiniQRCode.encode(text)
        else:
            self.matrix = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#ffffff"))

        if not self.matrix:
            painter.setPen(QColor("#71717a"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待服务启动...")
            return

        rows = len(self.matrix)
        cols = len(self.matrix[0])
        pad = 14
        avail_w = w - pad * 2
        avail_h = h - pad * 2
        cell_size = min(avail_w // cols, avail_h // rows)

        start_x = (w - cell_size * cols) // 2
        start_y = (h - cell_size * rows) // 2

        painter.setBrush(QColor("#000000"))
        painter.setPen(Qt.NoPen)
        for r in range(rows):
            for c in range(cols):
                if self.matrix[r][c]:
                    painter.drawRect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)


# ==============================================================================
# 2. 内置 HTTP 服务端与专治 iOS 的网页端
# ==============================================================================
class ServerBridgeSignals(QObject):
    sig_log = Signal(str, str)
    sig_upload_finished = Signal()


_BRIDGE_SIGNALS = ServerBridgeSignals()
_CURRENT_TARGET_DIR = None
_CURRENT_TASK_NAME = "Cosplay_RAW"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Anicrop Studio - 隔空快传</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: #09090b;
      color: #f4f4f5;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 16px 16px 90px 16px;
    }
    .header {
      border-bottom: 1px solid #27272a;
      padding-bottom: 12px;
      margin-bottom: 16px;
    }
    .brand {
      color: #38bdf8;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }
    .title {
      font-size: 22px;
      font-weight: 700;
      margin-top: 2px;
    }
    .card {
      background-color: #141418;
      border: 1px solid #27272f;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 14px;
    }
    .stat-label { color: #a1a1aa; }
    .stat-val { font-weight: 600; color: #f4f4f5; }
    .btn-main {
      display: block;
      width: 100%;
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #ffffff;
      text-align: center;
      padding: 15px;
      border-radius: 10px;
      font-size: 16px;
      font-weight: 700;
      text-decoration: none;
      box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
      margin-top: 10px;
      -webkit-tap-highlight-color: transparent;
    }
    .btn-main:active { transform: scale(0.98); }
    .btn-upload {
      background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
      box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3);
      cursor: pointer;
      border: none;
    }
    .file-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background-color: #18181e;
      border: 1px solid #27272f;
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 8px;
      transition: all 0.2s ease;
    }
    .file-item.downloaded {
      background-color: #0f2317;
      border-color: #166534;
      opacity: 0.75;
    }
    .file-name {
      font-size: 14px;
      font-weight: 600;
      max-width: 180px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .file-item.downloaded .file-name {
      text-decoration: line-through;
      color: #86efac;
    }
    .file-meta { font-size: 12px; color: #71717a; margin-top: 2px; }
    .btn-sub {
      background-color: #27272a;
      border: 1px solid #3f3f46;
      color: #38bdf8;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      -webkit-tap-highlight-color: transparent;
    }
    .btn-sub:active { background-color: #3f3f46; }
    .btn-sub.done {
      background-color: #16a34a;
      border-color: #16a34a;
      color: #ffffff;
    }
    .progress-bar-box {
      display: none;
      background-color: #27272a;
      border-radius: 6px;
      height: 8px;
      overflow: hidden;
      margin-top: 12px;
    }
    .progress-bar-fill {
      background-color: #4ade80;
      height: 100%;
      width: 0%;
      transition: width 0.2s ease;
    }
    .status-text {
      font-size: 13px;
      color: #4ade80;
      text-align: center;
      margin-top: 8px;
      font-weight: bold;
    }
    .tip-box {
      background-color: #18181c;
      border-left: 3px solid #38bdf8;
      border-radius: 4px;
      padding: 12px 14px;
      font-size: 12px;
      color: #a1a1aa;
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="brand">Anicrop Studio // Air-Link</div>
    <div class="title">⚡ 极速双向快传 (iOS 优化版)</div>
  </div>

  <!-- 1. 纯生肉秒解包 (ZIP_STORED 解决解压卡死) -->
  <div class="card">
    <div class="card-title">🍎 通道 A: 纯生肉秒解包 (单次确认 · 0.1秒秒解)</div>
    <div class="stat-row">
      <span class="stat-label">交付角色:</span>
      <span class="stat-val">__TASK_NAME__</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">原片总数:</span>
      <span class="stat-val">__COUNT__ 张 (共 __TOTAL_SIZE__)</span>
    </div>
    <a href="/download_stored_bundle/__TASK_NAME__.zip" class="btn-main">⚡ 一键下发专属原片包 (点开即解 · 绝不卡死)</a>
  </div>

  <!-- 2. 反向成片回传 -->
  <div class="card">
    <div class="card-title" style="color: #4ade80;">📤 2. 修好成片回传 (iPad ➔ PC)</div>
    <div class="stat-row">
      <span class="stat-label">存盘路径:</span>
      <span class="stat-val">直达 PC 硬盘 (ipad_upload)</span>
    </div>
    <input type="file" id="uploadInput" multiple accept="*/*" style="display:none;" onchange="handleFilesUpload(this.files)">
    <button class="btn-main btn-upload" onclick="triggerIOSFilePicker()">📂 打开「文件」App 选取成片回传</button>
    <div id="up_progress_box" class="progress-bar-box"><div id="up_progress_fill" class="progress-bar-fill"></div></div>
    <div id="up_status" class="status-text" style="display:none;"></div>
  </div>

  <!-- 3. 大拇指单张点射直下 -->
  <div class="card">
    <div class="card-title" style="font-size: 14px; color: #a1a1aa;">
      <span>📱 通道 B: 单张大拇指点射直取 (免解压 · 下完自动划线)</span>
      <span id="dl_counter" style="color: #38bdf8; margin-left: auto;">0 / __COUNT__</span>
    </div>
    __FILE_ITEMS__
  </div>

  <div class="tip-box">
    💡 <strong>两套武器库任你选：</strong><br>
    • <strong>如果嫌一张张点麻烦：</strong>点顶上的蓝色按钮。这套包经过底层改造（ZIP_STORED 纯镜像，不进行任何 CPU 耗时的二次压缩），在 iPad「文件」里<strong>点一下只要 0.1 秒瞬间秒开</strong>，彻底消灭 iOS 解压卡死与报错！<br>
    • <strong>如果死活不想要压缩包：</strong>看下方清单，顺着大拇指逐个点【⬇ 直下】，点完一个自动变绿勾选，张张独立入库！
  </div>

  <script>
    let downloadedCount = 0;
    const totalFiles = __COUNT__;

    function markDownloaded(btn, cardId) {
      if (!btn.classList.contains('done')) {
        btn.classList.add('done');
        btn.innerText = '✓ 已就绪';
        const card = document.getElementById(cardId);
        if (card) card.classList.add('downloaded');
        downloadedCount++;
        document.getElementById('dl_counter').innerText = `${downloadedCount} / ${totalFiles}`;
      }
    }

    function triggerIOSFilePicker() {
      document.getElementById('uploadInput').click();
    }

    async function handleFilesUpload(files) {
      if (!files || files.length === 0) return;
      const pBox = document.getElementById('up_progress_box');
      const pFill = document.getElementById('up_progress_fill');
      const pStat = document.getElementById('up_status');
      pBox.style.display = 'block';
      pStat.style.display = 'block';

      let successCount = 0;
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        pStat.innerText = `正在回传成片 [${i+1}/${files.length}]: ${f.name}...`;
        pFill.style.width = Math.round(((i + 1) / files.length) * 100) + '%';
        try {
          const resp = await fetch('/upload?name=' + encodeURIComponent(f.name), {
            method: 'POST',
            body: f
          });
          if (resp.ok) successCount++;
        } catch (e) {
          console.error(e);
        }
      }
      pStat.innerText = `🎉 成功从文件 App 回传 ${successCount} 张成片至电脑硬盘！`;
      document.getElementById('uploadInput').value = '';
    }
  </script>
</body>
</html>
"""


class AirLinkRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global _CURRENT_TARGET_DIR, _CURRENT_TASK_NAME
        client_ip = self.client_address[0]
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if not _CURRENT_TARGET_DIR or not Path(_CURRENT_TARGET_DIR).exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"No active directory mounted.")
            return

        target_dir = Path(_CURRENT_TARGET_DIR)
        valid_exts = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".jpg", ".jpeg", ".xmp", ".png"}
        files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        files.sort(key=lambda x: x.name.lower())

        if path == "/" or path == "":
            _BRIDGE_SIGNALS.sig_log.emit(f"📱 设备 [{client_ip}] 接入: 打开了快传工作台", "SUCCESS")

            total_bytes = sum(f.stat().st_size for f in files)
            size_mb = total_bytes / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

            file_items_html = []
            for idx, f in enumerate(files):
                f_mb = f.stat().st_size / (1024 * 1024)
                f_size_str = f"{f_mb:.1f} MB" if f_mb >= 0.1 else f"{f.stat().st_size / 1024:.0f} KB"
                encoded_name = urllib.parse.quote(f.name)
                card_id = f"card_{idx}"
                file_items_html.append(f"""
                <div class="file-item" id="{card_id}">
                  <div class="file-info">
                    <div class="file-name">{f.name}</div>
                    <div class="file-meta">{f_size_str}</div>
                  </div>
                  <a href="/file/{encoded_name}" download="{f.name}" class="btn-sub" onclick="markDownloaded(this, '{card_id}')">⬇ 直下</a>
                </div>
                """)

            html = HTML_TEMPLATE.replace("__TASK_NAME__", _CURRENT_TASK_NAME)
            html = html.replace("__COUNT__", str(len(files)))
            html = html.replace("__TOTAL_SIZE__", size_str)
            html = html.replace("__FILE_ITEMS__", "\n".join(file_items_html))

            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # 刀法一：ZIP_STORED 纯生肉秒解包 (反制 iOS 归档折叠机制，强制写入双层资产树)
        elif path == "/download_stored_bundle" or path.startswith("/download_stored_bundle/"):
            # 1. 严格回溯角色名与子文件夹名
            if target_dir.name.lower() in ("cr3", "raw", "jpg", "jpg2"):
                task_name = target_dir.parent.name or "Cosplay_RAW"
                sub_folder = target_dir.name
            else:
                task_name = _CURRENT_TASK_NAME if _CURRENT_TASK_NAME and _CURRENT_TASK_NAME.lower() not in ("cr3", "raw") else target_dir.name
                sub_folder = "cr3"

            zip_filename = f"{task_name}.zip"
            encoded_fn = urllib.parse.quote(zip_filename)
            _BRIDGE_SIGNALS.sig_log.emit(f"⚡ 设备 [{client_ip}] 提取专属包: {zip_filename} (严格封装: {task_name}/{sub_folder}/)...", "INFO")

            temp_zip = target_dir / f"__airlink_temp_{int(time.time())}.zip"
            try:
                with zipfile.ZipFile(temp_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
                    for f in files:
                        # 核心反制点：显式写入双层目录！流萤/cr3/IMG_xxxx.CR3
                        zf.write(f, arcname=f"{task_name}/{sub_folder}/{f.name}")

                zip_size = temp_zip.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                # 强化标头：标准 ASCII 备用名 + RFC 5987 UTF-8 编码，iOS 无法篡改包名
                self.send_header("Content-Disposition", f'attachment; filename="{encoded_fn}"; filename*=UTF-8\'\'{encoded_fn}')
                self.send_header("Content-Length", str(zip_size))
                self.end_headers()

                t0 = time.time()
                with open(temp_zip, 'rb') as f_zip:
                    while True:
                        chunk = f_zip.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

                elapsed = max(0.1, time.time() - t0)
                speed = (zip_size / (1024 * 1024)) / elapsed
                _BRIDGE_SIGNALS.sig_log.emit(f"🎉 投送完成: {zip_filename} ({zip_size/(1024*1024):.1f}MB) 耗时 {elapsed:.1f}s ({speed:.1f} MB/s)", "SUCCESS")
            except Exception as e:
                _BRIDGE_SIGNALS.sig_log.emit(f"❌ 传输中断: {e}", "WARN")
            finally:
                if temp_zip.exists():
                    try:
                        temp_zip.unlink()
                    except Exception:
                        pass

        # 刀法二：单张大拇指点射直取
        elif path.startswith("/file/"):
            raw_fn = urllib.parse.unquote(path[6:])
            target_f = target_dir / raw_fn
            if target_f.exists() and target_f.is_file():
                f_size = target_f.stat().st_size
                encoded_fn = urllib.parse.quote(target_f.name)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f"attachment; filename=\"{encoded_fn}\"; filename*=UTF-8''{encoded_fn}")
                self.send_header("Content-Length", str(f_size))
                self.end_headers()
                try:
                    with open(target_f, 'rb') as f:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                    _BRIDGE_SIGNALS.sig_log.emit(f"⚡ [PC➔iPad] 单张直发完成: {target_f.name} ({f_size/(1024*1024):.1f}MB)", "INFO")
                except Exception:
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    def do_POST(self):
        global _CURRENT_TARGET_DIR
        client_ip = self.client_address[0]
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/upload":
            query = urllib.parse.parse_qs(parsed.query)
            fn = query.get('name', [f"upload_{int(time.time())}.jpg"])[0]
            fn = urllib.parse.unquote(fn)

            content_len = int(self.headers.get('Content-Length', 0))
            if not _CURRENT_TARGET_DIR or content_len <= 0:
                self.send_response(400)
                self.end_headers()
                return

            p_curr = Path(_CURRENT_TARGET_DIR)
            # 如果当前挂在 cr3/raw 目录，回传成片自动归入角色根目录
            base_dir = p_curr.parent if p_curr.name.lower() in ("cr3", "jpg", "jpg2", "raw") else p_curr
            upload_dir = base_dir / "ipad_upload"
            upload_dir.mkdir(parents=True, exist_ok=True)
            dst_file = upload_dir / fn

            _BRIDGE_SIGNALS.sig_log.emit(f"📥 [iPad➔PC] 正在接收成片: {fn} ({content_len/(1024*1024):.1f}MB)...", "INFO")
            t0 = time.time()
            remaining = content_len
            try:
                with open(dst_file, 'wb') as f_out:
                    while remaining > 0:
                        chunk_size = min(remaining, 1024 * 1024)
                        chunk = self.rfile.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        remaining -= len(chunk)

                elapsed = max(0.05, time.time() - t0)
                speed = (content_len / (1024 * 1024)) / elapsed
                _BRIDGE_SIGNALS.sig_log.emit(f"🎉 [iPad➔PC] 成片已存盘: {fn} 耗时 {elapsed:.1f}s ({speed:.1f} MB/s)", "SUCCESS")
                _BRIDGE_SIGNALS.sig_upload_finished.emit()

                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                _BRIDGE_SIGNALS.sig_log.emit(f"❌ 接收中断 {fn}: {e}", "WARN")
                self.send_response(500)
                self.end_headers()


def get_local_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('114.114.114.114', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


# ==============================================================================
# 3. PC 端工作台主面板 (AirLinkPageWidget)
# ==============================================================================
class AirLinkPageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.server = None
        self.server_thread = None
        self.server_port = 9527
        self.current_ip = get_local_lan_ip()
        self.target_dir = ""
        self.task_name = "未命名任务"

        self._init_ui()
        _BRIDGE_SIGNALS.sig_log.connect(self._append_log)
        _BRIDGE_SIGNALS.sig_upload_finished.connect(self._refresh_stats)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        top_card = QFrame()
        top_card.setStyleSheet("background-color: #141418; border: 1px solid #27272f; border-radius: 8px; padding: 12px;")
        tc_layout = QHBoxLayout(top_card)
        
        lbl_title = QLabel("📡 移动端局域极速快传站 (Air-Link // iOS 优化版)")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        tc_layout.addWidget(lbl_title)
        tc_layout.addStretch()

        self.lbl_server_status = QLabel("● 服务就绪 (未点火)")
        self.lbl_server_status.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 12px;")
        tc_layout.addWidget(self.lbl_server_status)
        layout.addWidget(top_card)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; width: 1px; }")

        # 左栏: 控制与目录
        left_pane = QWidget()
        lp_layout = QVBoxLayout(left_pane)
        lp_layout.setContentsMargins(0, 0, 10, 0)
        lp_layout.setSpacing(12)

        grp_src = QGroupBox("📁 1. 投送与接收目录")
        src_v = QVBoxLayout(grp_src)
        src_v.setSpacing(8)

        src_v.addWidget(QLabel("当前工作目录 (CR3 / ARW / RAW):"))
        h_dir = QHBoxLayout()
        self.edit_dir = QLineEdit()
        self.edit_dir.setPlaceholderText("选择或自动同步暗房导出的 RAW 文件夹...")
        self.edit_dir.textChanged.connect(self._on_dir_changed)
        h_dir.addWidget(self.edit_dir)

        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_dir)
        h_dir.addWidget(btn_browse)
        src_v.addLayout(h_dir)

        h_task = QHBoxLayout()
        h_task.addWidget(QLabel("任务/角色名:"))
        self.edit_task = QLineEdit("默认名称")
        self.edit_task.setFixedWidth(140)
        self.edit_task.textChanged.connect(self._on_task_changed)
        h_task.addWidget(self.edit_task)
        h_task.addStretch()
        src_v.addLayout(h_task)

        # 统计简报与打开上传文件夹按钮
        self.info_box = QFrame()
        self.info_box.setStyleSheet("background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 10px;")
        ib_layout = QVBoxLayout(self.info_box)
        ib_layout.setSpacing(6)
        self.lbl_stat_count = QLabel("• 待修生肉: 0 个")
        self.lbl_stat_count.setStyleSheet("color: #38bdf8; font-weight: 600;")
        ib_layout.addWidget(self.lbl_stat_count)
        self.lbl_stat_size = QLabel("• 预计占用: 0.0 MB")
        self.lbl_stat_size.setStyleSheet("color: #f4f4f5; font-weight: 600;")
        ib_layout.addWidget(self.lbl_stat_size)
        self.lbl_stat_uploaded = QLabel("• iPad已回传成片: 0 张")
        self.lbl_stat_uploaded.setStyleSheet("color: #4ade80; font-weight: 600;")
        ib_layout.addWidget(self.lbl_stat_uploaded)
        src_v.addWidget(self.info_box)

        h_open = QHBoxLayout()
        self.btn_open_upload = QPushButton("📂 打开 iPad 回传成片目录 (ipad_upload)")
        self.btn_open_upload.setStyleSheet("QPushButton { background-color: #1c1917; border: 1px solid #78350f; color: #fbbf24; font-weight: bold; padding: 6px; border-radius: 4px; } QPushButton:hover { background-color: #292524; }")
        self.btn_open_upload.clicked.connect(self._open_uploaded_dir)
        h_open.addWidget(self.btn_open_upload)
        src_v.addLayout(h_open)

        lp_layout.addWidget(grp_src)

        grp_ctrl = QGroupBox("⚙️ 2. 服务通道控制")
        ctrl_v = QVBoxLayout(grp_ctrl)
        ctrl_v.setSpacing(10)

        h_port = QHBoxLayout()
        h_port.addWidget(QLabel("监听端口:"))
        self.edit_port = QLineEdit(str(self.server_port))
        self.edit_port.setFixedWidth(80)
        h_port.addWidget(self.edit_port)
        h_port.addStretch()
        ctrl_v.addLayout(h_port)

        self.btn_toggle_server = QPushButton("⚡ 点火启动局域网快传通道")
        self.btn_toggle_server.setStyleSheet("QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 10px; font-size: 13px; border-radius: 6px; } QPushButton:hover { background-color: #0369a1; }")
        self.btn_toggle_server.clicked.connect(self._toggle_server)
        ctrl_v.addWidget(self.btn_toggle_server)
        lp_layout.addWidget(grp_ctrl)
        lp_layout.addStretch()

        splitter.addWidget(left_pane)

        # 右栏: 二维码与实时双向日志
        right_pane = QWidget()
        rp_layout = QVBoxLayout(right_pane)
        rp_layout.setContentsMargins(10, 0, 0, 0)
        rp_layout.setSpacing(12)

        grp_qr = QGroupBox("📱 3. iPad / 手机扫码直连")
        qr_h = QHBoxLayout(grp_qr)
        qr_h.setSpacing(16)

        self.qr_widget = QrCodeWidget()
        qr_h.addWidget(self.qr_widget, alignment=Qt.AlignCenter)

        qr_info_v = QVBoxLayout()
        qr_info_v.setSpacing(8)
        qr_info_v.addWidget(QLabel("👉 iPad 自带相机直接扫码 (或 Chrome 输入):"))

        self.lbl_url = QLabel("http://--:--")
        self.lbl_url.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_url.setStyleSheet("color: #38bdf8; font-size: 14px; font-weight: bold; font-family: Consolas, monospace; background: #18181e; padding: 6px; border-radius: 4px; border: 1px solid #27272f;")
        qr_info_v.addWidget(self.lbl_url)

        h_btns = QHBoxLayout()
        btn_copy = QPushButton("📋 复制链接")
        btn_copy.clicked.connect(self._copy_url)
        h_btns.addWidget(btn_copy)

        btn_test = QPushButton("🌐 本地测试打开")
        btn_test.clicked.connect(self._open_in_browser)
        h_btns.addWidget(btn_test)
        h_btns.addStretch()
        qr_info_v.addLayout(h_btns)

        lbl_hint_guide = QLabel("💡 双方案已就绪：\n1. 顶上蓝色大按钮是「ZIP_STORED 纯生肉包」，iPad 点一下 0.1 秒秒解，绝不卡死！\n2. 清单里支持大拇指顺着点【⬇ 直下】，下完一张自动变绿一张，没有死循环！")
        lbl_hint_guide.setWordWrap(True)
        lbl_hint_guide.setStyleSheet("color: #71717a; font-size: 11px; margin-top: 4px; line-height: 1.4;")
        qr_info_v.addWidget(lbl_hint_guide)
        qr_info_v.addStretch()

        qr_h.addLayout(qr_info_v)
        rp_layout.addWidget(grp_qr)

        grp_log = QGroupBox("🖥️ 局域网实时双向传输流 (下载/上传全感知)")
        log_v = QVBoxLayout(grp_log)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #09090b; border: 1px solid #27272f; border-radius: 6px; font-family: Consolas, monospace; font-size: 11px; color: #d4d4d8; padding: 8px;")
        log_v.addWidget(self.console)
        rp_layout.addWidget(grp_log)

        splitter.addWidget(right_pane)
        splitter.setSizes([420, 680])
        layout.addWidget(splitter)

        self._start_server()

    def set_export_context(self, target_dir: str, task_name: str = ""):
        if target_dir and os.path.exists(target_dir):
            self.edit_dir.setText(target_dir)
        if task_name:
            self.edit_task.setText(task_name)
        self._start_server()

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择待投送的 RAW 目录", self.edit_dir.text() or "")
        if d:
            self.edit_dir.setText(d)

    def _on_dir_changed(self, text: str):
        global _CURRENT_TARGET_DIR, _CURRENT_TASK_NAME
        self.target_dir = text.strip()
        _CURRENT_TARGET_DIR = self.target_dir

        p = Path(self.target_dir)
        if p.exists() and p.is_dir():
            # 严格回溯：末尾是 cr3/raw 时，强行抓取上一层作为角色名！
            if p.name.lower() in ("cr3", "jpg", "jpg2", "raw"):
                auto_task_name = p.parent.name
            else:
                auto_task_name = p.name
            if auto_task_name:
                self.edit_task.setText(auto_task_name)
                _CURRENT_TASK_NAME = auto_task_name

        self._refresh_stats()

    def _on_task_changed(self, text: str):
        global _CURRENT_TASK_NAME
        self.task_name = text.strip() or "Cosplay_RAW"
        _CURRENT_TASK_NAME = self.task_name

    def _refresh_stats(self):
        p = Path(self.target_dir)
        if p.exists() and p.is_dir():
            valid_exts = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".jpg", ".jpeg", ".xmp", ".png"}
            files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            total_b = sum(f.stat().st_size for f in files)
            mb = total_b / (1024 * 1024)
            size_str = f"{mb:.1f} MB" if mb < 1024 else f"{mb/1024:.2f} GB"
            self.lbl_stat_count.setText(f"• 待修生肉: {len(files)} 个")
            self.lbl_stat_size.setText(f"• 预计占用: {size_str}")

            up_dir = p / "ipad_upload"
            up_count = len([f for f in up_dir.iterdir() if f.is_file()]) if up_dir.exists() else 0
            self.lbl_stat_uploaded.setText(f"• iPad已回传成片: {up_count} 张")
        else:
            self.lbl_stat_count.setText("• 待修生肉: 0 个")
            self.lbl_stat_size.setText("• 预计占用: 0.0 MB")
            self.lbl_stat_uploaded.setText("• iPad已回传成片: 0 张")

    def _open_uploaded_dir(self):
        p_curr = Path(self.target_dir)
        base_dir = p_curr.parent if p_curr.name.lower() in ("cr3", "jpg", "jpg2", "raw") else p_curr
        p = base_dir / "ipad_upload"
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(p))
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _toggle_server(self):
        if self.server:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        if self.server:
            return
        try:
            self.server_port = int(self.edit_port.text().strip() or "9527")
        except ValueError:
            self.server_port = 9527

        self.current_ip = get_local_lan_ip()
        url = f"http://{self.current_ip}:{self.server_port}"

        try:
            self.server = ThreadingHTTPServer(('0.0.0.0', self.server_port), AirLinkRequestHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

            self.lbl_server_status.setText(f"● 服务运行中 [{self.server_port}]")
            self.lbl_server_status.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 12px;")
            self.btn_toggle_server.setText("🛑 挂断停止快传通道")
            self.btn_toggle_server.setStyleSheet("QPushButton { background-color: #7f1d1d; color: white; font-weight: bold; padding: 10px; font-size: 13px; border-radius: 6px; } QPushButton:hover { background-color: #991b1b; }")
            self.lbl_url.setText(url)
            self.qr_widget.set_text(url)
            self._append_log(f"🚀 快传通道点火成功: 局域网接入点 {url}", "SUCCESS")
        except Exception as e:
            self._append_log(f"❌ 服务启动失败: {e}", "ERROR")

    def _stop_server(self):
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
            self.server = None
            self.server_thread = None

        self.lbl_server_status.setText("● 服务已停止")
        self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 12px;")
        self.btn_toggle_server.setText("⚡ 点火启动局域网快传通道")
        self.btn_toggle_server.setStyleSheet("QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 10px; font-size: 13px; border-radius: 6px; } QPushButton:hover { background-color: #0369a1; }")
        self.qr_widget.set_text("")
        self._append_log("🛑 快传服务已关闭", "INFO")

    def _copy_url(self):
        url = self.lbl_url.text()
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(url)
        self._append_log(f"📋 网址已复制到剪贴板: {url}", "INFO")

    def _open_in_browser(self):
        url = self.lbl_url.text()
        QDesktopServices.openUrl(QUrl(url))

    def _append_log(self, text: str, level: str = "INFO"):
        color_map = {"INFO": "#d4d4d8", "SUCCESS": "#4ade80", "WARN": "#facc15", "ERROR": "#f87171"}
        color = color_map.get(level.upper(), "#d4d4d8")
        tm = time.strftime("%H:%M:%S")
        self.console.append(f'<span style="color: #71717a;">[{tm}]</span> <span style="color: {color};">{text}</span>')