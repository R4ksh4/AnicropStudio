from watermark_ui import WatermarkPageWidget
from boot_flash import BootFlash
import sys
import os
import shutil
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QCheckBox, QProgressBar, QSlider,
    QTextEdit, QFrame, QMessageBox, QGroupBox, QStackedWidget, QComboBox,
    QTabWidget, QDialog
)
from PySide6.QtCore import Qt, Signal, QSettings, QUrl
from PySide6.QtGui import QTextCursor, QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent, QDesktopServices

from raw_sorter import RawSorterWorker
from album import (
    AlbumPageWidget, RawGalleryPageWidget, DarkroomPageWidget, parse_photo_exif, get_app_cache_root,
    DEFAULT_SHORTCUTS, FIXED_SHORTCUTS, load_shortcuts, save_shortcut, reset_shortcuts
)
from indicator_sphere import IndicatorSphere
from splash_overlay import SplashOverlay

DARK_STYLE_SHEET = """
QMainWindow {
    background-color: #0e0e11;
}

QWidget {
    color: #e4e4e7;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

#TopNavWidget {
    background-color: #141418;
    border-bottom: 1px solid #27272f;
}

#AppTitleLabel {
    font-size: 13px;
    font-weight: bold;
    color: #38bdf8;
    padding-left: 6px;
    padding-right: 14px;
}

QPushButton.NavTab {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 5px 12px;
    font-weight: 600;
    font-size: 12px;
    color: #a1a1aa;
}

QPushButton.NavTab:hover {
    color: #f4f4f5;
    background-color: #1f1f24;
}

QPushButton.NavTabActive {
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
    background-color: #18181e;
}

QPushButton.NavTabDisabled {
    color: #52525b;
}

#LeftRailWidget {
    background-color: #141418;
    border-right: 1px solid #27272f;
}

QGroupBox {
    background-color: #141418;
    border: 1px solid #27272f;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #38bdf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

QFrame.DropBox {
    background-color: #18181e;
    border: 2px dashed #3f3f46;
    border-radius: 8px;
    padding: 8px;
}

QFrame.DropBoxHover {
    background-color: #0c4a6e;
    border: 2px dashed #38bdf8;
}

QFrame.DropBoxReady {
    background-color: #18181e;
    border: 1px solid #0284c7;
}

QLineEdit, QComboBox {
    background-color: #1c1c22;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 10px;
    color: #fafafa;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
    background-color: #202028;
}

QComboBox QAbstractItemView {
    background-color: #141418;
    border: 1px solid #3f3f46;
    selection-background-color: #0284c7;
    color: #f4f4f5;
    padding: 4px;
}

QPushButton {
    background-color: #1f1f24;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2f2f38;
}

QPushButton#PrimaryBtn {
    background-color: #0284c7;
    border: 1px solid #0284c7;
    color: #ffffff;
    font-weight: bold;
    padding: 8px 20px;
    font-size: 14px;
}

QPushButton#PrimaryBtn:hover {
    background-color: #0369a1;
}

QPushButton#DangerBtn {
    background-color: #991b1b;
    border: 1px solid #991b1b;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar {
    background-color: #1f1f24;
    border: 1px solid #3f3f46;
    border-radius: 6px;
    text-align: center;
    color: #f4f4f5;
    font-weight: bold;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 5px;
}

QTextEdit#LogConsole {
    background-color: #09090b;
    border: 1px solid #27272f;
    border-radius: 6px;
    font-family: "Consolas", "Cascadia Code", monospace;
    font-size: 12px;
    color: #d4d4d8;
    padding: 8px;
}

QTabWidget::pane {
    border: 1px solid #27272f;
    background-color: #141418;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #18181e;
    color: #a1a1aa;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #141418;
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
}

QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
"""


class PathDropBox(QFrame):
    path_changed = Signal(str)
    sig_log_notify = Signal(str, str)

    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.title_text = title
        self.subtitle_text = subtitle
        self.current_path = ""
        self.setAcceptDrops(True)
        self.setProperty("class", "DropBox")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(self.title_text)
        self.lbl_title.setStyleSheet("font-weight: bold; color: #f4f4f5; font-size: 13px;")
        self.lbl_title.setAcceptDrops(False)
        header_layout.addWidget(self.lbl_title)

        header_layout.addStretch()

        self.btn_browse = QPushButton("📁 浏览...")
        self.btn_browse.setAcceptDrops(False)
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        header_layout.addWidget(self.btn_browse)

        self.btn_clear = QPushButton("✕ 清空")
        self.btn_clear.setAcceptDrops(False)
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self.clear_path)
        header_layout.addWidget(self.btn_clear)

        layout.addLayout(header_layout)

        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText(self.subtitle_text)
        self.edit_path.setAcceptDrops(False)
        self.edit_path.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.edit_path)

        self.lbl_hint = QLabel("💡 拖拽文件夹扔到此方框内，或点击右上角浏览")
        self.lbl_hint.setStyleSheet("color: #71717a; font-size: 11px;")
        self.lbl_hint.setAcceptDrops(False)
        layout.addWidget(self.lbl_hint)

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasFormat("text/uri-list"):
            event.acceptProposedAction()
            self.setProperty("class", "DropBox DropBoxHover")
            self.style().polish(self)
            self.sig_log_notify.emit(f"🎯 识别到拖拽移入: [{self.title_text}]", "INFO")
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self._refresh_box_style()
        event.accept()

    def dropEvent(self, event: QDropEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            urls = mime.urls()
            if urls:
                local_path = urls[0].toLocalFile()
                p = Path(local_path)
                target_dir = str(p if p.is_dir() else p.parent)
                self.set_path(target_dir)
                self.sig_log_notify.emit(f"✅ 成功吸附路径: {target_dir}", "SUCCESS")
                event.acceptProposedAction()
        self._refresh_box_style()

    def _on_browse_clicked(self):
        selected_dir = QFileDialog.getExistingDirectory(self, f"选择 {self.title_text}", self.current_path or "")
        if selected_dir:
            self.set_path(selected_dir)

    def _on_text_changed(self, text: str):
        self.current_path = text.strip()
        self.btn_clear.setVisible(bool(self.current_path))
        self._refresh_box_style()
        self.path_changed.emit(self.current_path)

    def _refresh_box_style(self):
        self.setProperty("class", "DropBox DropBoxReady" if self.current_path else "DropBox")
        self.style().polish(self)

    def set_path(self, path: str):
        self.edit_path.setText(path)

    def get_path(self) -> str:
        return self.edit_path.text().strip()

    def clear_path(self):
        self.edit_path.clear()


class LogConsole(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setObjectName("LogConsole")
        self.setReadOnly(True)

    def append_log(self, text: str, level: str = "INFO"):
        color_map = {"INFO": "#d4d4d8", "SUCCESS": "#4ade80", "WARN": "#facc15", "ERROR": "#f87171"}
        color = color_map.get(level.upper(), "#d4d4d8")
        self.append(f'<span style="color: {color};">{text}</span>')
        self.moveCursor(QTextCursor.End)


class _KeyCaptureDialog(QDialog):
    """模态按键捕获：按一个 A~Z 字母键完成绑定，Esc 取消"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.key_name = ""
        self.setWindowTitle("设置快捷键")
        self.setModal(True)
        self.setFixedSize(340, 120)
        v = QVBoxLayout(self)
        lbl = QLabel("请按下新的快捷键（仅限 A ~ Z 字母键）\n按 Esc 取消")
        lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(lbl)

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.reject()
            return
        if Qt.Key_A <= k <= Qt.Key_Z:
            self.key_name = chr(k)
            self.accept()


# ==============================================================================
# 系统设置组件
# ==============================================================================
class SettingsPageWidget(QWidget):
    sig_default_path_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.settings_tabs = QTabWidget()

        # ---------------- 子页 1: 常规与缓存 ----------------
        general_page = QWidget()
        g_layout = QVBoxLayout(general_page)
        g_layout.setContentsMargins(16, 16, 16, 16)
        g_layout.setSpacing(16)

        group_pref = QGroupBox("⚙️ 通用体验偏好")
        gp_layout = QVBoxLayout(group_pref)
        gp_layout.setSpacing(14)

        gp_layout.addWidget(QLabel("📁 1. 默认相册根路径 (画廊相册启动时优先加载):"))
        h1 = QHBoxLayout()
        self.edit_default_album = QLineEdit()
        self.edit_default_album.setPlaceholderText("留空则不自动加载...")
        h1.addWidget(self.edit_default_album)

        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_default_path)
        h1.addWidget(btn_browse)

        btn_clear = QPushButton("清除")
        btn_clear.clicked.connect(self._clear_default_path)
        h1.addWidget(btn_clear)
        gp_layout.addLayout(h1)

        hint1 = QLabel("💡 提示：设置后，画廊相册启动将默认打开该目录。")
        hint1.setStyleSheet("color: #71717a; font-size: 11px;")
        gp_layout.addWidget(hint1)

        self.cb_show_face_rect = QCheckBox("🎯 显示人脸雷达检测框 (蓝色虚线框)")
        self.cb_show_face_rect.setStyleSheet("color: #e4e4e7; font-weight: 500;")
        _st = QSettings("AnicropGeeks", "AnicropStudio")
        self.cb_show_face_rect.setChecked(_st.value("show_face_radar_rect", True, type=bool))
        self.cb_show_face_rect.toggled.connect(self._on_toggle_show_face_rect)
        gp_layout.addWidget(self.cb_show_face_rect)

        hint_face = QLabel("💡 提示：关闭后仅隐藏蓝色虚线人脸框，R 键双眼极速特写与绿框合焦检测依旧完全生效。")
        hint_face.setStyleSheet("color: #71717a; font-size: 11px;")
        gp_layout.addWidget(hint_face)

        gp_layout.addWidget(QLabel("🖱️ 2. 鼠标滚轮滚动平滑度 (单次滚动位移距离):"))
        h_scroll = QHBoxLayout()
        self.slider_scroll_step = QSlider(Qt.Horizontal)
        self.slider_scroll_step.setRange(20, 180)
        self.slider_scroll_step.setValue(60)
        self.slider_scroll_step.setFixedWidth(200)
        h_scroll.addWidget(self.slider_scroll_step)

        self.lbl_scroll_step_val = QLabel("60 px (顺滑)")
        self.lbl_scroll_step_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        h_scroll.addWidget(self.lbl_scroll_step_val)
        h_scroll.addStretch()
        gp_layout.addLayout(h_scroll)

        hint_scroll = QLabel("💡 提示：数值越小滚动越细腻如丝，数值越大翻页越快；即时生效无需重启。")
        hint_scroll.setStyleSheet("color: #71717a; font-size: 11px;")
        gp_layout.addWidget(hint_scroll)

        gp_layout.addWidget(QLabel("🧪 3. 暗房选片模式选择 (决定初始状态与选片心流):"))
        self.combo_cull_mode = QComboBox()
        self.combo_cull_mode.addItem("批量亮片选暗片（默认全保留，手动剔除废片）", "reject")
        self.combo_cull_mode.addItem("批量暗片选亮片（默认全压暗，手动精选好片）", "pick")
        self.combo_cull_mode.currentIndexChanged.connect(self._on_cull_mode_changed)
        gp_layout.addWidget(self.combo_cull_mode)

        self.lbl_restart_hint = QLabel("💡 更改暗房模式后，重启软件即可彻底生效。")
        self.lbl_restart_hint.setStyleSheet("color: #71717a; font-size: 11px;")
        gp_layout.addWidget(self.lbl_restart_hint)

        gp_layout.addWidget(QLabel("🗄️ 4. 缩略图缓存 (.cache):"))
        self.lbl_cache_path = QLabel()
        self.lbl_cache_path.setStyleSheet("color: #38bdf8; font-size: 11px; font-family: Consolas, monospace;")
        self.lbl_cache_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gp_layout.addWidget(self.lbl_cache_path)

        self.lbl_cache_size = QLabel()
        self.lbl_cache_size.setStyleSheet("color: #f4f4f5; font-size: 12px; font-weight: bold;")
        gp_layout.addWidget(self.lbl_cache_size)

        h3 = QHBoxLayout()
        self.btn_clear_cache = QPushButton("🗑️ 清除缓存")
        self.btn_clear_cache.setObjectName("DangerBtn")
        self.btn_clear_cache.clicked.connect(self._clear_cache)
        h3.addWidget(self.btn_clear_cache)
        btn_refresh_cache = QPushButton("刷新统计")
        btn_refresh_cache.clicked.connect(self._refresh_cache_info)
        h3.addWidget(btn_refresh_cache)
        h3.addStretch()
        gp_layout.addLayout(h3)

        hint3 = QLabel("💡 清除仅删除缩略图缓存，不影响任何原始照片；下次浏览相册会重新预热。")
        hint3.setStyleSheet("color: #71717a; font-size: 11px;")
        gp_layout.addWidget(hint3)

# ---------------- 导出规则与命名自定义 (修复 QSS 污染 + 支持默认路径) ----------------
        gp_layout.addWidget(QLabel("📦 5. 暗房导出规则、命名定制与默认路径:"))
        
        export_box = QFrame()
        export_box.setObjectName("ExportBox")
        export_box.setStyleSheet("""
            QFrame#ExportBox {
                background-color: #18181e;
                border: 1px solid #27272f;
                border-radius: 8px;
                padding: 14px;
            }
            QFrame#ExportBox QLabel {
                border: none;
                background: transparent;
                color: #e4e4e7;
                font-size: 12px;
                padding: 0px;
            }
            QFrame#ExportBox QCheckBox {
                border: none;
                background: transparent;
                color: #f4f4f5;
                font-size: 12px;
                font-weight: 500;
                min-height: 26px;
                padding: 0px;
            }
            QFrame#ExportBox QLineEdit {
                background-color: #141418;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 5px 8px;
                color: #ffffff;
                font-size: 12px;
                min-height: 22px;
            }
            QFrame#ExportBox QLineEdit:focus {
                border: 1px solid #38bdf8;
                background-color: #1a1a22;
            }
            QFrame#ExportBox QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
                font-weight: bold;
                color: #e4e4e7;
                min-height: 22px;
            }
            QFrame#ExportBox QPushButton:hover {
                background-color: #3f3f46;
                color: #38bdf8;
            }
        """)
        eb_layout = QVBoxLayout(export_box)
        eb_layout.setSpacing(12)

        # 1. 文件夹开关与命名
        h_exp1 = QHBoxLayout()
        self.cb_exp_all = QCheckBox("导出完整暗房全量照片 (粗选全量)")
        h_exp1.addWidget(self.cb_exp_all)
        h_exp1.addStretch()
        h_exp1.addWidget(QLabel("文件夹名:"))
        self.edit_name_all = QLineEdit("jpg")
        self.edit_name_all.setFixedWidth(110)
        h_exp1.addWidget(self.edit_name_all)
        eb_layout.addLayout(h_exp1)

        h_exp2 = QHBoxLayout()
        self.cb_exp_sel = QCheckBox("导出精选出片照片 (未隐藏/已点亮)")
        h_exp2.addWidget(self.cb_exp_sel)
        h_exp2.addStretch()
        h_exp2.addWidget(QLabel("文件夹名:"))
        self.edit_name_sel = QLineEdit("jpg2")
        self.edit_name_sel.setFixedWidth(110)
        h_exp2.addWidget(self.edit_name_sel)
        eb_layout.addLayout(h_exp2)

        h_exp3 = QHBoxLayout()
        self.cb_exp_raw = QCheckBox("导出匹配 RAW 生肉 (CR3/ARW)")
        h_exp3.addWidget(self.cb_exp_raw)
        h_exp3.addStretch()
        h_exp3.addWidget(QLabel("文件夹名:"))
        self.edit_name_raw = QLineEdit("cr3")
        self.edit_name_raw.setFixedWidth(110)
        h_exp3.addWidget(self.edit_name_raw)
        eb_layout.addLayout(h_exp3)

        # 分割线
        sep_exp = QFrame()
        sep_exp.setFrameShape(QFrame.HLine)
        sep_exp.setStyleSheet("border-top: 1px solid #27272f; max-height: 1px;")
        eb_layout.addWidget(sep_exp)

        # 2. 默认工作根目录路径 (本地)
        eb_layout.addWidget(QLabel("📁 默认本地工作根目录 (包含全量 RAW 素材，导出时默认选中):"))
        h_def_work = QHBoxLayout()
        self.edit_def_work_dir = QLineEdit()
        self.edit_def_work_dir.setPlaceholderText("留空则每次手动选择...")
        h_def_work.addWidget(self.edit_def_work_dir)
        btn_b_def_work = QPushButton("浏览...")
        h_def_work.addWidget(btn_b_def_work)
        eb_layout.addLayout(h_def_work)

        # 3. 默认移动端修图根目录 (iPad)
        eb_layout.addWidget(QLabel("📱 默认移动端/异地修图根目录 (本机共享给 iPad 的目录，导出时默认选中):"))
        h_def_remote = QHBoxLayout()
        self.edit_def_remote_dir = QLineEdit()
        self.edit_def_remote_dir.setPlaceholderText("留空则不默认填入，如 D:\\iPad_Share...")
        h_def_remote.addWidget(self.edit_def_remote_dir)
        btn_b_def_remote = QPushButton("浏览...")
        h_def_remote.addWidget(btn_b_def_remote)
        eb_layout.addLayout(h_def_remote)

        hint_exp = QLabel("💡 提示：取消勾选则导出时不生成该项；名字与默认路径可随时定制并即时记忆。")
        hint_exp.setStyleSheet("color: #71717a; font-size: 11px;")
        eb_layout.addWidget(hint_exp)

        gp_layout.addWidget(export_box)

        # 持久化读取与绑定
        _st_exp = QSettings("AnicropGeeks", "AnicropStudio")
        self.cb_exp_all.setChecked(_st_exp.value("export_enable_all_jpg", True, type=bool))
        self.edit_name_all.setText(str(_st_exp.value("export_name_all_jpg", "jpg")))
        self.cb_exp_sel.setChecked(_st_exp.value("export_enable_selected_jpg", True, type=bool))
        self.edit_name_sel.setText(str(_st_exp.value("export_name_selected_jpg", "jpg2")))
        self.cb_exp_raw.setChecked(_st_exp.value("export_enable_raw", True, type=bool))
        self.edit_name_raw.setText(str(_st_exp.value("export_name_raw", "cr3")))
        self.edit_def_work_dir.setText(str(_st_exp.value("last_work_root_dir", "")))
        self.edit_def_remote_dir.setText(str(_st_exp.value("export_remote_path", "")))

        self.cb_exp_all.toggled.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_enable_all_jpg", v))
        self.edit_name_all.textChanged.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_name_all_jpg", v.strip() or "jpg"))
        self.cb_exp_sel.toggled.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_enable_selected_jpg", v))
        self.edit_name_sel.textChanged.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_name_selected_jpg", v.strip() or "jpg2"))
        self.cb_exp_raw.toggled.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_enable_raw", v))
        self.edit_name_raw.textChanged.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_name_raw", v.strip() or "cr3"))

        self.edit_def_work_dir.textChanged.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("last_work_root_dir", v.strip()))
        self.edit_def_remote_dir.textChanged.connect(lambda v: QSettings("AnicropGeeks", "AnicropStudio").setValue("export_remote_path", v.strip()))

        def _on_browse_def_work():
            d = QFileDialog.getExistingDirectory(self, "选择默认本地工作根目录", self.edit_def_work_dir.text() or "")
            if d:
                self.edit_def_work_dir.setText(d)
        btn_b_def_work.clicked.connect(_on_browse_def_work)

        def _on_browse_def_remote():
            d = QFileDialog.getExistingDirectory(self, "选择默认移动端修图根目录", self.edit_def_remote_dir.text() or "")
            if d:
                self.edit_def_remote_dir.setText(d)
        btn_b_def_remote.clicked.connect(_on_browse_def_remote)

        g_layout.addWidget(group_pref)
        g_layout.addStretch()

        # ---------------- 给常规设置套上优雅的平滑滚动条 ----------------
        from PySide6.QtWidgets import QScrollArea
        scroll_general = QScrollArea()
        scroll_general.setWidgetResizable(True)
        scroll_general.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_general.setWidget(general_page)
        self.settings_tabs.addTab(scroll_general, "常规偏好")

        # ---------------- 子页 2: RAW 分流调度工具 ----------------
        sorter_page = self._build_sorter_tab()
        self.settings_tabs.addTab(sorter_page, "⚡ RAW 分流归档")

        # ---------------- 子页 3: 快捷键自定义 ----------------
        shortcut_page = self._build_shortcut_page()
        self.settings_tabs.addTab(shortcut_page, "快捷键映射")

        # ---------------- 子页 4: 关于与许可 (独立关于子页) ----------------
        about_page = self._build_about_page()
        self.settings_tabs.addTab(about_page, "关于与许可")

        layout.addWidget(self.settings_tabs)

        settings = QSettings("AnicropGeeks", "AnicropStudio")
        saved_path = settings.value("default_album_path", "")
        self.edit_default_album.setText(saved_path)
        saved_step = int(settings.value("mouse_scroll_step", 60))
        self.slider_scroll_step.setValue(saved_step)
        self._update_scroll_step_label(saved_step)
        self.slider_scroll_step.valueChanged.connect(self._on_scroll_step_changed)

        self.combo_cull_mode.blockSignals(True)
        saved_mode = settings.value("darkroom_cull_mode", "reject")
        idx = 1 if saved_mode == "pick" else 0
        self.combo_cull_mode.setCurrentIndex(idx)
        self.combo_cull_mode.blockSignals(False)

    def _build_sorter_tab(self) -> QWidget:
        tab = QWidget()
        work_layout = QVBoxLayout(tab)
        work_layout.setContentsMargins(12, 12, 12, 12)
        work_layout.setSpacing(10)

        path_group = QGroupBox("📁 路径配置（大方框支持将文件夹直接扔入）")
        path_layout = QVBoxLayout(path_group)
        self.box_jpg2 = PathDropBox("1. 细筛精选照片目录 (jpg2)", "拖入装有细筛 JPG 的文件夹，或点击浏览...")
        path_layout.addWidget(self.box_jpg2)
        self.box_raw_src = PathDropBox("2. RAW 全量源目录 (机械盘 / SD卡)", "拖入包含全量 CR3/ARW 的原始照片池...")
        path_layout.addWidget(self.box_raw_src)
        self.box_cr3_dst = PathDropBox("3. RAW 目标输出目录 (cr3/arw)", "拖入希望导出存放筛选 RAW 的最终目录...")
        path_layout.addWidget(self.box_cr3_dst)
        work_layout.addWidget(path_group)

        control_group = QGroupBox("⚙️ 调度策略与任务控制")
        control_layout = QVBoxLayout(control_group)
        opts_layout = QHBoxLayout()
        self.cb_strict_cr3 = QCheckBox("严格锁死 RAW 后缀 (.cr3 / .arw)")
        self.cb_strict_cr3.setChecked(True)
        self.cb_strict_cr3.setEnabled(False)
        opts_layout.addWidget(self.cb_strict_cr3)

        self.cb_tolerance = QCheckBox("开启相机前缀容错 (IMG_/_MG_/DSC)")
        self.cb_tolerance.setChecked(True)
        opts_layout.addWidget(self.cb_tolerance)

        self.cb_skip_exist = QCheckBox("目标文件已存在时自动跳过")
        self.cb_skip_exist.setChecked(True)
        opts_layout.addWidget(self.cb_skip_exist)
        opts_layout.addStretch()
        control_layout.addLayout(opts_layout)

        btns_layout = QHBoxLayout()
        self.btn_start = QPushButton("⚡ 开始分流调度")
        self.btn_start.setObjectName("PrimaryBtn")
        self.btn_start.clicked.connect(self._start_task)
        btns_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("🛑 中止任务")
        self.btn_cancel.setObjectName("DangerBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_task)
        btns_layout.addWidget(self.btn_cancel)

        btns_layout.addStretch()
        control_layout.addLayout(btns_layout)
        work_layout.addWidget(control_group)

        progress_group = QGroupBox("📊 执行进度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("就绪: 等待拖入路径并启动分流任务")
        self.lbl_status.setStyleSheet("color: #a1a1aa;")
        progress_layout.addWidget(self.lbl_status)
        work_layout.addWidget(progress_group)

        log_group = QGroupBox("🖥️ 终端日志控制台")
        log_layout = QVBoxLayout(log_group)
        self.console = LogConsole()
        log_layout.addWidget(self.console)

        log_actions = QHBoxLayout()
        log_actions.addStretch()
        btn_clear_log = QPushButton("清空控制台")
        btn_clear_log.clicked.connect(self.console.clear)
        log_actions.addWidget(btn_clear_log)
        log_layout.addLayout(log_actions)
        work_layout.addWidget(log_group)

        self.box_jpg2.sig_log_notify.connect(self.console.append_log)
        self.box_raw_src.sig_log_notify.connect(self.console.append_log)
        self.box_cr3_dst.sig_log_notify.connect(self.console.append_log)

        return tab

    def _start_task(self):
        jpg2_path = self.box_jpg2.get_path()
        raw_src_path = self.box_raw_src.get_path()
        cr3_dst_path = self.box_cr3_dst.get_path()

        if not jpg2_path or not raw_src_path or not cr3_dst_path:
            QMessageBox.warning(self, "路径缺失", "请先将对应目录拖入方框中！")
            return

        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("last_raw_source_dir", raw_src_path)

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在执行分流调度...")

        self.worker = RawSorterWorker(
            jpg2_dir=jpg2_path,
            raw_source_dir=raw_src_path,
            cr3_dest_dir=cr3_dst_path,
            tolerance_prefix=self.cb_tolerance.isChecked(),
            skip_existing=self.cb_skip_exist.isChecked()
        )
        self.worker.sig_log.connect(self.console.append_log)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_finished.connect(self._on_finished)
        self.worker.start()

    def _cancel_task(self):
        if self.worker and self.worker.isRunning():
            self.console.append_log("⚠️ 收到用户中止指令...", "WARN")
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, current: int, total: int, filename: str):
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(f"搬运中 [{current}/{total}] ({percent}%): {filename}")

    def _on_finished(self, summary: dict):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if summary.get("canceled", False):
            self.lbl_status.setText("🛑 任务已取消")
            return
        if not summary.get("success", False):
            self.lbl_status.setText("❌ 任务异常终止")
            QMessageBox.critical(self, "任务失败", summary.get("error", "未知错误"))
            return

        self.progress_bar.setValue(100)
        self.lbl_status.setText("✅ 分流任务全部完成！")
        msg = (
            f"分流统计报告：\n\n"
            f"• 细筛 JPG 总量: {summary['total_jpg']} 张\n"
            f"• 成功匹配 RAW: {summary['matched_count']} 张\n"
            f"• 实际写入新文件: {summary['copied_count']} 张\n"
            f"• 缺失 RAW 孤儿图: {summary['orphan_count']} 张\n"
            f"• 总耗时: {summary['elapsed_sec']:.2f} 秒"
        )
        QMessageBox.information(self, "调度完成", msg)

    def _build_shortcut_page(self) -> QWidget:
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(16)

        group = QGroupBox("⌨️ 检视与对比快捷键（点击按键徽章，按下 A~Z 字母即可改绑）")
        g = QVBoxLayout(group)
        g.setSpacing(10)

        self._shortcut_key_btns = {}
        sc = load_shortcuts()
        for action, (default, desc, alt) in DEFAULT_SHORTCUTS.items():
            row = QHBoxLayout()
            lbl = QLabel(desc)
            lbl.setMinimumWidth(240)
            row.addWidget(lbl)

            key_btn = QPushButton(sc[action])
            key_btn.setFixedWidth(90)
            key_btn.setStyleSheet("QPushButton { font-weight: bold; color: #38bdf8; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 10px; background-color: #18181e; font-family: Consolas, monospace; } QPushButton:hover { border-color: #38bdf8; }")
            key_btn.clicked.connect(lambda checked=False, a=action, b=key_btn: self._capture_shortcut(a, b))
            self._shortcut_key_btns[action] = key_btn
            row.addWidget(key_btn)

            alt_lbl = QLabel(f"固定备用键: {alt}" if alt else "")
            alt_lbl.setStyleSheet("color: #71717a; font-size: 11px;")
            row.addWidget(alt_lbl)
            row.addStretch()
            g.addLayout(row)
        v.addWidget(group)

        group2 = QGroupBox("🔒 系统保留快捷键（与交互体系深度绑定，不可修改）")
        g2 = QVBoxLayout(group2)
        g2.setSpacing(6)
        for keys, desc in FIXED_SHORTCUTS:
            row = QHBoxLayout()
            k = QLabel(keys)
            k.setMinimumWidth(170)
            k.setStyleSheet("font-weight: bold; color: #a1a1aa; font-family: Consolas, monospace;")
            row.addWidget(k)
            row.addWidget(QLabel(desc))
            row.addStretch()
            g2.addLayout(row)
        v.addWidget(group2)

        h = QHBoxLayout()
        btn_reset = QPushButton("↩️ 恢复默认快捷键")
        btn_reset.clicked.connect(self._reset_shortcuts)
        h.addWidget(btn_reset)
        hint = QLabel("💡 仅允许绑定 A~Z 字母键；修改即时生效，无需重启；冲突按键会被拒绝。")
        hint.setStyleSheet("color: #71717a; font-size: 11px;")
        h.addWidget(hint)
        h.addStretch()
        v.addLayout(h)
        v.addStretch()
        scroll.setWidget(page)
        return scroll

    def _build_about_page(self) -> QWidget:
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(14)

        group_about = QGroupBox("📜 关于项目与开源协议")
        about_layout = QVBoxLayout(group_about)
        about_layout.setSpacing(12)
        about_layout.setContentsMargins(16, 18, 16, 16)

        lbl_app_info = QLabel("Anicrop Studio (Beta 1.0.1)")
        lbl_app_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        about_layout.addWidget(lbl_app_info)


        lbl_desc = QLabel(
            "Anicrop Studio 是一款专为二次元漫展返图流与高频选片打造的高性能本地工作台。\n"
            "本项目基于 Apache License 2.0 协议完全免费开源，不包含任何商业收费模块，且不收集、不上报用户照片及隐私数据。"
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #d4d4d8; font-size: 12px; line-height: 1.5;")
        about_layout.addWidget(lbl_desc)

        # 防倒卖与版权警示框
        notice_frame = QFrame()
        notice_frame.setStyleSheet("""
            QFrame {
                background-color: #1c1917;
                border: 1px solid #78350f;
                border-radius: 6px;
                padding: 12px;
            }
            QLabel {
                color: #fbbf24;
                font-size: 12px;
                line-height: 1.5;
            }
        """)
        notice_layout = QVBoxLayout(notice_frame)
        notice_layout.setContentsMargins(8, 6, 8, 6)
        lbl_notice = QLabel(
            "⚠️ 正版与防侵权声明：\n"
            "本软件源代码与便携发布包仅在官方 GitHub 仓库和作者粉丝群公开分发，以及作者内部测试包私发。\n"
            "根据 Apache License 2.0 条款，任何二次分发、衍生开发均须保留原始版权声明及许可证文本。\n"
            "若您通过电商平台或其他有偿渠道购买了本软件，说明该渠道存在违规侵权销售行为，建议向平台举报并申请全额退款。"
        )
        lbl_notice.setWordWrap(True)
        notice_layout.addWidget(lbl_notice)
        about_layout.addWidget(notice_frame)

        # 官方直达跳转按钮组
        links_layout = QHBoxLayout()
        links_layout.setSpacing(10)

        btn_repo = QPushButton("🌐 GitHub 官方开源仓库")
        btn_repo.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                padding: 7px 16px;
                font-size: 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        btn_repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/R4ksh4/AnicropStudio")))
        links_layout.addWidget(btn_repo)

        btn_releases = QPushButton("📦 历史版本下载和更新")
        btn_releases.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                color: #e4e4e7;
                font-weight: 500;
                padding: 7px 16px;
                font-size: 12px;
                border: 1px solid #3f3f46;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3f3f46; color: #38bdf8; }
        """)
        btn_releases.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/R4ksh4/AnicropStudio/tags")))
        links_layout.addWidget(btn_releases)

        btn_author = QPushButton("👤 开发者主页 (R4ksh4)")
        btn_author.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                color: #e4e4e7;
                font-weight: 500;
                padding: 7px 16px;
                font-size: 12px;
                border: 1px solid #3f3f46;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3f3f46; color: #38bdf8; }
        """)
        btn_author.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/R4ksh4")))
        links_layout.addWidget(btn_author)
        lbl_signature = QLabel("Design by R4ksh4")
        lbl_signature.setStyleSheet("font-size: 20px; font-weight: 600;  font-family: Consolas, 'Segoe UI', monospace; margin-top: 4px;")
        about_layout.addWidget(lbl_signature)

        links_layout.addStretch()
        about_layout.addLayout(links_layout)

        v.addWidget(group_about)
        v.addStretch()

        scroll.setWidget(page)
        return scroll

    def _capture_shortcut(self, action: str, btn: QPushButton):
        dlg = _KeyCaptureDialog(self)
        if dlg.exec() != QDialog.Accepted or not dlg.key_name:
            return
        name = dlg.key_name
        sc = load_shortcuts()
        for a, cur in sc.items():
            if a != action and cur == name:
                QMessageBox.warning(self, "快捷键冲突", f"按键 {name} 已绑定到「{DEFAULT_SHORTCUTS[a][1]}」。\n请换一个按键。")
                return
        save_shortcut(action, name)
        btn.setText(name)

    def _reset_shortcuts(self):
        reset_shortcuts()
        sc = load_shortcuts()
        for a, b in self._shortcut_key_btns.items():
            b.setText(sc[a])

    def _browse_default_path(self):
        d = QFileDialog.getExistingDirectory(self, "选择默认相册根目录", self.edit_default_album.text() or "")
        if d:
            self.edit_default_album.setText(d)
            settings = QSettings("AnicropGeeks", "AnicropStudio")
            settings.setValue("default_album_path", d)
            self.sig_default_path_changed.emit(d)

    def _clear_default_path(self):
        self.edit_default_album.clear()
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("default_album_path", "")
        self.sig_default_path_changed.emit("")

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_cache_info()

    def _refresh_cache_info(self):
        root = get_app_cache_root()
        self.lbl_cache_path.setText(f"📁 {root}")
        total_bytes, file_cnt = 0, 0
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    fp = os.path.join(dirpath, fn)
                    try:
                        total_bytes += os.path.getsize(fp)
                        file_cnt += 1
                    except OSError:
                        pass
        except Exception:
            pass
        if total_bytes >= 1024 ** 3:
            size_str = f"{total_bytes / 1024 ** 3:.2f} GB"
        elif total_bytes >= 1024 ** 2:
            size_str = f"{total_bytes / 1024 ** 2:.1f} MB"
        else:
            size_str = f"{total_bytes / 1024:.0f} KB"
        self.lbl_cache_size.setText(f"当前占用: {size_str}（{file_cnt} 个缓存文件）")

    def _clear_cache(self):
        root = get_app_cache_root()
        ret = QMessageBox.question(
            self, "确认清除缓存",
            f"将删除以下目录下的全部缓存文件：\n{root}\n\n"
            "删除后下次浏览相册需重新预热生成，不影响任何原始照片。\n\n确认清除？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return
        deleted, failed = 0, 0
        try:
            for child in root.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=False)
                    else:
                        child.unlink()
                    deleted += 1
                except Exception:
                    failed += 1
        except Exception as e:
            QMessageBox.critical(self, "清除失败", f"缓存目录访问异常: {e}")
            return
        if failed:
            QMessageBox.warning(self, "部分清除", f"已清除 {deleted} 项，{failed} 项被占用无法删除。")
        else:
            QMessageBox.information(self, "清除完成", f"缓存已全部清空，共删除 {deleted} 项。")
        self._refresh_cache_info()

    def _update_scroll_step_label(self, val: int):
        desc = "细腻" if val <= 40 else ("顺滑" if val <= 90 else "高速")
        self.lbl_scroll_step_val.setText(f"{val} px ({desc})")

    def _on_scroll_step_changed(self, val: int):
        self._update_scroll_step_label(val)
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("mouse_scroll_step", val)
        win = self.window()
        if hasattr(win, "page_album"):
            win.page_album.apply_scroll_step(val)
        if hasattr(win, "page_darkroom"):
            win.page_darkroom.apply_scroll_step(val)

    def _on_cull_mode_changed(self):
        mode_val = self.combo_cull_mode.currentData()
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("darkroom_cull_mode", mode_val)
        self.lbl_restart_hint.setText(f"⚠️ 设置已保存：【{self.combo_cull_mode.currentText()}】，重启应用后生效！")
        self.lbl_restart_hint.setStyleSheet("color: #facc15; font-weight: bold; font-size: 11px;")

    def _on_toggle_show_face_rect(self, checked: bool):
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("show_face_radar_rect", checked)
        win = self.window()
        if hasattr(win, "page_album"):
            win.page_album.photo_viewer.update()
        if hasattr(win, "page_darkroom"):
            win.page_darkroom.photo_viewer.update()


# ==============================================================================
# 主程序窗口
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anicrop Studio - 二次元漫展返图流极客工作台 [Beta 1.0.1]")
        self.resize(1320, 880)
        self.setMinimumSize(1024, 720)
        self.setStyleSheet(DARK_STYLE_SHEET)

        self.is_cinema_mode = False
        self.prev_window_state = Qt.WindowNoState
        self.prev_geometry = None

        # 开机遮罩（由 main.py 决定是否启动）
        self.splash_overlay = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.root_layout = QVBoxLayout(central_widget)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        # 1. 顶部全局导航
        self.top_nav = QWidget()
        self.top_nav.setObjectName("TopNavWidget")
        self.top_nav.setFixedHeight(32)
        top_nav_layout = QHBoxLayout(self.top_nav)
        top_nav_layout.setContentsMargins(10, 0, 10, 0)
        top_nav_layout.setSpacing(2)

        title_lbl = QLabel("ANICROP STUDIO")
        title_lbl.setObjectName("AppTitleLabel")
        top_nav_layout.addWidget(title_lbl)

        self.btn_tab_album = QPushButton("🖼️ 漫展画廊相册")
        self.btn_tab_album.setProperty("class", "NavTab NavTabActive")
        self.btn_tab_album.clicked.connect(lambda: self._switch_tab(0))
        top_nav_layout.addWidget(self.btn_tab_album)

        self.btn_tab_darkroom = QPushButton("🧪 极客暗房工坊")
        self.btn_tab_darkroom.setProperty("class", "NavTab")
        self.btn_tab_darkroom.setToolTip("全格式统一选片暗房")
        self.btn_tab_darkroom.clicked.connect(lambda: self._switch_tab(1))
        top_nav_layout.addWidget(self.btn_tab_darkroom)

        self.btn_tab_watermark = QPushButton("🏷️ 参数画框水印")
        self.btn_tab_watermark.setProperty("class", "NavTab")
        self.btn_tab_watermark.clicked.connect(lambda: self._switch_tab(2))
        top_nav_layout.addWidget(self.btn_tab_watermark)

        self.btn_tab_settings = QPushButton("⚙️ 系统偏好设置")
        self.btn_tab_settings.setProperty("class", "NavTab")
        self.btn_tab_settings.clicked.connect(lambda: self._switch_tab(3))
        top_nav_layout.addWidget(self.btn_tab_settings)

        top_nav_layout.addStretch()
        self.root_layout.addWidget(self.top_nav)

        # 2. 下部主体
        self.body_widget = QWidget()
        self.body_layout = QHBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(8, 4, 8, 8)
        self.body_layout.setSpacing(8)

        # 2.1 左侧辅助栏
        self.left_rail = QWidget()
        self.left_rail.setObjectName("LeftRailWidget")
        self.left_rail.setFixedWidth(250)
        self.left_layout = QVBoxLayout(self.left_rail)
        self.left_layout.setContentsMargins(12, 12, 12, 12)
        self.left_layout.setSpacing(10)

        self.rail_title = QLabel("📌 空间看板")
        self.rail_title.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.left_layout.addWidget(self.rail_title)

        # 状态指示球体
        self.indicator = IndicatorSphere(
            size=226,
            header_left="R4ksh4_ChooseAgent.exe",
            header_right="v0.3",
        )
        self.left_layout.addWidget(self.indicator, alignment=Qt.AlignCenter)

        # 专业 EXIF 检查面板
        from PySide6.QtWidgets import QGridLayout
        self.exif_group = QFrame()
        self.exif_group.setObjectName("ExifCard")
        self.exif_group.setStyleSheet("""
            QFrame#ExifCard {
                background-color: #141418;
                border: 1px solid #27272f;
                border-radius: 8px;
                padding: 8px;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        exif_vlayout = QVBoxLayout(self.exif_group)
        exif_vlayout.setContentsMargins(6, 6, 6, 6)
        exif_vlayout.setSpacing(8)

        lbl_exif_header = QLabel("EXIF 拍摄参数")
        lbl_exif_header.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        exif_vlayout.addWidget(lbl_exif_header)

        gear_box = QVBoxLayout()
        gear_box.setSpacing(2)
        self.lbl_exif_camera = QLabel("--")
        self.lbl_exif_camera.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        gear_box.addWidget(self.lbl_exif_camera)

        self.lbl_exif_lens = QLabel("--")
        self.lbl_exif_lens.setWordWrap(True)
        self.lbl_exif_lens.setStyleSheet("color: #f4f4f5; font-size: 11px;")
        gear_box.addWidget(self.lbl_exif_lens)
        exif_vlayout.addLayout(gear_box)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("border-top: 1px solid #27272f; max-height: 1px;")
        exif_vlayout.addWidget(sep)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        def add_prop(row_idx, label_text, attr_name):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #a1a1aa; font-size: 11px;")
            val = QLabel("--")
            val.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 600; font-family: 'Segoe UI', 'Consolas', sans-serif;")
            grid.addWidget(lbl, row_idx, 0)
            grid.addWidget(val, row_idx, 1)
            setattr(self, attr_name, val)

        add_prop(0, "光圈", "val_exif_aperture")
        add_prop(1, "快门", "val_exif_shutter")
        add_prop(2, "感光度", "val_exif_iso")
        add_prop(3, "焦距", "val_exif_focal")
        add_prop(4, "曝光补偿", "val_exif_ev")
        add_prop(5, "原始像素", "val_exif_dim")
        add_prop(6, "文件大小", "val_exif_size")
        add_prop(7, "拍摄时间", "val_exif_date")

        exif_vlayout.addLayout(grid)
        self.left_layout.addWidget(self.exif_group)

        self.lbl_rail_hint = QLabel("")
        self.lbl_rail_hint.setWordWrap(True)
        self.lbl_rail_hint.setStyleSheet("color: #71717a; font-size: 11px;")
        self.left_layout.addWidget(self.lbl_rail_hint)

        self.left_layout.addStretch()
        self.body_layout.addWidget(self.left_rail)

        # 2.2 多页面堆栈容器
        self.stack = QStackedWidget()

        # Page 0: 漫展画廊相册
        self.page_album = AlbumPageWidget()
        self.page_album.sig_selection_changed.connect(self._on_photo_selected)
        self.page_album.sig_toggle_cinema_fullscreen.connect(self._toggle_cinema_fullscreen)
        self.page_album.sig_enter_darkroom.connect(self._on_enter_darkroom_requested)
        self.stack.addWidget(self.page_album)

        # Page 1: 极客暗房
        self.page_darkroom = DarkroomPageWidget()
        self.page_darkroom.sig_selection_changed.connect(self._on_photo_selected)
        self.page_darkroom.sig_toggle_cinema_fullscreen.connect(self._toggle_cinema_fullscreen)
        self.page_darkroom.sig_return_to_gallery.connect(lambda: self._switch_tab(0))
        self.stack.addWidget(self.page_darkroom)

        # Page 2: 参数画框水印工作台 (独立模块)
        self.page_watermark = WatermarkPageWidget()
        self.stack.addWidget(self.page_watermark)

        # Page 3: 系统偏好设置
        self.page_settings = SettingsPageWidget()
        self.page_settings.sig_default_path_changed.connect(self._on_default_path_updated)
        self.stack.addWidget(self.page_settings)
        self.body_layout.addWidget(self.stack, stretch=1)

        # 全宽工具面板挂载
        self.toolbar_stack = QStackedWidget()
        self.toolbar_stack.addWidget(self.page_album.header_widget)
        self.toolbar_stack.addWidget(self.page_darkroom.header_widget)
        self.root_layout.addWidget(self.toolbar_stack)
        self.root_layout.addWidget(self.body_widget, stretch=1)

        # 监听大图检视模式
        self.page_album.sig_viewer_active.connect(self._on_viewer_active_changed)
        self.page_darkroom.sig_viewer_active.connect(self._on_viewer_active_changed)

        self._switch_tab(0)
        self.page_darkroom.restore_session()

    def start_boot_splash(self):
        """由 main.py 首次启动时调用"""
        self.splash_overlay = SplashOverlay(self)
        self.splash_overlay.setGeometry(0, 0, self.width(), self.height())
        self.splash_overlay.sig_finished.connect(self._on_splash_done)
        self.splash_overlay.start()
        self.splash_overlay.raise_()
        self.splash_overlay.setFocus()

    def _on_splash_done(self):
        if self.splash_overlay:
            self.splash_overlay.deleteLater()
            self.splash_overlay = None

        from boot_flash import BootFlash
        steps = [
            (0,   self.body_widget),
            (2000,   self.toolbar_stack),   # 0.3s 开始闪
            (6000,  self.top_nav),          # 前一个闪完 0.2s 后开始
            (10000,  self.left_rail),
            (12000,  self.stack),
        ]
        self.boot_flash = BootFlash(self, steps, flash_duration=3000)
        self.boot_flash.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "splash_overlay", None):
            self.splash_overlay.setGeometry(0, 0, self.width(), self.height())

    def _switch_tab(self, index: int):
        self.stack.setCurrentIndex(index)
        if index < 2:
            self.toolbar_stack.setVisible(True)
            self.toolbar_stack.setCurrentIndex(index)
        else:
            self.toolbar_stack.setVisible(False)

        tabs = [self.btn_tab_album, self.btn_tab_darkroom, self.btn_tab_watermark, self.btn_tab_settings]
        for i, b in enumerate(tabs):
            b.setProperty("class", "NavTab NavTabActive" if i == index else "NavTab")
            b.style().polish(b)

        self.exif_group.setVisible(index in (0, 1))

        title_map = {0: "🖼️ 漫展画廊初筛", 1: "🧪 极客暗房工坊", 2: "🏷️ 参数画框工作台", 3: "⚙️ 系统全局偏好"}
        self.rail_title.setText(title_map.get(index, "📌 空间看板"))
        
    def _on_photo_selected(self, file_path: str, idx: int, total: int):
        exif = parse_photo_exif(file_path)
        self.lbl_exif_camera.setText(exif['camera'])
        self.lbl_exif_lens.setText(exif['lens'])
        self.val_exif_aperture.setText(exif['aperture'])
        self.val_exif_shutter.setText(exif['shutter'])
        self.val_exif_iso.setText(exif['iso'])
        self.val_exif_focal.setText(exif['focal'])
        self.val_exif_ev.setText(exif['ev'])
        self.val_exif_dim.setText(exif['dimensions'])
        self.val_exif_size.setText(exif['size'])
        self.val_exif_date.setText(exif['date'])

    def _on_default_path_updated(self, path: str):
        if path and Path(path).exists() and Path(path).is_dir():
            self.page_album.load_folder(path, record_history=True)

    def _on_enter_darkroom_requested(self, selected_photos: list, thumb_cache: dict, source_dir: str):
        self.page_darkroom.load_from_gallery(selected_photos, thumb_cache, source_dir)
        self._switch_tab(1)

    def _toggle_cinema_fullscreen(self):
        if not self.is_cinema_mode:
            self.is_cinema_mode = True
            self.prev_window_state = self.windowState()
            self.prev_geometry = self.geometry()

            self.top_nav.setVisible(False)
            self.left_rail.setVisible(False)
            self.toolbar_stack.setVisible(False)
            self.body_layout.setContentsMargins(0, 0, 0, 0)

            self.showFullScreen()
            if self.stack.currentIndex() == 0:
                self.page_album.photo_viewer.set_pure_fullscreen_mode(True)
            elif self.stack.currentIndex() == 1:
                self.page_darkroom.photo_viewer.set_pure_fullscreen_mode(True)
        else:
            self.is_cinema_mode = False
            self.top_nav.setVisible(True)
            self.left_rail.setVisible(True)
            self.toolbar_stack.setVisible(True)
            self.body_layout.setContentsMargins(8, 4, 8, 8)

            if bool(self.prev_window_state & Qt.WindowMaximized):
                self.showMaximized()
            else:
                self.showNormal()
                if hasattr(self, 'prev_geometry') and self.prev_geometry:
                    self.setGeometry(self.prev_geometry)

            if self.stack.currentIndex() == 0:
                self.page_album.photo_viewer.set_pure_fullscreen_mode(False)
            elif self.stack.currentIndex() == 1:
                self.page_darkroom.photo_viewer.set_pure_fullscreen_mode(False)

    def _on_viewer_active_changed(self, active: bool):
        if self.is_cinema_mode:
            self.toolbar_stack.setVisible(False)
        else:
            if self.stack.currentIndex() < 2:
                self.toolbar_stack.setVisible(True)