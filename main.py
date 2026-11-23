# -*- coding: utf-8 -*-
import sys
import os

# 物理打通 Windows DLL 寻址链：让 ONNX 与 cuDNN 内部加载器同时吃进 CUDA/cuDNN 运行时
for p in sys.path:
    nv_dir = os.path.join(p, "nvidia")
    if os.path.isdir(nv_dir):
        for sub in os.listdir(nv_dir):
            bin_dir = os.path.join(nv_dir, sub, "bin")
            if os.path.isdir(bin_dir):
                # 核心杀招：cuDNN 9 内部子引擎动态加载硬性依赖系统 PATH
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings

# 唯一全局版本号入口：以后发版只改这一行
APP_VERSION = "Beta 5.0.0"


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    import traceback
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"CRITICAL UNCAUGHT EXCEPTION:\n{error_msg}", file=sys.stderr)


def main():
    sys.excepthook = handle_exception

    # 1. 物理锁死 Windows 独立任务栏图标 (消灭 Python 蟒蛇图标，未打包也独立显示)
    import os
    from pathlib import Path
    if os.name == 'nt':
        import ctypes
        myappid = 'anicropgeeks.anicropstudio.desktop.beta1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--ignore-certificate-errors --disable-gpu-shader-disk-cache"
    app = QApplication(sys.argv)
    app.setApplicationName("AnicropStudio")
    app.setOrganizationName("AnicropGeeks")

    # 2. 动态解析图标绝对路径 (源码 / onefile / onedir 全打通)
    from PySide6.QtGui import QIcon
    search_roots = []
    if hasattr(sys, '_MEIPASS'):
        search_roots.append(Path(sys._MEIPASS))
    if getattr(sys, 'frozen', False):
        search_roots.append(Path(sys.executable).parent)
    search_roots.append(Path(__file__).resolve().parent)

    icon_path = Path()
    for root in search_roots:
        for rel in (Path("assets") / "anicrop.ico", Path("anicrop.ico")):
            cand = root / rel
            if cand.exists():
                icon_path = cand
                break
        if icon_path.exists():
            break

    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    # 强制启用 Fusion 风格并全量注入深色调色板，彻底斩断 Windows 浅色主题渗透
    from PySide6.QtGui import QPalette, QColor
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor("#0e0e11"))
    dark_palette.setColor(QPalette.WindowText, QColor("#e4e4e7"))
    dark_palette.setColor(QPalette.Base, QColor("#141418"))
    dark_palette.setColor(QPalette.AlternateBase, QColor("#1e1e24"))
    dark_palette.setColor(QPalette.ToolTipBase, QColor("#18181c"))
    dark_palette.setColor(QPalette.ToolTipText, QColor("#f4f4f5"))
    dark_palette.setColor(QPalette.Text, QColor("#e4e4e7"))
    dark_palette.setColor(QPalette.Button, QColor("#1f1f24"))
    dark_palette.setColor(QPalette.ButtonText, QColor("#e4e4e7"))
    dark_palette.setColor(QPalette.BrightText, QColor("#38bdf8"))
    dark_palette.setColor(QPalette.Highlight, QColor("#0284c7"))
    dark_palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(dark_palette)

    from ui import MainWindow
    window = MainWindow()
    if icon_path.exists():
        window.setWindowIcon(app_icon)
    window.show()


    # 首次启动才播 splash
    settings = QSettings("AnicropGeeks", "AnicropStudio")
    has_seen = settings.value("has_seen_splash", False, type=bool)
    if not has_seen:
        settings.setValue("has_seen_splash", True)
        window.start_boot_splash()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()