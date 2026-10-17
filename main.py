# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from ui import MainWindow


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

    app = QApplication(sys.argv)
    app.setApplicationName("AnicropStudio")
    app.setOrganizationName("AnicropGeeks")

    # 2. 动态解析图标绝对路径 (开发环境与打包环境 100% 互通)
    from PySide6.QtGui import QIcon
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys._MEIPASS if hasattr(sys, '_MEIPASS') else sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parent

    icon_path = base_dir / "assets" / "anicrop.ico"
    if not icon_path.exists():
        icon_path = base_dir / "anicrop.ico"  # 兜底备用：根目录

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

    window = MainWindow()
    if icon_path.exists():
        window.setWindowIcon(app_icon)
    window.show()
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