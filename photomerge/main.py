import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

# 同时支持 `python photomerge/main.py` 和 `python -m photomerge.main`。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photomerge.merge_ui import PhotoMergePageWidget


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    win = PhotoMergePageWidget()
    win.setWindowTitle("Anicrop Studio - 二次元光学溶图工作台")
    win.resize(1280, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
