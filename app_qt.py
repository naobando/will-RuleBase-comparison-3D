"""PySide6版エントリポイント: uv run python app_qt.py"""
import sys
import os

# プロジェクトルートをパスに追加（uv run 時の相対インポート対応）
sys.path.insert(0, os.path.dirname(__file__))


from PySide6.QtWidgets import QApplication
from config import load_config
from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("板金画像比較分析ツール")
    app.setOrganizationName("SymmetryUI")

    config = load_config()
    window = MainWindow(config)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
