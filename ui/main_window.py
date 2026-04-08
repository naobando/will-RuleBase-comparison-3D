"""メインウィンドウ: モード切替ホルダー"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QMainWindow, QPushButton,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from ui.debug.debug_window import DebugWindow
from ui.user.user_window import UserWindow


class MainWindow(QMainWindow):
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self.setWindowTitle(config["ui"].get("title", "板金画像比較分析ツール"))
        self.resize(1400, 900)

        # ルートウィジェット
        root_widget = QWidget()
        root_layout = QVBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root_widget)

        # モード切替バー
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background: #111;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(4)

        self._btn_user = QPushButton("ユーザーモード")
        self._btn_debug = QPushButton("デバッグモード")
        for btn in (self._btn_user, self._btn_debug):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        bar_layout.addWidget(self._btn_user)
        bar_layout.addWidget(self._btn_debug)
        bar_layout.addStretch()

        # スタック
        self._stack = QStackedWidget()

        self._user_window = UserWindow(config)
        self._debug_window = DebugWindow(config)
        self._stack.addWidget(self._user_window)
        self._stack.addWidget(self._debug_window)

        root_layout.addWidget(bar)
        root_layout.addWidget(self._stack, stretch=1)

        # シグナル接続
        self._btn_user.clicked.connect(lambda: self._switch(0))
        self._btn_debug.clicked.connect(lambda: self._switch(1))

        # 起動時はUserモード
        self._switch(0)

        # ダークテーマ
        self.setStyleSheet(_DARK_STYLE)

    def _switch(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._btn_user.setChecked(index == 0)
        self._btn_debug.setChecked(index == 1)

    def closeEvent(self, event):
        self._user_window.close()
        self._debug_window.close()
        super().closeEvent(event)


_DARK_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #ddd;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ddd;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton:checked {
    background-color: #1565c0;
    border-color: #1976d2;
    color: #fff;
}
QComboBox, QSpinBox, QSlider {
    background-color: #2a2a2a;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 2px 4px;
    color: #ddd;
}
QCheckBox {
    color: #ccc;
}
QScrollBar:vertical {
    background: #2a2a2a;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 4px;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
}
QTabBar::tab {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    padding: 4px 10px;
    color: #aaa;
}
QTabBar::tab:selected {
    background: #3a3a3a;
    color: #fff;
}
QStatusBar {
    background: #161616;
    color: #999;
    font-size: 11px;
}
QSplitter::handle {
    background: #333;
}
QMenuBar {
    background: #161616;
    color: #ccc;
}
QMenu {
    background: #2a2a2a;
    color: #ccc;
    border: 1px solid #555;
}
QMenu::item:selected {
    background: #3a6a3a;
}
"""
