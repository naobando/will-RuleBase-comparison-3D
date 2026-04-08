"""折り畳み可能なグループボックス (st.expander の代替)"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)


class CollapsibleSection(QWidget):
    """クリックでコンテンツを折り畳むセクション"""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self._collapsed = collapsed

        # ヘッダー行
        header = QWidget()
        header.setStyleSheet(
            "background: #2d2d2d; border-radius: 4px; padding: 2px;"
        )
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(6, 4, 6, 4)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setStyleSheet("border: none; background: transparent;")
        self._toggle_btn.setArrowType(
            Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.DownArrow
        )

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: bold; color: #ccc;")

        header_layout.addWidget(self._toggle_btn)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        # コンテンツ
        self._content = QFrame()
        self._content.setFrameShape(QFrame.Shape.NoFrame)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 4, 4)
        self._content_layout.setSpacing(4)
        self._content.setVisible(not collapsed)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        main_layout.addWidget(header)
        main_layout.addWidget(self._content)

        self._toggle_btn.clicked.connect(self.toggle)
        header.mousePressEvent = lambda _: self.toggle()

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        self._toggle_btn.setArrowType(
            Qt.ArrowType.RightArrow if self._collapsed else Qt.ArrowType.DownArrow
        )

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsed != collapsed:
            self.toggle()
