"""アスペクト比を保持する画像表示ラベル"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from ui.utils import cv2_to_qpixmap, fit_pixmap


class ImageLabel(QLabel):
    """cv2画像またはQPixmapを表示。リサイズ時にアスペクト比を保持。"""

    def __init__(self, placeholder_text: str = "画像なし", parent=None):
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._placeholder = placeholder_text
        self.setText(placeholder_text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(100, 80)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet("background: #1a1a1a; color: #666; border: 1px solid #333;")

    def set_cv2(self, img: np.ndarray | None) -> None:
        """OpenCV BGR画像をセット"""
        if img is None:
            self.clear_image()
            return
        self._source_pixmap = cv2_to_qpixmap(img)
        self._update_display()

    def set_pixmap_source(self, pixmap: QPixmap | None) -> None:
        """QPixmapをソースとしてセット"""
        if pixmap is None or pixmap.isNull():
            self.clear_image()
            return
        self._source_pixmap = pixmap
        self._update_display()

    def clear_image(self) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self._placeholder)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = fit_pixmap(self._source_pixmap, self.width(), self.height())
        self.setText("")
        super().setPixmap(scaled)
