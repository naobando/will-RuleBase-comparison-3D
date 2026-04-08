"""右パネル: 基準画像・比較画像表示"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ui.widgets.image_label import ImageLabel


class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 基準画像
        lbl_master = QLabel("基準画像")
        lbl_master.setStyleSheet("font-weight: bold; color: #aaa;")
        lbl_master.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._master_img = ImageLabel("基準画像なし")
        self._master_img.setMinimumHeight(200)

        # 比較画像
        lbl_test = QLabel("比較画像")
        lbl_test.setStyleSheet("font-weight: bold; color: #aaa;")
        lbl_test.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._test_img = ImageLabel("比較画像なし")
        self._test_img.setMinimumHeight(200)

        layout.addWidget(lbl_master)
        layout.addWidget(self._master_img, stretch=1)
        layout.addWidget(lbl_test)
        layout.addWidget(self._test_img, stretch=1)

    def set_master(self, img: np.ndarray | None) -> None:
        self._master_img.set_cv2(img)

    def set_test(self, img: np.ndarray | None) -> None:
        self._test_img.set_cv2(img)
