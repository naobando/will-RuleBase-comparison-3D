"""共通ユーティリティ: OpenCV ↔ Qt 変換など"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(img: np.ndarray) -> QPixmap:
    """OpenCV BGR/Grayscale 画像を QPixmap に変換"""
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        qi = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    # QImage は data のライフタイムに依存するため copy() でオーナーシップを持つ
    return QPixmap.fromImage(qi.copy())


def fit_pixmap(pixmap: QPixmap, w: int, h: int) -> QPixmap:
    """アスペクト比を保ちながら (w, h) 内に収まるようスケール"""
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


def numpy_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """OpenCV 画像を PNG/JPEG バイト列に変換"""
    ok, buf = cv2.imencode(fmt, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: fmt={fmt}")
    return bytes(buf)
