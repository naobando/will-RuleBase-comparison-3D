"""
画像ユーティリティモジュール

画像の読み込み、保存などの基本的な画像操作機能を提供します。
"""
import io
import os

import cv2
import numpy as np
from PIL import Image


def safe_imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """日本語パスでも読み込めるcv2.imread互換関数（Windows対応）"""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def safe_imwrite(path: str, img: np.ndarray, params=None) -> bool:
    """日本語パスでも保存できるcv2.imwrite互換関数（Windows対応）"""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
    result, buf = cv2.imencode(ext, img, params or [])
    if not result:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf.tofile(path)
    return True


def load_image_from_bytes(file_bytes):
    """
    バイトデータから画像を読み込む（HEIC対応）

    Args:
        file_bytes: 画像ファイルのバイトデータ

    Returns:
        OpenCV形式の画像（BGR）
    """
    pil_img = Image.open(io.BytesIO(file_bytes))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
