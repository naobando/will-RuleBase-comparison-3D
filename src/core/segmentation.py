"""
前景セグメンテーションモジュール

基準画像から前景マスクを取得する機能を提供します。
"""
import cv2
import numpy as np


def get_foreground_mask(image, blur_ksize=5, keep_largest_ratio=0.0, preclose_kernel=15):
    """
    基準画像から前景マスクを画像処理で取得（Otsu + 中心を含む連結成分）。
    背景が単色に近いとき、サンプル（前景）領域だけをマスクする。

    Args:
        image: 基準画像（BGR or グレー）
        blur_ksize: ブラーカーネル（奇数）
        keep_largest_ratio: 最大連結成分の面積に対する割合以上の成分を追加で残す（0=中心成分のみ）
        preclose_kernel: 連結成分抽出前に行うCloseのカーネルサイズ（奇数）
    Returns:
        mask: 0/255 の二値マスク（255=前景）。失敗時は None（その場合はマスクなしで続行）
    """
    if image is None or image.size == 0:
        return None
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if preclose_kernel and int(preclose_kernel) > 0:
        ksize = max(3, int(preclose_kernel) if int(preclose_kernel) % 2 else int(preclose_kernel) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 2:
        return None
    cx, cy = w // 2, h // 2
    label_center = int(labels[cy, cx])
    areas = [stats[i, 4] for i in range(1, num_labels)]  # 4 = CC_STAT_AREA
    if label_center >= 1:
        chosen = label_center
    else:
        chosen = 1 + int(np.argmax(areas))
    fg = (labels == chosen).astype(np.uint8) * 255
    if keep_largest_ratio and keep_largest_ratio > 0 and areas:
        largest_area = max(areas)
        threshold = largest_area * float(keep_largest_ratio)
        for i in range(1, num_labels):
            if stats[i, 4] >= threshold:
                fg |= (labels == i).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    return fg
