"""
輪郭差分検出モジュール

位置合わせ済みのマスター画像とテスト画像から、
輪郭（エッジ）の差分を検出する機能を提供する。

パイプライン:
  1. グレースケール + Gaussian Blur
  2. Cannyエッジ検出（マスター・テスト両方）
  3. エッジ差分（XOR的な差分）
  4. 前景マスクのエッジ近傍に限定
  5. モルフォロジー（close）で断片を結合
  6. 連結成分分析 + フィルタリング
  7. BBOX生成

傷検出がBlack-hat強調で細長い暗線を捉えるのと同様に、
輪郭検出はCannyエッジ差分で形状のズレ・曲がりを捉える。
"""

import cv2
from src.utils.image_utils import safe_imwrite
import numpy as np
import os


def detect_contour_diff(
    imageA,
    imageB,
    blur_ksize=5,
    canny_low=50,
    canny_high=150,
    diff_thresh=0,
    min_area=500,
    morph_kernel_size=5,
    morph_close_iter=3,
    max_boxes=5,
    fg_mask=None,
    fg_mask_dilate_iter=20,
    edge_band_width=30,
    min_length=50,
    debug_dir=None,
):
    """
    位置合わせ済み画像から輪郭差分を検出する。

    Args:
        imageA: マスター画像（BGR or グレー）
        imageB: テスト画像（BGR or グレー）
        blur_ksize: Gaussian Blur カーネルサイズ（奇数）
        canny_low: Cannyエッジ検出の低閾値
        canny_high: Cannyエッジ検出の高閾値
        diff_thresh: エッジ差分の二値化閾値（0=自動）
        min_area: 最小面積（これ未満の領域は除去）
        morph_kernel_size: モルフォロジーカーネルサイズ（奇数）
        morph_close_iter: Close処理の回数（断片結合）
        max_boxes: 最大検出数
        fg_mask: 前景マスク（255=前景）。指定時はエッジ近傍のみで検出
        fg_mask_dilate_iter: fg_maskのエッジ帯幅（ピクセル）
        edge_band_width: エッジ近傍帯の幅
        min_length: 最小長辺長（短い断片を除外）
        debug_dir: デバッグ画像保存先（Noneなら保存しない）

    Returns:
        dict: {
            "contours": [(x, y, w, h), ...],  # 輪郭差分のBBOXリスト
            "mask": np.ndarray,                 # 輪郭差分マスク（255=差分）
            "metrics": {
                "contour_count": int,
                "total_area_px": int,
                "max_length_px": int,
            },
        }
    """

    def _save_debug(name, img):
        if debug_dir is None:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            safe_imwrite(os.path.join(debug_dir, f"{name}.png"), img)
        except Exception:
            pass

    # --- 1. グレースケール + ブラー ---
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA.copy()
    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB.copy()

    ksize = int(blur_ksize) | 1  # 奇数化
    grayA = cv2.GaussianBlur(grayA, (ksize, ksize), 0)
    grayB = cv2.GaussianBlur(grayB, (ksize, ksize), 0)

    _save_debug("contour_01_grayA", grayA)
    _save_debug("contour_01_grayB", grayB)

    # --- 2. Cannyエッジ検出 ---
    edgeA = cv2.Canny(grayA, int(canny_low), int(canny_high))
    edgeB = cv2.Canny(grayB, int(canny_low), int(canny_high))

    _save_debug("contour_02_edgeA", edgeA)
    _save_debug("contour_02_edgeB", edgeB)

    # --- 3. エッジ差分 ---
    # Aにあって Bにない + Bにあって Aにない = XOR的差分
    edge_diff = cv2.absdiff(edgeA, edgeB)

    _save_debug("contour_03_edge_diff", edge_diff)

    # --- 4. 前景マスクのエッジ近傍に限定 ---
    # 輪郭差分は部品の外縁付近でのみ意味がある
    if fg_mask is not None:
        # fg_maskのエッジ帯を生成
        fg_binary = (fg_mask > 127).astype(np.uint8) * 255
        # fg_maskの輪郭を検出
        fg_eroded = cv2.erode(fg_binary, np.ones((3, 3), np.uint8), iterations=int(edge_band_width))
        fg_dilated = cv2.dilate(fg_binary, np.ones((3, 3), np.uint8), iterations=int(fg_mask_dilate_iter))
        # エッジ帯 = 膨張 - 収縮 (外縁近傍のリング状領域)
        edge_band = cv2.subtract(fg_dilated, fg_eroded)
        edge_diff = cv2.bitwise_and(edge_diff, edge_band)

        _save_debug("contour_04_edge_band", edge_band)
        _save_debug("contour_04_masked_diff", edge_diff)

    # --- 5. モルフォロジーで断片結合 ---
    mk = int(morph_kernel_size) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    closed = cv2.morphologyEx(edge_diff, cv2.MORPH_CLOSE, kernel, iterations=int(morph_close_iter))

    _save_debug("contour_05_closed", closed)

    # --- 6. 連結成分分析 + フィルタリング ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    contours = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        max_side = max(w, h)

        if area < int(min_area):
            continue
        if max_side < int(min_length):
            continue

        contours.append((x, y, w, h, area))

    # 面積順でソート、上位max_boxes
    contours.sort(key=lambda c: c[4], reverse=True)
    contours = contours[:int(max_boxes)]

    # BBOXリスト (area除去)
    bbox_list = [(x, y, w, h) for (x, y, w, h, _) in contours]

    # マスク生成
    result_mask = np.zeros_like(closed)
    for (x, y, w, h) in bbox_list:
        result_mask[y:y+h, x:x+w] = closed[y:y+h, x:x+w]

    _save_debug("contour_06_result_mask", result_mask)

    # --- 7. 指標算出 ---
    total_area = sum(a for (_, _, _, _, a) in contours)
    max_length = max((max(w, h) for (_, _, w, h, _) in contours), default=0)

    metrics = {
        "contour_count": len(bbox_list),
        "total_area_px": total_area,
        "max_length_px": max_length,
    }

    return {
        "contours": bbox_list,
        "mask": result_mask,
        "metrics": metrics,
    }
