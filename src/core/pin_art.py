"""3Dピンアート型取り — 4方向走査による余白0切り出し"""
from __future__ import annotations

import cv2
import numpy as np


def pin_art_scan(mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    4方向からピンアート走査を行い、各方向の接触点座標を返す。

    Parameters
    ----------
    mask : np.ndarray
        uint8 二値マスク（物体=255, 背景=0）。

    Returns
    -------
    dict
        各方向の接触点。キー: "top", "bottom", "left", "right"
        値: (x座標配列, y座標配列) のタプル（物体が存在する行/列のみ）
    """
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    # 上から: 各列で最初に1が現れる行
    top_y = np.argmax(binary, axis=0)
    top_valid = binary[top_y, np.arange(w)] > 0

    # 下から: 反転して argmax
    bottom_y = h - 1 - np.argmax(binary[::-1, :], axis=0)
    bottom_valid = binary[bottom_y, np.arange(w)] > 0

    # 左から: 各行で最初に1が現れる列
    left_x = np.argmax(binary, axis=1)
    left_valid = binary[np.arange(h), left_x] > 0

    # 右から: 反転して argmax
    right_x = w - 1 - np.argmax(binary[:, ::-1], axis=1)
    right_valid = binary[np.arange(h), right_x] > 0

    return {
        "top": (np.arange(w)[top_valid], top_y[top_valid]),
        "bottom": (np.arange(w)[bottom_valid], bottom_y[bottom_valid]),
        "left": (left_x[left_valid], np.arange(h)[left_valid]),
        "right": (right_x[right_valid], np.arange(h)[right_valid]),
    }


def pin_art_crop(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    ピンアート走査に基づき、物体に密着したタイトな切り出しを行う。

    Parameters
    ----------
    image : np.ndarray
        入力BGR画像。
    mask : np.ndarray
        uint8 二値マスク（物体=255, 背景=0）。
    padding : int
        切り出し領域に追加する余白ピクセル数。

    Returns
    -------
    cropped_image : np.ndarray
        切り出し後のBGR画像（背景は黒）。
    cropped_mask : np.ndarray
        切り出し後の二値マスク。
    info : dict
        処理情報（method, bbox, area_ratio 等）。
    """
    h, w = image.shape[:2]

    contact = pin_art_scan(mask)

    # 各方向の接触点からタイトBBOXを算出
    all_x = np.concatenate([contact["top"][0], contact["bottom"][0],
                            contact["left"][0], contact["right"][0]])
    all_y = np.concatenate([contact["top"][1], contact["bottom"][1],
                            contact["left"][1], contact["right"][1]])

    if len(all_x) == 0 or len(all_y) == 0:
        # 物体が見つからない場合はそのまま返す
        return image.copy(), mask.copy(), {"method": "pin_art_fallback", "bbox": (0, 0, w, h)}

    x_min = max(0, int(np.min(all_x)) - padding)
    y_min = max(0, int(np.min(all_y)) - padding)
    x_max = min(w, int(np.max(all_x)) + 1 + padding)
    y_max = min(h, int(np.max(all_y)) + 1 + padding)

    # 切り出し
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max].copy()

    # マスク外を黒に
    if cropped_image.ndim == 3:
        cropped_image[cropped_mask == 0] = 0
    else:
        cropped_image[cropped_mask == 0] = 0

    obj_area = int(np.sum(cropped_mask > 0))
    bbox_area = (x_max - x_min) * (y_max - y_min)

    return cropped_image, cropped_mask, {
        "method": "pin_art",
        "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
        "object_area": obj_area,
        "area_ratio": obj_area / max(bbox_area, 1),
    }
