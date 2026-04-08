"""
差分閾値自動計算モジュール

差分画像から最適な閾値を自動的に計算する機能を提供します。
"""
import cv2
import numpy as np


def calculate_auto_diff_thresh(
    imageA,
    imageB,
    method="hybrid",
    percentile=95,
    contrast_low=8,
    contrast_mid=16,
    contrast_high=24,
    thresh_min=5,
    thresh_max=50,
):
    """
    差分閾値を自動計算

    Args:
        imageA, imageB: 比較する2つの画像
        method: 計算方法 ("otsu", "percentile", "image_based", "hybrid")
        percentile: パーセンタイル法用の閾値（上位X%）
        contrast_low/mid/high: コントラスト別の閾値
        thresh_min/max: 最小/最大値

    Returns:
        auto_thresh: 自動計算された閾値
        method_info: 計算方法の詳細情報
    """
    # サイズを揃える（imageBをimageAに合わせる）
    if imageA.shape[:2] != imageB.shape[:2]:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    # 前処理
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA

    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB

    # 差分計算
    diff = cv2.absdiff(grayA, grayB)

    # 画像特性の計算
    contrast = np.std(grayA)
    brightness = np.mean(grayA)

    method_info = {}

    if method == "otsu":
        # Otsu法
        from skimage.filters import threshold_otsu
        auto_thresh = threshold_otsu(diff)
        method_info = {"method": "Otsu", "raw_value": auto_thresh}

    elif method == "percentile":
        # パーセンタイル法
        diff_values = diff[diff > 0]  # 0を除外
        if len(diff_values) > 0:
            auto_thresh = np.percentile(diff_values, percentile)
        else:
            auto_thresh = contrast_mid
        method_info = {"method": "Percentile", "percentile": percentile, "raw_value": auto_thresh}

    elif method == "image_based":
        # 画像特性ベース
        if contrast < 30:
            auto_thresh = contrast_low
            contrast_level = "低"
        elif contrast < 80:
            auto_thresh = contrast_mid
            contrast_level = "中"
        else:
            auto_thresh = contrast_high
            contrast_level = "高"
        method_info = {
            "method": "ImageBased",
            "contrast": contrast,
            "contrast_level": contrast_level,
            "raw_value": auto_thresh,
        }

    elif method == "hybrid":
        # ハイブリッド方式
        # 1. Otsu法で初期値
        from skimage.filters import threshold_otsu
        otsu_thresh = threshold_otsu(diff)

        # 2. パーセンタイル法で検証
        diff_values = diff[diff > 0]
        if len(diff_values) > 0:
            percentile_thresh = np.percentile(diff_values, percentile)
        else:
            percentile_thresh = contrast_mid

        # 3. 画像特性で補正係数
        if contrast < 30:
            factor = 0.7  # 低コントラストは閾値を下げる
            contrast_level = "低"
        elif contrast < 80:
            factor = 1.0
            contrast_level = "中"
        else:
            factor = 1.2  # 高コントラストは閾値を上げる
            contrast_level = "高"

        # 4. 統合（Otsuとパーセンタイルの平均に補正係数を掛ける）
        auto_thresh = ((otsu_thresh + percentile_thresh) / 2) * factor

        method_info = {
            "method": "Hybrid",
            "otsu": otsu_thresh,
            "percentile": percentile_thresh,
            "contrast": contrast,
            "contrast_level": contrast_level,
            "factor": factor,
            "raw_value": auto_thresh,
        }

    else:
        # デフォルト
        auto_thresh = contrast_mid
        method_info = {"method": "Default", "raw_value": auto_thresh}

    # 範囲制限
    auto_thresh = np.clip(auto_thresh, thresh_min, thresh_max)
    method_info["final_value"] = auto_thresh

    return int(auto_thresh), method_info
