"""
画像類似度計算モジュール

MSE、SSIMなどの類似度指標を計算する機能を提供します。
"""
import numpy as np


def calculate_mse(imageA, imageB):
    """
    2枚の画像間のMSE(Mean Squared Error)を計算

    Args:
        imageA, imageB: 比較する2枚の画像

    Returns:
        MSE値(0に近いほど類似)
    """
    err = np.sum((imageA.astype(np.float64) - imageB.astype(np.float64)) ** 2)
    num_pixels = imageA.shape[0] * imageA.shape[1]
    if num_pixels == 0:
        return 0.0
    err /= float(num_pixels)
    return err
