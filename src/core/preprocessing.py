"""
画像前処理モジュール

板金検査向けの画像前処理機能を提供します。
"""
import cv2
import numpy as np


def preprocess_image(
    image,
    target_size=None,
    blur_ksize=3,
    mode="luminance",
    edge_method="canny",
    edge_low=50,
    edge_high=150,
    blackhat_kernel=15,
    contrast_gamma=1.2,
):
    """
    画像の前処理を行う関数（板金向け）。選択した画像処理項目で差分検知に使う。

    Args:
        image: 入力画像(カラーまたはグレースケール)
        target_size: リサイズ後のサイズ (width, height) のタプル。Noneの場合はリサイズしない
        blur_ksize: ガウシアンブラーのカーネルサイズ(奇数)。luminance/normalize/edge等で使用
        mode: 前処理モード
            - gray_blur: グレースケール＋ブラーのみ（SSIM/新スクリプトに近い）
            - luminance: 輝度＋ブラー（現状のまま）
            - normalize: 明るさ正規化（CLAHE＋平均・標準偏差合わせ）
            - edge: エッジ（Canny/Sobel/Laplacian）
            - blackhat: BlackHat（暗い小領域抽出、穴・凹み向き）
            - contrast: コントラスト強調（ガンマ補正）
        edge_method: edgeモード時の方法 "canny" | "sobel" | "laplacian"
        edge_low, edge_high: Cannyの閾値
        blackhat_kernel: BlackHatのカーネルサイズ（奇数推奨）
        contrast_gamma: ガンマ補正の値（1.0=そのまま）

    Returns:
        前処理済みのグレースケール画像（uint8）
    """
    if image is None or image.size == 0:
        raise ValueError("入力画像が None または空です")

    # グレースケールに変換(既にグレースケールの場合はそのまま)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # リサイズ処理
    if target_size is not None:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # ノイズ低減用ブラー（多くのモードで共通）
    k = max(3, blur_ksize if blur_ksize % 2 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if mode == "gray_blur" or mode == "luminance":
        return blurred

    if mode == "normalize":
        # CLAHE（局所コントラスト）＋ 全体の平均・標準偏差を合わせる（照明差吸収）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(blurred)
        return normalized

    if mode == "edge":
        if edge_method == "canny":
            low = max(0, min(255, edge_low))
            high = max(low, min(255, edge_high))
            edges = cv2.Canny(blurred, low, high)
            return edges
        if edge_method == "sobel":
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            mag = np.clip(mag, 0, 255).astype(np.uint8)
            return mag
        if edge_method == "laplacian":
            lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
            return lap
        # fallback: canny
        return cv2.Canny(blurred, edge_low, edge_high)

    if mode == "blackhat":
        ksize = max(3, blackhat_kernel if blackhat_kernel % 2 else blackhat_kernel + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        # 0〜255に正規化（元は暗い部分が正の値）
        if blackhat.max() > blackhat.min():
            out = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            out = blackhat
        return out

    if mode == "contrast":
        inv_gamma = 1.0 / max(0.1, float(contrast_gamma))
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(blurred, table)

    return blurred
