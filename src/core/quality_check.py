"""
画質チェックモジュール

撮影画像の白飛び・ブレを検出する機能を提供します。
"""
import cv2


def check_image_quality(
    image,
    white_ratio_threshold=0.05,
    blur_threshold=100,
    high_brightness=250,
    dark_ratio_threshold=None,
    low_brightness=5,
    low_contrast_threshold=None,
    specular_ratio_threshold=None,
    specular_high_v=240,
    specular_low_s=40,
    return_details=False,
):
    """
    撮影画像の画質を簡易チェック（白飛び・ブレなど）。
    比較画像の自動再取得の判定に使う。

    Args:
        image: 入力画像（BGR or グレー）
        white_ratio_threshold: 高輝度(>=high_brightness)ピクセルの割合がこれ以上なら白飛びと判定
        blur_threshold: ラプラシアン分散がこれ未満ならブレと判定（解像度・実機で要調整）
        high_brightness: 白飛び判定の輝度閾値
        dark_ratio_threshold: 低輝度(<=low_brightness)の割合がこれ以上なら暗すぎと判定
        low_brightness: 暗部判定の輝度閾値
        low_contrast_threshold: 画素の標準偏差がこれ未満ならコントラスト不足と判定
        specular_ratio_threshold: 高輝度かつ低彩度の割合がこれ以上なら反射強めと判定
        specular_high_v: 反射判定用のHSV V閾値
        specular_low_s: 反射判定用のHSV S閾値
        return_details: True の場合は詳細メトリクスを返す

    Returns:
        (ok, reason) または (ok, reason, metrics)
        ok が True なら合格。False なら reason に理由文字列。
    """
    if image is None or image.size == 0:
        if return_details:
            return False, "画像なし", {
                "white_ratio": None,
                "dark_ratio": None,
                "laplacian_var": None,
                "contrast_std": None,
                "mean": None,
                "specular_ratio": None,
            }
        return False, "画像なし"
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    total = gray.size
    # 白飛び: 高輝度ピクセルの割合
    white_ratio = (gray >= high_brightness).sum() / float(total)
    # 暗すぎ: 低輝度ピクセルの割合
    dark_ratio = (gray <= low_brightness).sum() / float(total)
    # ブレ: ラプラシアン分散（エッジの鋭さ）
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # コントラスト: 標準偏差
    contrast_std = gray.std()

    specular_ratio = None
    if specular_ratio_threshold is not None and image is not None and len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        specular_mask = (v >= specular_high_v) & (s <= specular_low_s)
        specular_ratio = float(specular_mask.sum()) / float(total)

    reasons = []
    if white_ratio_threshold is not None and white_ratio_threshold > 0 and white_ratio >= white_ratio_threshold:
        reasons.append("白飛び")
    if blur_threshold is not None and blur_threshold > 0 and lap_var < blur_threshold:
        reasons.append("ブレ")
    if dark_ratio_threshold is not None and dark_ratio_threshold > 0 and dark_ratio >= dark_ratio_threshold:
        reasons.append("暗すぎ")
    if low_contrast_threshold is not None and low_contrast_threshold > 0 and contrast_std < low_contrast_threshold:
        reasons.append("コントラスト不足")
    if specular_ratio_threshold is not None and specular_ratio_threshold > 0 and specular_ratio is not None and specular_ratio >= specular_ratio_threshold:
        reasons.append("反射")

    ok = len(reasons) == 0
    reason = " / ".join(reasons) if reasons else None

    if not return_details:
        return ok, reason

    metrics = {
        "white_ratio": float(white_ratio),
        "dark_ratio": float(dark_ratio),
        "laplacian_var": float(lap_var),
        "contrast_std": float(contrast_std),
        "mean": float(gray.mean()),
        "specular_ratio": float(specular_ratio) if specular_ratio is not None else None,
    }
    return ok, reason, metrics
