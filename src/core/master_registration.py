"""マスター画像自動登録: 広角画像から部品を切り出して角度補正する"""
from __future__ import annotations

import cv2
import numpy as np


def extract_master(
    image: np.ndarray,
    blur_ksize: int = 7,
    morph_close_iter: int = 3,
    morph_open_iter: int = 2,
    poly_epsilon_ratio: float = 0.04,
    min_area_ratio: float = 0.01,
    padding: int = 4,
) -> tuple[np.ndarray, dict]:
    """
    広角画像から部品を自動抽出・角度補正してマスター画像を生成する。

    2段階で試行:
      1. エッジベース走査 (Canny → 4辺スキャン → 外形特定)
      2. フォールバック: 二値化ベース (大津 → blob検出)

    Parameters
    ----------
    image : np.ndarray
        入力BGR画像
    blur_ksize : int
        前処理ガウシアンブラーのカーネルサイズ
    morph_close_iter : int
        モルフォロジーCloseの反復回数
    morph_open_iter : int
        モルフォロジーOpenの反復回数
    poly_epsilon_ratio : float
        (未使用、後方互換のため残存)
    min_area_ratio : float
        画像全体に対するブロブ最小面積比率（ノイズ除去）
    padding : int
        切り出しパディング（px）

    Returns
    -------
    result_image : np.ndarray
        切り出し・角度補正後のBGR画像
    info : dict
        処理情報 (method, angle, score, mask など)
        info["mask"] に前景マスク (uint8, 0=背景/255=部品) を格納
    """
    # まずエッジベースを試行
    result, info = _extract_edge_based(image, blur_ksize, padding, min_area_ratio)
    if result is None:
        # フォールバック: 二値化ベース
        result, info = _extract_binary_based(
            image, blur_ksize, morph_close_iter, morph_open_iter,
            min_area_ratio, padding,
        )

    # 切り出し後の画像から前景マスクを生成
    info["mask"] = _create_foreground_mask(result)
    return result, info


# ══════════════════════════════════════════════════════════════════════════════
# 方式1: エッジベース走査（Canny → 4辺スキャン → percentile 境界）
# ══════════════════════════════════════════════════════════════════════════════

def _extract_edge_based(
    image: np.ndarray,
    blur_ksize: int = 7,
    padding: int = 4,
    min_area_ratio: float = 0.01,
) -> tuple[np.ndarray | None, dict]:
    """
    Cannyエッジマップ上を4辺からスキャンして部品の外形を特定する。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Canny (自動閾値: メディアンベース)
    med = np.median(blurred)
    lo = int(max(0, 0.5 * med))
    hi = int(min(255, 1.5 * med))
    edges = cv2.Canny(blurred, lo, hi)

    # エッジを太くして隙間を埋める
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, dilate_k, iterations=2)

    # ── 4辺からスキャン: 各行/列で最初のエッジ位置 ─────────────────────────
    edge_bool = edges > 0

    margin = max(5, int(min(h, w) * 0.02))

    row_has_edge = edge_bool.any(axis=1)
    left_profile = np.where(row_has_edge, np.argmax(edge_bool, axis=1), -1)
    right_profile = np.where(
        row_has_edge,
        w - 1 - np.argmax(edge_bool[:, ::-1], axis=1),
        -1,
    )

    col_has_edge = edge_bool.any(axis=0)
    top_profile = np.where(col_has_edge, np.argmax(edge_bool, axis=0), -1)
    bottom_profile = np.where(
        col_has_edge,
        h - 1 - np.argmax(edge_bool[::-1, :], axis=0),
        -1,
    )

    # 有効な行/列のみ
    valid_rows = np.where(left_profile >= margin)[0]
    valid_cols = np.where(top_profile >= margin)[0]

    if len(valid_rows) < 10 or len(valid_cols) < 10:
        return None, {}

    # percentile で外形境界を決定（外れ値除去）
    left_vals = left_profile[valid_rows]
    right_vals = right_profile[valid_rows]
    top_vals = top_profile[valid_cols]
    bottom_vals = bottom_profile[valid_cols]

    x1 = int(np.percentile(left_vals, 10))
    x2 = int(np.percentile(right_vals, 90))
    y1 = int(np.percentile(top_vals, 10))
    y2 = int(np.percentile(bottom_vals, 90))

    # パディング適用
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop_w = x2 - x1
    crop_h = y2 - y1
    img_area = h * w

    # 切り出しが小さすぎる or 大きすぎる or アスペクト比が異常な場合はフォールバック
    if crop_w * crop_h < img_area * min_area_ratio:
        return None, {}
    if crop_w * crop_h > img_area * 0.95:
        return None, {}
    aspect = max(crop_w, crop_h) / max(min(crop_w, crop_h), 1)
    if aspect > 10:
        return None, {}

    result = image[y1:y2, x1:x2]
    if result.size == 0:
        return None, {}

    # 回転補正
    result, angle = _rotate_crop(result, padding)

    return result, {
        "method": f"edge_scan_{angle:.1f}deg",
        "score": crop_w * crop_h / img_area,
        "angle": round(angle, 2),
        "bbox": (x1, y1, crop_w, crop_h),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 方式2: 二値化ベース（大津 → Convex Hull → blob検出）
# ══════════════════════════════════════════════════════════════════════════════

def _extract_binary_based(
    image: np.ndarray,
    blur_ksize: int = 7,
    morph_close_iter: int = 3,
    morph_open_iter: int = 2,
    min_area_ratio: float = 0.01,
    padding: int = 4,
) -> tuple[np.ndarray, dict]:
    """二値化 → 最大ブロブ → 回転クロップ"""
    h, w = image.shape[:2]
    img_area = h * w

    # ── 1. グレースケール + ブラー ──────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # ── 2. 大津二値化（前景/背景分離）────────────────────────────────────────
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 背景が明るい場合と暗い場合の両方に対応
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # ── 3. モルフォロジー処理（穴埋め・ノイズ除去）────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)

    # ── 3b. Convex Hull 塗りつぶし ────────────────────────────────────────────
    # 内部の穴（大きな円形穴等）でブロブが分断される問題を解消。
    # 全輪郭の convex hull を塗りつぶして再度モルフォロジーで整形する。
    all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if all_contours:
        # 面積が一定以上のブロブだけ hull 化（ノイズは除外）
        significant = [c for c in all_contours if cv2.contourArea(c) >= img_area * min_area_ratio]
        if significant:
            hull_mask = np.zeros_like(binary)
            for c in significant:
                hull = cv2.convexHull(c)
                cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
            # 近接する hull 同士を結合（部品が複数ブロブに分かれた場合）
            kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, kernel_merge, iterations=2)
            binary = hull_mask

    # ── 4. 連結成分解析 ──────────────────────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cx_img = w / 2.0
    cy_img = h / 2.0
    max_dist = np.sqrt(cx_img ** 2 + cy_img ** 2)

    best_label = -1
    best_score = -1.0

    for i in range(1, n_labels):  # 0 = 背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < img_area * min_area_ratio:
            continue

        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # 輪郭からconvex hull面積 → solidity
        blob_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # 画像モーメント重心
        M = cv2.moments(blob_mask)
        if M["m00"] > 0:
            cx_blob = M["m10"] / M["m00"]
            cy_blob = M["m01"] / M["m00"]
        else:
            cx_blob = bx + bw / 2.0
            cy_blob = by + bh / 2.0

        dist = np.sqrt((cx_blob - cx_img) ** 2 + (cy_blob - cy_img) ** 2)
        center_score = 1.0 - dist / max_dist if max_dist > 0 else 0.0

        # 面積スコア
        area_score = area / img_area

        # 総合スコア
        score = area_score * 0.6 + center_score * 0.3 + solidity * 0.1

        if score > best_score:
            best_score = score
            best_label = i

    if best_label < 0:
        return image.copy(), {"method": "fallback_none", "score": 0.0}

    # ── 5. 最良ブロブのマスク・輪郭取得 ──────────────────────────────────────
    blob_mask = (labels == best_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # ── 6. 角度補正: minAreaRect → 回転 + クロップ（歪みなし）────────────────
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect

    # OpenCV の minAreaRect: 幅 < 高さのとき angle に -90° 近い値が出る → 補正
    if rw < rh:
        angle += 90
        rw, rh = rh, rw

    # 画像中心を軸に回転
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # 回転後のブロブ中心座標を算出
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    ox, oy = w / 2.0, h / 2.0
    new_cx = cos_a * (cx - ox) + sin_a * (cy - oy) + ox
    new_cy = -sin_a * (cx - ox) + cos_a * (cy - oy) + oy

    # パディング付きクロップ
    half_w = rw / 2.0 + padding
    half_h = rh / 2.0 + padding
    x1 = max(0, int(new_cx - half_w))
    y1 = max(0, int(new_cy - half_h))
    x2 = min(w, int(new_cx + half_w))
    y2 = min(h, int(new_cy + half_h))

    result = rotated[y1:y2, x1:x2]
    method = f"rotate_crop_{angle:.1f}deg"

    if result is None or result.size == 0:
        return image.copy(), {"method": "fallback_empty", "score": best_score}

    return result, {
        "method": method,
        "score": best_score,
        "angle": round(angle, 2),
        "rect_size": (int(rw), int(rh)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 前景マスク生成（クロマキー相当: 黒背景除去）
# ══════════════════════════════════════════════════════════════════════════════

def _create_foreground_mask(image: np.ndarray) -> np.ndarray:
    """
    切り出し済み画像から黒背景を除去する前景マスクを生成する。

    背景の原色は黒。切り出し後は部品が画像の大部分を占めるため、
    大津法で前景/背景を分離し、Canny エッジで輪郭を整える。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── 1. 大津法で粗い前景マスク ─────────────────────────────────────────────
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 2. 膨張してから外輪郭を取る（部品端の削れ防止）─────────────────────────
    # 大津法が部品の暗い部分を背景と誤判定するのを救済
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask, dilate_k, iterations=2)

    # ノイズ除去 (小さいゴミを消す)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.morphologyEx(mask_dilated, cv2.MORPH_OPEN, open_k, iterations=2)

    # 最大連結成分のみ残す
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)
    if n_labels > 1:
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_dilated = np.where(labels == best_label, 255, 0).astype(np.uint8)

    # 外輪郭 → 内部塗りつぶし
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_dilated

    cnt = max(contours, key=cv2.contourArea)

    # ── 3. 輪郭スムージング（ギザギザ除去）───────────────────────────────────
    # approxPolyDP で頂点を減らしてから描画
    peri = cv2.arcLength(cnt, True)
    epsilon = 0.005 * peri
    smoothed = cv2.approxPolyDP(cnt, epsilon, True)

    mask_final = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_final, [smoothed], -1, 255, cv2.FILLED)

    return mask_final


# ══════════════════════════════════════════════════════════════════════════════
# 回転補正ヘルパー
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_crop(image: np.ndarray, padding: int = 4) -> tuple[np.ndarray, float]:
    """
    切り出し済み画像内の部品に対して minAreaRect で回転補正する。
    回転角度が微小 (< 1°) の場合はそのまま返す。

    Returns
    -------
    result : np.ndarray
        回転補正後の画像
    angle : float
        適用された回転角度 (deg)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        angle += 90

    # 微小角度は無視、大角度(±45°超)も無視（90°回転は意図しない）
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return image, 0.0

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# ══════════════════════════════════════════════════════════════════════════════
# バリデーション
# ══════════════════════════════════════════════════════════════════════════════

def validate_master_registration(
    original: np.ndarray,
    master: np.ndarray,
    threshold: float = 0.65,
) -> tuple[float, bool, str]:
    """
    マスター画像の切り出し品質を元画像とのテンプレートマッチングで検証する。

    同一画像から切り出しているため、透視変換後でも類似度が高いはず。
    スコアが threshold 未満なら切り出し失敗と判定する。

    Returns
    -------
    score : float
        TM_CCOEFF_NORMED の最大値 (0.0 〜 1.0)
    is_valid : bool
        score >= threshold
    message : str
        診断メッセージ
    """
    if master is None or master.size == 0:
        return 0.0, False, "マスターが空です"

    def _gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    orig_g = _gray(original)
    mst_g = _gray(master)

    h_o, w_o = orig_g.shape[:2]
    h_m, w_m = mst_g.shape[:2]

    # マスターが元画像より大きければ収まるようスケールダウン
    if h_m >= h_o or w_m >= w_o:
        scale = min(h_o / (h_m + 1), w_o / (w_m + 1)) * 0.95
        new_w = max(1, int(w_m * scale))
        new_h = max(1, int(h_m * scale))
        mst_g = cv2.resize(mst_g, (new_w, new_h))

    result = cv2.matchTemplate(orig_g, mst_g, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(result)
    score = float(score)

    is_valid = score >= threshold
    if score >= 0.85:
        msg = f"良好 ({score:.2f})"
    elif score >= threshold:
        msg = f"許容範囲 ({score:.2f})"
    else:
        msg = f"切り出し品質が低い可能性 ({score:.2f})"

    return score, is_valid, msg
