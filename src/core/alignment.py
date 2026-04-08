"""
画像位置合わせモジュール

特徴点マッチングやエッジ検出を使用した画像の自動位置合わせ機能を提供します。
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_transform_magnitude(H):
    """
    ホモグラフィ行列から回転角度と平行移動量を推定

    Args:
        H: 3x3 ホモグラフィ行列

    Returns:
        rotation_deg: 回転角度（度）
        translation_px: 平行移動量（ピクセル）
    """
    if H is None:
        return 0.0, 0.0

    # 回転成分を抽出（2x2の左上部分行列）
    # atan2(H[1,0], H[0,0]) で回転角度を計算
    rotation_rad = np.arctan2(H[1, 0], H[0, 0])
    rotation_deg = np.abs(np.degrees(rotation_rad))

    # 平行移動成分を抽出
    tx = H[0, 2]
    ty = H[1, 2]
    translation_px = np.sqrt(tx**2 + ty**2)

    return rotation_deg, translation_px


def _rotate_image_by_angle(image, angle_deg):
    """画像を任意角度（度）で回転し、はみ出しを含む全体が入るサイズで返す。"""
    if angle_deg == 0 or abs(angle_deg % 360) < 1e-6:
        return image.copy()
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_rot = cv2.transform(corners.reshape(1, 4, 2), M).reshape(4, 2)
    min_x, min_y = corners_rot.min(axis=0)
    max_x, max_y = corners_rot.max(axis=0)
    nw, nh = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
    M[0, 2] -= min_x
    M[1, 2] -= min_y
    rotated = cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated


def auto_align_images(
    imageA,
    imageB,
    method="orb",
    max_features=500,
    match_ratio=0.75,
    min_matches=10,
    min_rotation=2.0,
    min_translation=20,
):
    """
    特徴点マッチングによる画像の自動位置合わせ

    Args:
        imageA: 基準画像（カラーまたはグレー）
        imageB: 補正対象画像（カラーまたはグレー）
        method: 特徴点検出アルゴリズム ("orb", "akaze", "sift")
        max_features: 検出する特徴点の最大数
        match_ratio: マッチング品質閾値（Lowe's ratio test）
        min_matches: 必要な最小マッチ数
        min_rotation: 補正を適用する最小回転角度（度）
        min_translation: 補正を適用する最小平行移動量（ピクセル）

    Returns:
        aligned_imageB: 位置合わせされた画像B
        success: 補正成功かどうか
        num_matches: マッチした特徴点数
        applied: 補正が実際に適用されたかどうか
        rotation_deg: 推定回転角度（度）
        translation_px: 推定平行移動量（ピクセル）
        inlier_ratio: インライア率（RANSAC後の有効マッチ率）
        failure_reason: 失敗理由（成功時はNone）
    """
    # グレースケール変換
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA

    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB

    # 特徴点検出器の選択
    if method.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.lower() == "akaze":
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.lower() == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        logger.warning(f"未対応のメソッド: {method}、ORBを使用します")
        detector = cv2.ORB_create(nfeatures=max_features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 特徴点検出と記述子計算
    try:
        kp1, des1 = detector.detectAndCompute(grayA, None)
        kp2, des2 = detector.detectAndCompute(grayB, None)
    except Exception as e:
        logger.error(f"特徴点検出エラー: {e}")
        return imageB, False, 0, False, 0.0, 0.0, 0.0, f"特徴点検出エラー: {e}"

    # 記述子が見つからない場合
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        kp1_count = len(kp1) if des1 is not None else 0
        kp2_count = len(kp2) if des2 is not None else 0
        reason = f"特徴点不足 (基準:{kp1_count}個, 比較:{kp2_count}個, 必要:{min_matches}個以上)"
        logger.warning(f"特徴点が不足しています (A:{kp1_count}, B:{kp2_count})")
        return imageB, False, 0, False, 0.0, 0.0, 0.0, reason

    # マッチング（k=2で2つの最良マッチを取得）
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except Exception as e:
        logger.error(f"マッチングエラー: {e}")
        return imageB, False, 0, False, 0.0, 0.0, 0.0, f"マッチングエラー: {e}"

    # Lowe's ratio testで良好なマッチのみ選択
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)

    logger.info(f"特徴点マッチング: {len(good_matches)} / {len(matches)} (method={method})")

    # マッチ数が不足している場合
    if len(good_matches) < min_matches:
        reason = f"良好なマッチ数不足 ({len(good_matches)}個 < 必要{min_matches}個) ※画像が大きく異なる可能性"
        logger.warning(f"マッチ数が不足しています: {len(good_matches)} < {min_matches}")
        return imageB, False, len(good_matches), False, 0.0, 0.0, 0.0, reason

    # 対応点の座標を抽出
    pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # アフィン変換行列の計算（RANSAC）— 回転+スケール+平行移動のみ（斜め変形なし）
    try:
        M, mask = cv2.estimateAffinePartial2D(pts_dst, pts_src, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    except Exception as e:
        reason = f"アフィン変換計算エラー: {e}"
        logger.error(f"アフィン変換計算エラー: {e}")
        return imageB, False, len(good_matches), False, 0.0, 0.0, 0.0, reason

    if M is None:
        reason = f"アフィン変換行列計算失敗 (マッチ品質不足、インライア率が低い可能性)"
        logger.warning("アフィン変換行列が計算できませんでした")
        return imageB, False, len(good_matches), False, 0.0, 0.0, 0.0, reason

    # 3x3 に拡張（estimate_transform_magnitude との互換性）
    H = np.vstack([M, [0, 0, 1]])

    # インライア数（RANSAC後の有効なマッチ数）
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0.0
    logger.info(f"インライア数: {inliers} / {len(good_matches)} (率: {inlier_ratio:.2f})")

    # 変換量を推定
    rotation_deg, translation_px = estimate_transform_magnitude(H)
    logger.info(f"推定変換量: 回転={rotation_deg:.2f}度, 平行移動={translation_px:.1f}px")

    # 閾値チェック：ずれが小さい場合は補正をスキップ
    if rotation_deg < min_rotation and translation_px < min_translation:
        logger.info(f"ずれが小さいため補正をスキップ (閾値: 回転>{min_rotation}度, 平行移動>{min_translation}px)")
        return imageB, True, len(good_matches), False, rotation_deg, translation_px, inlier_ratio, None

    # 画像Bを変換（アフィン変換 — 斜め歪みなし）
    target_h, target_w = imageA.shape[:2]
    try:
        aligned_imageB = cv2.warpAffine(imageB, M, (target_w, target_h))
    except Exception as e:
        logger.error(f"画像変換エラー: {e}")
        return imageB, False, len(good_matches), False, rotation_deg, translation_px, inlier_ratio, f"画像変換エラー: {e}"

    # 出力サイズがずれた場合の保険（1ピクセル差の不具合対策）
    if aligned_imageB.shape[:2] != (target_h, target_w):
        aligned_imageB = cv2.resize(aligned_imageB, (target_w, target_h))

    logger.info("補正を適用しました")
    return aligned_imageB, True, len(good_matches), True, rotation_deg, translation_px, inlier_ratio, None


def _get_main_rect_from_edges(gray, canny_low=50, canny_high=150, min_area_ratio=0.01):
    """エッジ画像から主輪郭の最小外接矩形を取得。失敗時は None。"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape[:2]
    min_area = int(h * w * min_area_ratio)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    main_c = max(valid, key=cv2.contourArea)
    return cv2.minAreaRect(main_c)


def _get_main_rect_from_bbox(gray, blur_ksize=5, min_area_ratio=0.01):
    """前景（Otsu＋中心成分）から主輪郭の最小外接矩形を取得。失敗時は None。"""
    h, w = gray.shape[:2]
    min_area = int(h * w * min_area_ratio)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 2:
        return None
    cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
    label_center = int(labels[cy, cx])
    if label_center >= 1:
        chosen = label_center
    else:
        areas = [stats[i, 4] for i in range(1, num_labels)]
        chosen = 1 + int(np.argmax(areas))
    fg = (labels == chosen).astype(np.uint8) * 255
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    main_c = max(valid, key=cv2.contourArea)
    return cv2.minAreaRect(main_c)


def _align_by_rect(imageA, imageB, get_rect_func, get_rect_kwargs=None):
    """
    2画像から get_rect_func で主矩形を取得し、B を A の矩形に合わせてアフィン変換。
    Returns:
        (aligned_imageB, success, num_matches, applied, rotation_deg, translation_px, inlier_ratio, failure_reason)
    """
    get_rect_kwargs = get_rect_kwargs or {}
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA
    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB
    rectA = get_rect_func(grayA, **get_rect_kwargs)
    rectB = get_rect_func(grayB, **get_rect_kwargs)
    if rectA is None:
        return imageB, False, 0, False, 0.0, 0.0, 0.0, "基準画像で主矩形を取得できませんでした"
    if rectB is None:
        return imageB, False, 0, False, 0.0, 0.0, 0.0, "比較画像で主矩形を取得できませんでした"
    (cxA, cyA), (wA, hA), angleA = rectA
    (cxB, cyB), (wB, hB), angleB = rectB
    angleA = angleA % 180.0
    angleB = angleB % 180.0
    rotation_deg = abs(angleA - angleB)
    if rotation_deg > 90:
        rotation_deg = 180.0 - rotation_deg
    translation_px = float(np.hypot(cxA - cxB, cyA - cyB))
    # スケールは1.0固定（カメラ固定のため部品サイズは変わらない。異常画像で矩形サイズが
    # 変わってもスケールを変えない）。回転と平行移動のみで位置合わせする。
    scale = 1.0
    angle_rad = np.deg2rad(angleA - angleB)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c * scale, -s * scale], [s * scale, c * scale]], dtype=np.float64)
    t = np.array([[cxA], [cyA]]) - R @ np.array([[cxB], [cyB]])
    M = np.hstack([R, t]).astype(np.float32)
    target_h, target_w = imageA.shape[:2]
    try:
        aligned = cv2.warpAffine(imageB, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
    except Exception as e:
        return imageB, False, 0, False, rotation_deg, translation_px, 0.0, f"アフィン変換エラー: {e}"
    if aligned.shape[:2] != (target_h, target_w):
        aligned = cv2.resize(aligned, (target_w, target_h))
    return aligned, True, 0, True, rotation_deg, translation_px, 0.0, None


def align_edge_based(imageA, imageB, canny_low=50, canny_high=150):
    """
    エッジ一致による位置合わせ。両画像のエッジから主輪郭の最小外接矩形を求め、B を A に合わせる。
    Returns: (aligned_imageB, success, num_matches, applied, rotation_deg, translation_px, inlier_ratio, failure_reason)
    """
    out = _align_by_rect(
        imageA, imageB,
        lambda g, **kw: _get_main_rect_from_edges(g, canny_low=canny_low, canny_high=canny_high, **kw),
        {"min_area_ratio": 0.01},
    )
    if out[1]:
        logger.info("エッジ一致で補正を適用しました")
    return out


def align_bbox_based(imageA, imageB):
    """
    外接矩形一致による位置合わせ。両画像の前景から主輪郭の最小外接矩形を求め、B を A に合わせる。
    Returns: (aligned_imageB, success, num_matches, applied, rotation_deg, translation_px, inlier_ratio, failure_reason)
    """
    out = _align_by_rect(
        imageA, imageB,
        _get_main_rect_from_bbox,
        {"min_area_ratio": 0.01},
    )
    if out[1]:
        logger.info("外接矩形一致で補正を適用しました")
    return out


def _get_fg_contour(gray, simplify_n=500):
    """前景マスク(Otsu)から主輪郭点を取得。"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels < 2:
        return None
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)]
    main_label = 1 + int(np.argmax(areas))
    fg = ((labels == main_label) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    main_c = max(contours, key=cv2.contourArea)
    pts = main_c.reshape(-1, 2).astype(np.float32)
    if simplify_n and len(pts) > simplify_n:
        idx = np.linspace(0, len(pts) - 1, simplify_n, dtype=int)
        pts = pts[idx]
    return pts


def align_fg_icp(imageA, imageB, max_iter=50, inlier_percentile=80):
    """
    fg_mask輪郭のICP（Iterative Closest Point）による位置合わせ。
    両画像の前景輪郭を抽出し、最近傍点の反復マッチングで剛体変換を推定。

    Returns: (aligned_imageB, success, num_matches, applied, rotation_deg, translation_px, inlier_ratio, failure_reason)
    """
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA
    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB

    ptsA = _get_fg_contour(grayA)
    ptsB = _get_fg_contour(grayB)
    if ptsA is None:
        return imageB, False, 0, False, 0.0, 0.0, 0.0, "基準画像で前景輪郭を取得できませんでした"
    if ptsB is None:
        return imageB, False, 0, False, 0.0, 0.0, 0.0, "比較画像で前景輪郭を取得できませんでした"

    src = ptsB.copy()
    total_M = np.eye(3, dtype=np.float64)

    for it in range(max_iter):
        # 最近傍点マッチング
        dists = np.zeros(len(src))
        matched = np.zeros_like(src)
        for i, p in enumerate(src):
            d = np.sum((ptsA - p) ** 2, axis=1)
            j = np.argmin(d)
            matched[i] = ptsA[j]
            dists[i] = d[j]

        # 外れ値除去
        thresh = np.percentile(dists, inlier_percentile)
        inlier = dists < thresh
        if np.sum(inlier) < 10:
            break

        src_in = src[inlier]
        dst_in = matched[inlier]

        # SVDで最適剛体変換
        mean_src = src_in.mean(axis=0)
        mean_dst = dst_in.mean(axis=0)
        H_mat = (src_in - mean_src).T @ (dst_in - mean_dst)
        U, S, Vt = np.linalg.svd(H_mat)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mean_dst - R @ mean_src

        # 更新
        src = (R @ src.T).T + t
        M_step = np.eye(3, dtype=np.float64)
        M_step[:2, :2] = R
        M_step[:2, 2] = t
        total_M = M_step @ total_M

        # 収束判定
        if np.max(np.abs(t)) < 0.01 and np.abs(np.arcsin(np.clip(R[1, 0], -1, 1))) < 1e-5:
            break

    M_affine = total_M[:2, :].astype(np.float32)
    h, w = imageA.shape[:2]
    aligned = cv2.warpAffine(imageB, M_affine, (w, h), flags=cv2.INTER_LINEAR)

    rotation_deg = float(np.degrees(np.arcsin(np.clip(total_M[1, 0], -1, 1))))
    translation_px = float(np.hypot(total_M[0, 2], total_M[1, 2]))
    n_inlier = int(np.sum(inlier)) if 'inlier' in dir() else 0
    inlier_ratio = n_inlier / len(ptsB) if len(ptsB) > 0 else 0.0

    logger.info(f"fg_mask ICP: iter={it+1} rot={abs(rotation_deg):.3f}° "
                f"tx={translation_px:.1f}px inlier={inlier_ratio:.2f}")
    return aligned, True, len(ptsB), True, abs(rotation_deg), translation_px, inlier_ratio, None


def align_ecc_refine(imageA, imageB, blur_ksize=5, max_iter=1000, eps=1e-6,
                     warp_mode="affine", fg_mask=None):
    """
    ECC（Enhanced Correlation Coefficient）によるサブピクセル精密位置合わせ。
    既存のalign処理の後に追加の仕上げステップとして使用。

    Args:
        imageA: 基準画像（カラーまたはグレー）
        imageB: 補正対象画像（カラーまたはグレー）
        blur_ksize: ガウシアンブラーのカーネルサイズ（安定化用）
        max_iter: ECC最大反復回数
        eps: ECC収束判定の精度
        warp_mode: "affine"（アフィン, 6パラメータ）or "euclidean"（回転+平行移動, 3パラメータ）
        fg_mask: 前景マスク（指定時は前景領域のみでECC計算）

    Returns:
        aligned_imageB: 位置合わせされた画像B
        success: 成功かどうか
        ecc_score: ECC相関係数（1.0に近いほど良い合わせ）
        translation: (tx, ty) サブピクセル平行移動量
        rotation_deg: 回転角度（度）
        warp_matrix: 推定された変換行列
    """
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA.copy()

    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB.copy()

    # サイズ合わせ
    if grayA.shape != grayB.shape:
        grayB = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]))

    # ガウシアンブラーで安定化
    if blur_ksize > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gA = cv2.GaussianBlur(grayA, (ksize, ksize), 0)
        gB = cv2.GaussianBlur(grayB, (ksize, ksize), 0)
    else:
        gA = grayA
        gB = grayB

    # warp_mode選択
    if warp_mode == "euclidean":
        mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    # マスク準備
    input_mask = None
    if fg_mask is not None:
        if fg_mask.ndim == 3:
            input_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        else:
            input_mask = fg_mask.copy()
        _, input_mask = cv2.threshold(input_mask, 0, 255, cv2.THRESH_BINARY)
        if input_mask.shape != gA.shape:
            input_mask = cv2.resize(input_mask, (gA.shape[1], gA.shape[0]))

    try:
        cc, warp_matrix = cv2.findTransformECC(
            gA, gB, warp_matrix, mode, criteria, inputMask=input_mask
        )
    except cv2.error as e:
        logger.warning(f"ECC位置合わせ失敗: {e}")
        return imageB, False, 0.0, (0.0, 0.0), 0.0, None

    # 変換量を抽出
    tx = float(warp_matrix[0, 2])
    ty = float(warp_matrix[1, 2])
    rot_deg = float(np.degrees(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])))

    logger.info(f"ECC精密合わせ: cc={cc:.6f}, tx={tx:.2f}px, ty={ty:.2f}px, rot={rot_deg:.4f}deg")

    # 変換適用
    h, w = imageA.shape[:2]
    try:
        aligned_imageB = cv2.warpAffine(
            imageB, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except Exception as e:
        logger.error(f"ECC変換適用エラー: {e}")
        return imageB, False, cc, (tx, ty), rot_deg, warp_matrix

    if aligned_imageB.shape[:2] != (h, w):
        aligned_imageB = cv2.resize(aligned_imageB, (w, h))

    return aligned_imageB, True, cc, (tx, ty), rot_deg, warp_matrix
