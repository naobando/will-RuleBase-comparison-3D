"""
画角合わせ（FOVクロップ）モジュール

比較画像とマスターの画角を揃える機能を提供します。
"""
import math

import cv2
import numpy as np

from src.core.alignment import _rotate_image_by_angle


def crop_to_master_fov(
    imageA,
    imageB,
    method="orb",
    max_features=500,
    match_ratio=0.75,
    min_matches=10,
    try_rotations=False,
    rotation_search_step_deg=90,
    try_flip=True,
    max_rotation_deg=15.0,
    min_inlier_ratio=0.3,
):
    """
    比較画像とマスターの画角を揃える。
    特徴点マッチングでホモグラフィを求め、
    - 比較画像がマスターより**大きい or 同サイズ**: imageB を imageA の枠にワープ（従来どおり）。
    - 比較画像がマスターより**小さい**: imageA（マスター）を imageB のサイズにワープし、
      比較画像は拡大しない（「最大から小さくする」で画質悪化を防ぐ）。
    try_rotations=True のとき: 比較画像の回転角度を探索し、good_matches が最も多い向きで画角合わせする。
      rotation_search_step_deg>=90: 0/90/180/270 の4向きのみ。
      rotation_search_step_deg<90: 枝分かれ式に細かく探索。

    Returns:
        (cropped_compare, warped_master_or_None, success) または失敗時 (None, None, False, diag_dict):
        - 比較が大きい/同サイズ: (warped_imageB, None, True)
        - 比較が小さい: (imageB, warped_imageA, True)
        - 失敗: (None, None, False, diag_dict)
    """
    if imageA.shape[:2] == imageB.shape[:2]:
        return imageB.copy(), None, True, {"good_matches": 0, "flipped": False}

    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA

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
        detector = cv2.ORB_create(nfeatures=max_features)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        kp1, des1 = detector.detectAndCompute(grayA, None)
    except Exception:
        return None, None, False, {"error": "detectAndCompute"}
    n_kp1 = len(kp1) if kp1 else 0
    min_kp_to_try = 3
    if des1 is None or n_kp1 < min_kp_to_try:
        return None, None, False, {"kp1": n_kp1, "kp2": 0, "good_matches": None}

    # 回転角度を試し、good_matches が最も多い向きを選ぶ
    def _eval_angle(angle_deg, src_img=None):
        """回転角度を評価。src_img が指定されていればそれを使う（反転済み画像用）"""
        base = src_img if src_img is not None else imageB
        if abs(angle_deg % 360) < 1e-6:
            img_rot = base
        else:
            img_rot = _rotate_image_by_angle(base, angle_deg)
        if len(img_rot.shape) == 3:
            grayB = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
        else:
            grayB = img_rot
        try:
            kp2, des2 = detector.detectAndCompute(grayB, None)
        except Exception:
            return -1, None, None, None, None
        n_kp2 = len(kp2) if kp2 else 0
        if des2 is None or n_kp2 < min_kp_to_try:
            return -1, None, None, None, None
        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return -1, None, None, None, None
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < match_ratio * n.distance:
                    good_matches.append(m)
        n_good = len(good_matches)
        return n_good, img_rot, kp2, des2, good_matches

    best_n_good = -1
    best_imageB = imageB
    best_kp2 = None
    best_des2 = None
    best_good_matches = None
    best_angle_deg = 0.0
    best_flipped = False  # 左右反転フラグ

    # 左右反転した画像を事前生成（水平反転 = flipCode 1）
    imageB_flipped = cv2.flip(imageB, 1) if try_flip else None

    if try_rotations:
        step_deg = min(90.0, max(1.0, float(rotation_search_step_deg)))
        angles_to_try = [0.0, 90.0, 180.0, 270.0]
        for ang in angles_to_try:
            # 通常（反転なし）
            n_good, img_rot, kp2, des2, good_matches = _eval_angle(ang)
            if n_good >= 3 and n_good > best_n_good:
                best_n_good, best_imageB = n_good, img_rot
                best_kp2, best_des2, best_good_matches = kp2, des2, good_matches
                best_angle_deg = ang
                best_flipped = False
            # 左右反転版
            if imageB_flipped is not None:
                n_good_f, img_rot_f, kp2_f, des2_f, gm_f = _eval_angle(ang, src_img=imageB_flipped)
                if n_good_f >= 3 and n_good_f > best_n_good:
                    best_n_good, best_imageB = n_good_f, img_rot_f
                    best_kp2, best_des2, best_good_matches = kp2_f, des2_f, gm_f
                    best_angle_deg = ang
                    best_flipped = True
        branch_step = 45.0
        while branch_step >= step_deg and best_n_good >= 3:
            for delta in (-branch_step, 0, branch_step):
                ang = (best_angle_deg + delta) % 360.0
                # 通常
                n_good, img_rot, kp2, des2, good_matches = _eval_angle(ang)
                if n_good >= 3 and n_good > best_n_good:
                    best_n_good, best_imageB = n_good, img_rot
                    best_kp2, best_des2, best_good_matches = kp2, des2, good_matches
                    best_angle_deg = ang
                    best_flipped = False
                # 左右反転版
                if imageB_flipped is not None:
                    n_good_f, img_rot_f, kp2_f, des2_f, gm_f = _eval_angle(ang, src_img=imageB_flipped)
                    if n_good_f >= 3 and n_good_f > best_n_good:
                        best_n_good, best_imageB = n_good_f, img_rot_f
                        best_kp2, best_des2, best_good_matches = kp2_f, des2_f, gm_f
                        best_angle_deg = ang
                        best_flipped = True
            branch_step /= 2.0
    else:
        # 通常
        n_good, img_rot, kp2, des2, good_matches = _eval_angle(0.0)
        if n_good >= 3:
            best_n_good, best_imageB = n_good, img_rot
            best_kp2, best_des2, best_good_matches = kp2, des2, good_matches
            best_flipped = False
        # 左右反転版も試す
        if imageB_flipped is not None:
            n_good_f, img_rot_f, kp2_f, des2_f, gm_f = _eval_angle(0.0, src_img=imageB_flipped)
            if n_good_f >= 3 and n_good_f > best_n_good:
                best_n_good, best_imageB = n_good_f, img_rot_f
                best_kp2, best_des2, best_good_matches = kp2_f, des2_f, gm_f
                best_flipped = True

    if best_n_good < 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) if len(imageB.shape) == 3 else imageB
        try:
            kp2, des2 = detector.detectAndCompute(grayB, None)
        except Exception:
            return None, None, False, {"error": "detectAndCompute", "flipped": False}
        n_kp2 = len(kp2) if kp2 else 0
        if des2 is None or n_kp2 < min_kp_to_try:
            return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, "good_matches": None, "flipped": False}
        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, "good_matches": None, "flipped": False}
        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < match_ratio * n.distance:
                    good_matches.append(m)
        n_good = len(good_matches)
        return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, "good_matches": n_good, "flipped": False}

    imageB = best_imageB
    good_matches = best_good_matches
    kp2 = best_kp2
    n_kp2 = len(kp2) if kp2 else 0
    n_good = best_n_good

    pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_h, target_w = imageA.shape[:2]
    area_a = target_h * target_w
    area_b = imageB.shape[0] * imageB.shape[1]

    _diag_base = {"good_matches": n_good, "flipped": best_flipped}

    # 3点以上: 相似変換（回転+等方スケール+平行移動のみ）で画角合わせ
    # getAffineTransform は6自由度（シアー含む）のため使わない
    use_partial = n_good >= 3 and n_good < min_matches
    if use_partial:
        try:
            M, _mask3 = cv2.estimateAffinePartial2D(pts_dst, pts_src, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is None:
                return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, **_diag_base}
            if area_b < area_a:
                M_inv = cv2.invertAffineTransform(M)
                warped_a = cv2.warpAffine(imageA, M_inv, (imageB.shape[1], imageB.shape[0]))
                if warped_a.shape[:2] != imageB.shape[:2]:
                    warped_a = cv2.resize(warped_a, (imageB.shape[1], imageB.shape[0]))
                return imageB.copy(), warped_a, True, _diag_base
            cropped = cv2.warpAffine(imageB, M, (target_w, target_h))
            if cropped.shape[:2] != (target_h, target_w):
                cropped = cv2.resize(cropped, (target_w, target_h))
            return cropped, None, True, _diag_base
        except Exception:
            return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, **_diag_base}

    # 4点以上: 相似変換（回転+等方スケール+平行移動, 4自由度）で画角合わせ
    # findHomography は透視変換（8自由度）のためシアー歪みが入る。
    # estimateAffinePartial2D は相似変換のみなのでアスペクト比が保たれる。
    try:
        M, mask = cv2.estimateAffinePartial2D(pts_dst, pts_src, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    except Exception:
        return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, **_diag_base}
    if M is None:
        return None, None, False, {"kp1": n_kp1, "kp2": n_kp2, **_diag_base}

    # --- 変換品質チェック ---
    # アフィン行列から推定回転角を算出（事前回転角を差し引いて残差のみチェック）
    raw_rotation_deg = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    residual_rotation = raw_rotation_deg - best_angle_deg
    residual_rotation = (residual_rotation + 180) % 360 - 180
    estimated_rotation_deg = abs(residual_rotation)
    _diag_base["estimated_rotation_deg"] = round(estimated_rotation_deg, 2)
    _diag_base["raw_rotation_from_M_deg"] = round(raw_rotation_deg, 2)
    _diag_base["pre_rotation_deg"] = round(best_angle_deg, 2)

    if max_rotation_deg > 0 and estimated_rotation_deg > max_rotation_deg:
        return None, None, False, {
            "kp1": n_kp1, "kp2": n_kp2,
            "reject": "rotation_too_large",
            "estimated_rotation_deg": round(estimated_rotation_deg, 2),
            "max_rotation_deg": max_rotation_deg,
            **_diag_base,
        }

    # RANSACインライア率チェック
    if mask is not None and len(mask) > 0:
        inlier_ratio = float(np.sum(mask)) / len(mask)
        _diag_base["inlier_ratio"] = round(inlier_ratio, 3)
        if min_inlier_ratio > 0 and inlier_ratio < min_inlier_ratio:
            return None, None, False, {
                "kp1": n_kp1, "kp2": n_kp2,
                "reject": "inlier_ratio_too_low",
                "inlier_ratio": round(inlier_ratio, 3),
                "min_inlier_ratio": min_inlier_ratio,
                **_diag_base,
            }

    # 変換の幾何学的妥当性チェック（スケールが負 = 反転 → 不正）
    scale = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    _diag_base["warp_scale"] = round(scale, 4)
    _diag_base["warp_det"] = round(det, 4)
    if det <= 0:
        return None, None, False, {
            "kp1": n_kp1, "kp2": n_kp2,
            "reject": "warp_degenerate",
            "warp_det": round(det, 4),
            **_diag_base,
        }

    # 比較画像がマスターより小さい → マスターを比較サイズにワープ
    if area_b < area_a:
        try:
            M_inv = cv2.invertAffineTransform(M)
            warped_a = cv2.warpAffine(imageA, M_inv, (imageB.shape[1], imageB.shape[0]))
        except Exception:
            return None, None, False, _diag_base
        if warped_a.shape[:2] != imageB.shape[:2]:
            warped_a = cv2.resize(warped_a, (imageB.shape[1], imageB.shape[0]))
        return imageB.copy(), warped_a, True, _diag_base

    # 比較画像がマスター以上 → imageB を imageA の枠にワープ
    try:
        cropped = cv2.warpAffine(imageB, M, (target_w, target_h))
    except Exception:
        return None, None, False, _diag_base
    if cropped.shape[:2] != (target_h, target_w):
        cropped = cv2.resize(cropped, (target_w, target_h))
    return cropped, None, True, _diag_base


def _trim_dark_borders(image, dark_thresh=30, min_dark_ratio=0.5, rotation_deg=0.0, diag=None):
    """
    画像の四辺にある暗い帯（回転由来の黒い余白）を自動トリムする。

    各辺から内側に向かってスキャンし、暗いピクセルが過半数の行/列を除去する。
    回転角度から理論的に発生する黒い帯の最大幅を計算し、それ以上は削らない。

    Args:
        image: 入力画像（BGR）
        dark_thresh: この輝度未満を「暗い」とみなす閾値
        min_dark_ratio: 行/列中の暗いピクセル比率がこれ以上なら帯とみなす
        rotation_deg: 回転角度（度）。これに基づいてトリム量の上限を決める
        diag: 診断用辞書（情報を追記する）

    Returns:
        トリムされた画像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h, w = gray.shape[:2]

    # 回転角度から理論的に発生する黒い帯の最大幅を計算
    # 回転角θで幅wの画像を回転すると、上下に最大 w*sin(θ) の黒帯ができる
    abs_rot = abs(rotation_deg % 360)
    if abs_rot > 180:
        abs_rot = 360 - abs_rot
    sin_rot = math.sin(math.radians(abs_rot))
    # 余裕として2倍+2px確保（paddingやリサイズ誤差を考慮）
    max_trim_h = int(w * sin_rot * 2) + 2  # 上下方向の最大トリム
    max_trim_w = int(h * sin_rot * 2) + 2  # 左右方向の最大トリム
    # 全体の上限も設ける（画像の15%まで）
    max_trim_h = min(max_trim_h, int(h * 0.15))
    max_trim_w = min(max_trim_w, int(w * 0.15))

    # 上辺
    top_trim = 0
    for row in range(min(max_trim_h, h // 2)):
        if (gray[row, :] < dark_thresh).mean() >= min_dark_ratio:
            top_trim = row + 1
        else:
            break

    # 下辺
    bottom_trim = 0
    for row in range(1, min(max_trim_h, h // 2) + 1):
        if (gray[h - row, :] < dark_thresh).mean() >= min_dark_ratio:
            bottom_trim = row
        else:
            break

    # 左辺
    left_trim = 0
    for col in range(min(max_trim_w, w // 2)):
        if (gray[:, col] < dark_thresh).mean() >= min_dark_ratio:
            left_trim = col + 1
        else:
            break

    # 右辺
    right_trim = 0
    for col in range(1, min(max_trim_w, w // 2) + 1):
        if (gray[:, w - col] < dark_thresh).mean() >= min_dark_ratio:
            right_trim = col
        else:
            break

    total_trim = top_trim + bottom_trim + left_trim + right_trim
    if diag is not None:
        diag["dark_border_trim"] = {
            "top": top_trim, "bottom": bottom_trim,
            "left": left_trim, "right": right_trim,
        }

    if total_trim > 0:
        y0 = top_trim
        y1 = h - bottom_trim
        x0 = left_trim
        x1 = w - right_trim
        if y1 > y0 and x1 > x0:
            return image[y0:y1, x0:x1]

    return image


def template_match_crop(imageA, imageB, try_rotations=False, min_score=0.3, padding_ratio=0.05):
    """
    テンプレートマッチングによるフォールバッククロップ。

    特徴点マッチが全スケールで失敗した場合に使用。
    マスター（imageA）をテンプレートとして比較画像（imageB）内で位置検索し、
    見つかった位置で矩形クロップする。

    Args:
        imageA: マスター画像（テンプレート）
        imageB: 比較画像（検索対象）
        try_rotations: Trueの場合、比較画像の4方向回転（0°, 90°, 180°, 270°）も試す
        min_score: テンプレートマッチングの最小相関スコア（TM_CCOEFF_NORMED）
        padding_ratio: クロップ領域のpadding率（見切れ防止）

    Returns:
        (cropped_image, diag_dict) or (None, diag_dict)
    """
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) if len(imageA.shape) == 3 else imageA.copy()
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) if len(imageB.shape) == 3 else imageB.copy()

    hA, wA = grayA.shape[:2]
    hB, wB = grayB.shape[:2]

    # 探索スケール: マスターを縮小してテンプレートにする
    candidate_scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    # テンプレートの最小ピクセル数（これ未満は信頼性が低い）
    min_template_px = 100
    # ピラミッドのダウンサンプル率（大画像ほど効果大）
    _ds = 4
    _min_coarse_template = 25  # 粗い探索でのテンプレート最小サイズ

    # --- 収束型回転探索 ---
    best_score = -1.0
    best_result = None  # (score, scale, rotation, x, y, tw, th, rotated_imageB)

    def _eval_rotation(rot_deg):
        """
        指定角度でピラミッド型テンプレートマッチ。
        Step1: 1/_ds解像度で全スケール粗探索 → ベストスケール・位置を特定
        Step2: ベストスケール±0.05の3スケールをROI内で精密マッチ
        """
        nonlocal best_score, best_result
        if abs(rot_deg % 360) < 1e-6:
            img_search = grayB
            img_search_color = imageB
        else:
            img_search = _rotate_image_by_angle(grayB, rot_deg % 360)
            img_search_color = _rotate_image_by_angle(imageB, rot_deg % 360)

        hS, wS = img_search.shape[:2]
        local_best_score = -1.0

        # Step 1: 粗い探索（1/_ds解像度で全スケール）
        small_A = cv2.resize(grayA, (wA // _ds, hA // _ds), interpolation=cv2.INTER_AREA)
        small_S = cv2.resize(img_search, (wS // _ds, hS // _ds), interpolation=cv2.INTER_AREA)
        shA, swA = small_A.shape[:2]
        shS, swS = small_S.shape[:2]

        coarse_best_score = -1.0
        coarse_best_scale = None
        coarse_best_loc = None

        for scale in candidate_scales:
            tw = int(swA * scale)
            th = int(shA * scale)
            if tw >= swS or th >= shS or tw < _min_coarse_template or th < _min_coarse_template:
                continue
            template = cv2.resize(small_A, (tw, th), interpolation=cv2.INTER_LINEAR)
            try:
                result = cv2.matchTemplate(small_S, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except Exception:
                continue
            if max_val > coarse_best_score:
                coarse_best_score = max_val
                coarse_best_scale = scale
                coarse_best_loc = (max_loc[0] * _ds, max_loc[1] * _ds)

        if coarse_best_scale is None:
            return local_best_score

        # Step 2: 精密マッチ（ベストスケール±0.05をROI内で）
        refine_scales = [coarse_best_scale]
        if coarse_best_scale + 0.05 <= 1.0:
            refine_scales.append(coarse_best_scale + 0.05)
        if coarse_best_scale - 0.05 >= 0.1:
            refine_scales.append(coarse_best_scale - 0.05)

        # ROI: 粗い位置の周辺（テンプレートサイズ×0.3のマージン）
        margin = int(max(wA, hA) * 0.3)
        tw_est = int(wA * coarse_best_scale)
        th_est = int(hA * coarse_best_scale)
        roi_x0 = max(0, coarse_best_loc[0] - margin)
        roi_y0 = max(0, coarse_best_loc[1] - margin)
        roi_x1 = min(wS, coarse_best_loc[0] + tw_est + margin)
        roi_y1 = min(hS, coarse_best_loc[1] + th_est + margin)
        roi = img_search[roi_y0:roi_y1, roi_x0:roi_x1]

        for scale in refine_scales:
            tw = int(wA * scale)
            th = int(hA * scale)
            if tw >= roi.shape[1] or th >= roi.shape[0] or tw < min_template_px or th < min_template_px:
                continue
            template = cv2.resize(grayA, (tw, th), interpolation=cv2.INTER_LINEAR)
            try:
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except Exception:
                continue

            # ROI座標を元画像座標に変換
            abs_x = roi_x0 + max_loc[0]
            abs_y = roi_y0 + max_loc[1]

            if max_val > local_best_score:
                local_best_score = max_val
            if max_val > best_score and max_val >= min_score:
                best_score = max_val
                best_result = (max_val, scale, rot_deg % 360, abs_x, abs_y, tw, th, img_search_color)

        return local_best_score

    if not try_rotations:
        # 回転探索なし: 0°のみ
        _eval_rotation(0.0)
    else:
        # 1段目: 粗い探索（0°, 90°, 180°, 270°）
        coarse_angles = [0.0, 90.0, 180.0, 270.0]
        scored = []
        for ang in coarse_angles:
            s = _eval_rotation(ang)
            scored.append((s, ang))
        scored.sort(key=lambda x: -x[0])

        # 上位2角度を取得
        top1_angle = scored[0][1]
        top2_angle = scored[1][1] if len(scored) > 1 else top1_angle

        # 2段目以降: 上位2角度の周辺を細分化して収束
        # 各上位角度の±20°範囲を探索し、繰り返し絞り込む
        search_range = 20.0   # 初期探索範囲（±度）
        search_step = 5.0     # 初期刻み
        min_step = 0.01       # 収束終了の最小刻み（0.01°精度）

        while search_step >= min_step - 1e-9:
            new_scored = []
            tried_angles = set()
            for base_angle in [top1_angle, top2_angle]:
                delta = -search_range
                while delta <= search_range + 1e-6:
                    ang = base_angle + delta
                    ang_key = round(ang % 360, 4)  # 0.01°精度で重複排除
                    if ang_key not in tried_angles:
                        tried_angles.add(ang_key)
                        s = _eval_rotation(ang)
                        new_scored.append((s, ang % 360))
                    delta += search_step

            if new_scored:
                new_scored.sort(key=lambda x: -x[0])
                top1_angle = new_scored[0][1]
                top2_angle = new_scored[1][1] if len(new_scored) > 1 else top1_angle

            # 探索範囲を縮小して次のラウンドへ
            search_range = search_step  # 前回の刻みが次の範囲
            search_step = search_step / 5.0
            if search_step < min_step - 1e-9:
                break

    diag = {
        "method": "template_match",
        "best_score": round(best_score, 4) if best_score > 0 else None,
        "min_score": min_score,
        "tried_rotations": try_rotations,
    }

    if best_result is None:
        diag["reject"] = "no_match_above_threshold"
        return None, diag

    score, scale, rot_deg, x, y, tw, th, img_color = best_result
    diag["adopted_scale"] = round(scale, 2)
    diag["adopted_rotation"] = round(rot_deg, 2)
    diag["match_position"] = (x, y)
    diag["template_size"] = (tw, th)
    diag["score"] = round(score, 4)

    # padding付きでクロップ領域を算出
    # 回転由来の黒帯が発生する分、追加paddingを加える
    abs_rot = abs(rot_deg % 360)
    if abs_rot > 180:
        abs_rot = 360 - abs_rot
    sin_rot = math.sin(math.radians(abs_rot))
    # 回転で上下に tw*sin(θ), 左右に th*sin(θ) の黒帯が発生
    rot_pad_x = int(math.ceil(th * sin_rot)) + 1 if abs_rot > 0.1 else 0
    rot_pad_y = int(math.ceil(tw * sin_rot)) + 1 if abs_rot > 0.1 else 0

    pad_x = int(tw * padding_ratio) + rot_pad_x
    pad_y = int(th * padding_ratio) + rot_pad_y
    crop_x0 = max(0, x - pad_x)
    crop_y0 = max(0, y - pad_y)
    crop_x1 = min(img_color.shape[1], x + tw + pad_x)
    crop_y1 = min(img_color.shape[0], y + th + pad_y)

    diag["crop_region"] = (crop_x0, crop_y0, crop_x1, crop_y1)
    diag["padding_ratio"] = padding_ratio
    diag["rotation_extra_pad"] = (rot_pad_x, rot_pad_y)

    cropped = img_color[crop_y0:crop_y1, crop_x0:crop_x1]
    if cropped.size == 0:
        diag["reject"] = "empty_crop"
        return None, diag

    # --- 回転由来の暗い縁を自動トリム ---
    # 回転した画像からクロップすると端に黒い三角が残る。
    # 追加paddingで広めにクロップした分、ここでトリムしてちょうど良くする。
    if abs_rot > 0.1:
        cropped = _trim_dark_borders(cropped, dark_thresh=30, min_dark_ratio=0.5,
                                     rotation_deg=rot_deg, diag=diag)

    # マスターサイズにリサイズ（アスペクト比保持 + 黒パディング）
    h_crop, w_crop = cropped.shape[:2]
    scale_x = wA / max(w_crop, 1)
    scale_y = hA / max(h_crop, 1)
    if abs(scale_x - scale_y) > 0.005:
        # アスペクト比が異なる → 均一スケールでフィットし黒パディング
        uniform_scale = min(scale_x, scale_y)
        new_w = int(round(w_crop * uniform_scale))
        new_h = int(round(h_crop * uniform_scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 不足分を黒パディング（右辺・下辺に追加、左上起点で位置保持）
        if resized.ndim == 3:
            padded = np.zeros((hA, wA, resized.shape[2]), dtype=resized.dtype)
        else:
            padded = np.zeros((hA, wA), dtype=resized.dtype)
        paste_h = min(new_h, hA)
        paste_w = min(new_w, wA)
        padded[:paste_h, :paste_w] = resized[:paste_h, :paste_w]
        cropped = padded
        diag["aspect_ratio_pad"] = {
            "original": (w_crop, h_crop),
            "scale_x": round(scale_x, 4),
            "scale_y": round(scale_y, 4),
            "uniform_scale": round(uniform_scale, 4),
            "padded": (wA - paste_w, hA - paste_h),
        }
    else:
        cropped = cv2.resize(cropped, (wA, hA), interpolation=cv2.INTER_LINEAR)

    # 回転していた場合、逆回転して戻す必要はない
    # （比較画像を回転して探索し、その回転後の画像からクロップしているため、
    #   クロップ結果はマスターと同じ向きになっている）

    return cropped, diag
