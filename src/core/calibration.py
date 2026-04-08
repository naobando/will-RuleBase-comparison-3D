"""
マスター自己キャリブレーション

マスター画像に対し擬似的な摂動（回転・明るさ変動）を加えて自己比較し、
ノイズフロアを計測。E+SIFTの検出閾値を自動決定する。
"""

import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from .preprocessing import preprocess_image


def _compute_ensemble(procA, procB, e2scale_min_side=250):
    """
    Multi-Scale SSIM ensemble (Phase 1 of E+SIFT) を計算する。
    symmetry.py L958-991 と同一ロジック。

    Returns:
        ensemble: uint8 差分マップ (0-255)
    """
    h, w = procA.shape[:2]
    min_side = min(h, w)

    # 1x scale
    _, s1 = ssim(procA, procB, full=True, data_range=255.0)
    ed1 = ((1.0 - np.clip(s1, 0, 1)) * 255).astype(np.uint8)

    # 1/2x scale
    hA = cv2.resize(procA, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    hB = cv2.resize(procB, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ws2 = min(7, min(hA.shape[:2]) - 1)
    if ws2 % 2 == 0:
        ws2 -= 1
    ws2 = max(3, ws2)
    _, s2 = ssim(hA, hB, full=True, win_size=ws2, data_range=255.0)
    ed2 = cv2.resize(
        ((1.0 - np.clip(s2, 0, 1)) * 255).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )

    use_2scale = min_side < e2scale_min_side
    if use_2scale:
        return np.minimum(ed1, ed2)

    # 1/4x scale
    qA = cv2.resize(procA, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    qB = cv2.resize(procB, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    ws4 = min(7, min(qA.shape[:2]) - 1)
    if ws4 % 2 == 0:
        ws4 -= 1
    ws4 = max(3, ws4)
    _, s4 = ssim(qA, qB, full=True, win_size=ws4, data_range=255.0)
    ed4 = cv2.resize(
        ((1.0 - np.clip(s4, 0, 1)) * 255).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_LINEAR,
    )

    return np.minimum(np.minimum(ed1, ed2), ed4)


def _generate_perturbations(processed):
    """
    前処理済みマスター画像から摂動バリエーションを生成。
    回転2種 + 平行移動2種 + 明るさ4種 = 8パターン。

    回転角度は控えめ（±0.3°）。実際のカメラ再撮影では
    自動アライメントが大きなズレを補正するため、残差は微小。
    """
    h, w = processed.shape[:2]
    center = (w / 2.0, h / 2.0)
    perturbations = []

    # 回転: ±0.3°（アライメント残差レベル）
    for angle in [0.3, -0.3]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            processed, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        perturbations.append(rotated)

    # 平行移動: ±1px（サブピクセルアライメント残差）
    for dx, dy in [(1, 0), (0, 1)]:
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            processed, M_shift, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        perturbations.append(shifted)

    # 明るさ: ±5%, ±10%
    for factor in [1.05, 0.95, 1.10, 0.90]:
        brightened = np.clip(
            processed.astype(np.float32) * factor, 0, 255
        ).astype(np.uint8)
        perturbations.append(brightened)

    return perturbations


def _apply_masks(ensemble, procA, procB, h, w, fg_mask,
                 bg_brightness_max, edge_suppress_ratio, edge_suppress_max):
    """
    ensemble差分マップにfg_mask, bgクロマキー, edge抑制を適用。
    symmetry.py L983-1022 と同一ロジック。
    """
    result = ensemble.copy()

    # fg_mask
    if fg_mask is not None:
        result = cv2.bitwise_and(result, fg_mask)

    # 背景クロマキー（画像端に連結する暗領域のみマスク）
    if bg_brightness_max > 0:
        bg_dark = ((procA <= bg_brightness_max) | (procB <= bg_brightness_max)).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bg_dark, connectivity=8)
        bg_bg = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n):
            bx, by, bw, bh, _ = stats[i]
            if bx == 0 or by == 0 or (bx + bw) >= w or (by + bh) >= h:
                bg_bg[labels == i] = 255
        bg_dilated = cv2.dilate(
            bg_bg,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if int(np.count_nonzero(bg_dilated)) > 0:
            result[bg_dilated > 0] = 0

    # edge抑制
    if edge_suppress_ratio > 0:
        edge_band = max(1, int(edge_suppress_ratio * min(h, w)))
        if edge_suppress_max > 0:
            edge_band = min(edge_band, edge_suppress_max)
        result[:edge_band, :] = 0
        result[-edge_band:, :] = 0
        result[:, :edge_band] = 0
        result[:, -edge_band:] = 0

    return result


def calibrate_master(
    master_bgr,
    preprocess_mode="luminance",
    preprocess_blur_ksize=3,
    ensemble_2scale_min_side=250,
    ensemble_edge_suppress_ratio=0.04,
    ensemble_edge_suppress_max=8,
    ensemble_bg_brightness_max=30,
    fg_mask=None,
    config_fallback_thresh=70,
    config_fallback_min_mean=60,
    calibration_margin_ratio=0.3,
    calibration_min_thresh=30,
    calibration_max_thresh=200,
):
    """
    マスター画像の自己キャリブレーションを実行。

    Args:
        master_bgr: マスター画像 (BGR, numpy array)
        preprocess_mode: 前処理モード
        preprocess_blur_ksize: ブラーカーネルサイズ
        ensemble_2scale_min_side: 2スケール切替の短辺閾値
        ensemble_edge_suppress_ratio: 端帯抑制比率
        ensemble_edge_suppress_max: 端帯最大幅
        ensemble_bg_brightness_max: 背景クロマキー輝度閾値
        fg_mask: 前景マスク (None可)
        config_fallback_thresh: config.yamlのensemble_thresh値
        config_fallback_min_mean: config.yamlのensemble_bbox_min_mean値
        calibration_margin_ratio: ノイズp99に対するマージン比率
        calibration_min_thresh: キャリブレーション後の最小閾値
        calibration_max_thresh: キャリブレーション後の最大閾値

    Returns:
        dict: キャリブレーション結果
    """
    t0 = time.time()

    # 前処理
    processedM = preprocess_image(
        master_bgr, mode=preprocess_mode, blur_ksize=preprocess_blur_ksize,
    )
    h, w = processedM.shape[:2]

    # 摂動生成
    perturbations = _generate_perturbations(processedM)

    # 全摂動でensemble差分を計算し統計量を収集
    all_pixels = []
    noise_cluster_means = []

    for perturbed in perturbations:
        ensemble = _compute_ensemble(
            processedM, perturbed, e2scale_min_side=ensemble_2scale_min_side,
        )
        ensemble = _apply_masks(
            ensemble, processedM, perturbed, h, w, fg_mask,
            ensemble_bg_brightness_max,
            ensemble_edge_suppress_ratio,
            ensemble_edge_suppress_max,
        )

        # 非ゼロピクセルを収集
        nonzero = ensemble[ensemble > 0]
        if len(nonzero) > 0:
            all_pixels.append(nonzero)

        # ノイズクラスタの平均輝度を計測
        # config_fallback_threshで二値化して連結成分を見る
        _, tmask = cv2.threshold(ensemble, config_fallback_thresh, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(tmask) > 0:
            n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(tmask, connectivity=4)
            for ci in range(1, n_comp):
                cx, cy, cw, ch, carea = stats[ci]
                if carea >= 5:  # 極小クラスタは無視
                    roi = ensemble[cy:cy + ch, cx:cx + cw]
                    noise_cluster_means.append(float(np.mean(roi)))

    # 統計量計算
    if len(all_pixels) > 0:
        combined = np.concatenate(all_pixels)
        max_pixel = int(np.max(combined))
        p99_pixel = float(np.percentile(combined, 99))
        p95_pixel = float(np.percentile(combined, 95))
        mean_nonzero = float(np.mean(combined))
    else:
        # ノイズが全く出ない（非常にクリーンな画像）
        max_pixel = 0
        p99_pixel = 0.0
        p95_pixel = 0.0
        mean_nonzero = 0.0

    noise_cluster_mean_max = max(noise_cluster_means) if noise_cluster_means else 0.0

    # 閾値決定
    margin = max(10, int(p99_pixel * calibration_margin_ratio))
    calibrated_thresh = int(p99_pixel) + margin
    calibrated_thresh = max(calibration_min_thresh, min(calibration_max_thresh, calibrated_thresh))

    calibrated_bbox_min_mean = max(20, int(noise_cluster_mean_max * 1.5))
    # bbox_min_meanはthreshより少し下に設定（threshを超えた成分の中で弱いものを除く）
    calibrated_bbox_min_mean = max(calibrated_bbox_min_mean, calibrated_thresh - 20)

    elapsed_ms = (time.time() - t0) * 1000

    result = {
        "calibrated": True,
        "ensemble_thresh": calibrated_thresh,
        "ensemble_bbox_min_mean": calibrated_bbox_min_mean,
        "noise_stats": {
            "max_pixel": max_pixel,
            "p99_pixel": p99_pixel,
            "p95_pixel": p95_pixel,
            "mean_nonzero": mean_nonzero,
            "noise_cluster_mean_max": noise_cluster_mean_max,
            "n_perturbations": len(perturbations),
        },
        "config_fallback": {
            "ensemble_thresh": config_fallback_thresh,
            "ensemble_bbox_min_mean": config_fallback_min_mean,
        },
        "image_info": {
            "height": h,
            "width": w,
            "min_side": min(h, w),
            "area": h * w,
        },
        "elapsed_ms": elapsed_ms,
    }

    print(f"[CALIBRATION] thresh: {config_fallback_thresh} -> {calibrated_thresh} "
          f"(noise p99={p99_pixel:.1f}, margin={margin})")
    print(f"[CALIBRATION] bbox_min_mean: {config_fallback_min_mean} -> {calibrated_bbox_min_mean} "
          f"(noise cluster max mean={noise_cluster_mean_max:.1f})")
    print(f"[CALIBRATION] {elapsed_ms:.0f}ms, {len(perturbations)} perturbations, "
          f"image={w}x{h}")

    return result
