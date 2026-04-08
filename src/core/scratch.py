"""
傷検出モジュール

位置合わせ済みのマスター画像とテスト画像から、
金属表面の傷を検出する機能を提供する。

パイプライン:
  1. グレースケール + Gaussian Blur
  2. 正常画像との差分（absdiff）
  3. Black-hat 処理（暗い細線構造を強調）
  4. diff と blackhat を max で融合
  5. 二値化 + Morphology
  6. 小領域除去（connectedComponents）
  7. 傷指標算出 + BBOX生成

参考: 傷検出アルゴリズム検討・検証結果に基づく実装
"""

import cv2
from src.utils.image_utils import safe_imwrite
import numpy as np
import os


def detect_scratches(
    imageA,
    imageB,
    blur_ksize=5,
    blackhat_kernel_size=15,
    diff_thresh=40,
    min_area=50,
    min_width=0,
    morph_kernel_size=3,
    morph_open_iter=1,
    morph_close_iter=1,
    max_scratches=20,
    min_aspect_ratio=2.0,
    max_area_ratio=5.0,
    fg_mask=None,
    fg_mask_erode_iter=10,
    debug_dir=None,
):
    """
    位置合わせ済み画像から傷を検出する。

    Args:
        imageA: マスター画像（BGR or グレー）
        imageB: テスト画像（BGR or グレー）
        blur_ksize: Gaussian Blur カーネルサイズ（奇数）
        blackhat_kernel_size: Black-hat カーネルサイズ（奇数、大きいほど太い傷も拾う）
        diff_thresh: 融合画像の二値化閾値
        min_area: 最小面積（これ未満の領域は除去）
        morph_kernel_size: Morphology カーネルサイズ（奇数）
        morph_open_iter: Open処理の回数（ノイズ除去）
        morph_close_iter: Close処理の回数（傷の連続性補完）
        max_scratches: 最大検出数
        min_aspect_ratio: 最小アスペクト比（傷は細長い。これ未満の丸い領域は除外）
        max_area_ratio: 単一領域の最大面積比率%（これ以上は傷ではなく面差分と判断）
        fg_mask: 前景マスク（255=前景）。指定時は前景領域のみで検出
        fg_mask_erode_iter: fg_maskの収縮回数（境界のズレ差分を除外）
        debug_dir: デバッグ画像保存先（Noneなら保存しない）

    Returns:
        dict: {
            "scratches": [(x, y, w, h), ...],  # 傷のBBOXリスト
            "mask": np.ndarray,                  # 傷マスク（255=傷）
            "metrics": {
                "total_area_px": int,
                "area_ratio_percent": float,
                "scratch_count": int,
                "max_length_px": int,
                "mean_aspect_ratio": float,
            },
            "fused": np.ndarray,  # diff+blackhat融合画像
        }
    """

    def _save_debug(name, img):
        if debug_dir is None:
            return
        try:
            safe_imwrite(os.path.join(debug_dir, f"scratch_{name}.png"), img)
        except Exception:
            pass

    # --- Step 1: グレースケール + Gaussian Blur ---
    if len(imageA.shape) == 3:
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    else:
        grayA = imageA.copy()

    if len(imageB.shape) == 3:
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    else:
        grayB = imageB.copy()

    blurA = cv2.GaussianBlur(grayA, (blur_ksize, blur_ksize), 0)
    blurB = cv2.GaussianBlur(grayB, (blur_ksize, blur_ksize), 0)

    _save_debug("01_blurA", blurA)
    _save_debug("01_blurB", blurB)

    # --- Step 2: 差分（absdiff） ---
    diff = cv2.absdiff(blurA, blurB)
    _save_debug("02_diff", diff)

    # --- Step 3: Black-hat 処理 ---
    # テスト画像のみに適用（傷＝暗い細線構造）
    bh_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (blackhat_kernel_size, blackhat_kernel_size)
    )
    blackhat = cv2.morphologyEx(blurB, cv2.MORPH_BLACKHAT, bh_kernel)
    _save_debug("03_blackhat", blackhat)

    # --- Step 4: diff と blackhat を max で融合 ---
    fused = np.maximum(diff, blackhat)
    _save_debug("04_fused", fused)

    # --- Step 5: 前景マスク適用（指定時） ---
    if fg_mask is not None:
        # fg_maskのサイズを合わせる
        if fg_mask.shape[:2] != fused.shape[:2]:
            fg_mask = cv2.resize(
                fg_mask, (fused.shape[1], fused.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        # fg_maskが3chの場合はグレーに変換
        if len(fg_mask.shape) == 3:
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        # 前景マスクを収縮して境界のズレ差分を除外（傷は内部にある）
        if fg_mask_erode_iter > 0:
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.erode(fg_mask, erode_k, iterations=fg_mask_erode_iter)
            _save_debug("04a_fg_mask_eroded", fg_mask)
        nz = cv2.countNonZero(fg_mask)
        print(f"[ScratchDetection] fg_mask適用: erode={fg_mask_erode_iter}, nonzero={nz}/{fg_mask.size} ({100*nz/fg_mask.size:.1f}%)")
        fused = cv2.bitwise_and(fused, fg_mask)
        _save_debug("04b_fused_masked", fused)
    else:
        print("[ScratchDetection] fg_mask=None（前景マスクなし）")

    # --- Step 6: 二値化 + Morphology ---
    _, binary = cv2.threshold(fused, diff_thresh, 255, cv2.THRESH_BINARY)
    _save_debug("05_binary", binary)

    morph_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )

    # Open: 点ノイズ除去
    if morph_open_iter > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_k, iterations=morph_open_iter)
        _save_debug("06_after_open", binary)

    # Close: 傷の連続性を補完
    if morph_close_iter > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_k, iterations=morph_close_iter)
        _save_debug("06_after_close", binary)

    # --- Step 7: 小領域除去 + フィルタ + BBOX生成 + 指標算出 ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    img_area = grayA.shape[0] * grayA.shape[1]
    max_single_area = int(img_area * max_area_ratio / 100.0) if max_area_ratio > 0 else img_area

    scratches = []
    total_area = 0
    max_length = 0
    aspect_ratios = []
    dropped_aspect = 0
    dropped_area = 0
    dropped_width = 0

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            # 小領域除去（ラベルを消す）
            labels[labels == label_id] = 0
            continue
        # 最小幅フィルタ: 短辺が小さすぎる線はエッジノイズの可能性が高い
        if min_width > 0 and min(w, h) < min_width:
            labels[labels == label_id] = 0
            dropped_width += 1
            continue
        # アスペクト比フィルタ: 傷は細長い形状のはず
        aspect = max(w, h) / max(min(w, h), 1)
        if min_aspect_ratio > 0 and aspect < min_aspect_ratio:
            labels[labels == label_id] = 0
            dropped_aspect += 1
            continue
        # 単一領域の面積上限: 巨大な領域は面差分であり傷ではない
        if area > max_single_area:
            labels[labels == label_id] = 0
            dropped_area += 1
            continue
        scratches.append((x, y, w, h, area))
        total_area += area
        length = max(w, h)
        max_length = max(max_length, length)
        aspect_ratios.append(aspect)

    if dropped_aspect > 0 or dropped_area > 0 or dropped_width > 0:
        print(f"[ScratchDetection] フィルタ除外: aspect={dropped_aspect}, area={dropped_area}, width={dropped_width}")

    # 面積順ソート（大きい順）
    scratches.sort(key=lambda s: s[4], reverse=True)

    # 上限適用
    if len(scratches) > max_scratches:
        scratches = scratches[:max_scratches]

    # マスクを再構築（小領域除去後）
    final_mask = (labels > 0).astype(np.uint8) * 255
    _save_debug("07_final_mask", final_mask)

    # BBOX形式に変換（areaは除外）
    bbox_list = [(x, y, w, h) for (x, y, w, h, _) in scratches]

    # 指標
    metrics = {
        "total_area_px": int(total_area),
        "area_ratio_percent": round(100.0 * total_area / max(img_area, 1), 4),
        "scratch_count": len(bbox_list),
        "max_length_px": int(max_length),
        "mean_aspect_ratio": round(
            float(np.mean(aspect_ratios)) if aspect_ratios else 0.0, 2
        ),
    }

    # デバッグ: BBOX描画
    if debug_dir is not None:
        vis = imageB.copy() if len(imageB.shape) == 3 else cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
        for (bx, by, bw, bh) in bbox_list:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)  # 黄色
        _save_debug("08_bbox_result", vis)

        # 指標をテキスト保存
        try:
            with open(os.path.join(debug_dir, "scratch_metrics.txt"), "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    f.write(f"{k}={v}\n")
                f.write(f"\nBBOX list ({len(bbox_list)}):\n")
                for i, (bx, by, bw, bh) in enumerate(bbox_list):
                    f.write(f"  [{i}] x={bx} y={by} w={bw} h={bh}\n")
        except Exception:
            pass

    print(f"[ScratchDetection] count={metrics['scratch_count']}, "
          f"total_area={metrics['total_area_px']}px, "
          f"area_ratio={metrics['area_ratio_percent']}%, "
          f"max_length={metrics['max_length_px']}px, "
          f"mean_aspect={metrics['mean_aspect_ratio']}")

    return {
        "scratches": bbox_list,
        "mask": final_mask,
        "metrics": metrics,
        "fused": fused,
    }
