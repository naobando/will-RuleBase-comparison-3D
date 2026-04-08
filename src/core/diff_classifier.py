"""
差分の性質別分類モジュール

差分画像を「構造（歪み）」「傷」「ナット」「その他」の4種類に分離する。

分類ロジック（二段階）:
  第1段階 - 空間領域分離（2ゾーン）:
    structure_zone = fg_mask - erode(fg_mask)   外周エッジ帯
    internal_zone  = erode(fg_mask)             内部領域

  第2段階 - 各ゾーン内の差分検出:
    structure_zone → 構造差分（raw_diff）
    internal_zone  → 平坦面フィルタ → 傷検出（pipeline_diff）
                   → ナット検出: 暗ブロブ穴検出（raw_diff）
                   → その他差分 = internal raw_diff - 傷 - ナット
"""
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def classify_diff(
    diff_gray,
    fg_mask,
    image_gray=None,
    pipeline_diff=None,
    # ゾーン分離パラメータ
    structure_erode_iter=10,
    structure_kernel_size=5,
    # 構造マスク後処理
    structure_close_kernel=25,
    structure_close_iter=3,
    # 平坦面フィルタパラメータ
    flat_canny_low=50,
    flat_canny_high=150,
    # ナット検出パラメータ（エッジ密度チェック用）
    nut_canny_low=50,
    nut_canny_high=150,
    flat_edge_dilate_iter=2,
    flat_density_block_size=32,
    flat_density_threshold=0.08,
    # 傷の判別パラメータ
    scratch_min_aspect=2.5,
    scratch_min_area=30,
    scratch_max_area_ratio=0.03,
    scratch_min_mean_intensity=30,
    # ナットの判別パラメータ
    nut_adaptive_block_size=51,
    nut_adaptive_c=10,
    nut_min_hole_area=100,
    nut_max_hole_area=10000,
    nut_min_circularity=0.5,
    nut_ring_width=15,
    nut_min_edge_density=0.15,
    nut_edge_ring_width=10,
    nut_min_compactness=0.3,
    # その他差分パラメータ
    other_min_area=50,
    # 差分閾値
    diff_thresh=10,
):
    """
    差分画像を性質別に分類する。

    入力の使い分け:
      - diff_gray (raw_diff): 構造・ナット・その他差分の検出に使用。
            パイプラインのフィルタ前の素の差分。
      - pipeline_diff: 傷検出のみに使用。
            パイプラインのノイズ除去済み差分。平坦面上の傷検出に適する。
            Noneの場合はdiff_grayを使用。

    Args:
        diff_gray: 素の差分画像 (uint8, 0-255)。cv2.absdiff() の出力。
        fg_mask: 前景マスク (uint8, 0/255)。255=部品前景。
        image_gray: 元画像のグレースケール。平坦面・ナット検出用。
        pipeline_diff: パイプライン処理済み差分 (uint8)。傷検出用。
        diff_thresh: 差分を有意とみなす閾値

    Returns:
        result: 分類結果の辞書
            - structure_mask: 構造差分マスク (uint8, 0/255)
            - scratch_mask: 傷差分マスク (uint8, 0/255)
            - nut_mask: ナット差分マスク (uint8, 0/255)
            - other_mask: その他差分マスク (uint8, 0/255)
            - structure_zone: 構造領域（外周エッジ帯）のマスク
            - internal_zone: 内部領域のマスク
            - flat_zone: 平坦面マスク
            - nut_zone: ナット領域マスク
            - info: 詳細情報の辞書
    """
    h, w = diff_gray.shape[:2]
    total_area = h * w

    # pipeline_diffが未指定ならdiff_grayを使用
    if pipeline_diff is None:
        pipeline_diff = diff_gray

    # pipeline_diffのサイズ合わせ
    if pipeline_diff.shape[:2] != (h, w):
        pipeline_diff = cv2.resize(pipeline_diff, (w, h))

    info = {
        "structure_pixels": 0,
        "scratch_pixels": 0,
        "nut_pixels": 0,
        "other_pixels": 0,
        "scratch_count": 0,
        "nut_count": 0,
        "other_count": 0,
        "holes_detected": 0,
        "flat_zone_pixels": 0,
    }

    # 出力マスク初期化
    structure_mask = np.zeros((h, w), dtype=np.uint8)
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    nut_mask = np.zeros((h, w), dtype=np.uint8)
    other_mask = np.zeros((h, w), dtype=np.uint8)

    # 構造・ナット・その他用: raw_diffを閾値で二値化
    raw_binary = (diff_gray >= diff_thresh).astype(np.uint8) * 255

    # 傷検出用: pipeline_diffを閾値で二値化
    pipeline_binary = (pipeline_diff >= diff_thresh).astype(np.uint8) * 255

    # ========================================
    # 第1段階: 空間領域分離（2ゾーン）
    # ========================================
    structure_zone, internal_zone = _define_zones(
        fg_mask, structure_erode_iter, structure_kernel_size
    )

    # 構造差分 = 構造領域内の有意差分（raw_diffを使用）
    structure_mask = cv2.bitwise_and(raw_binary, structure_zone)

    # 構造マスクのclosing: 外周帯に散らばる断片を結合して視認性を上げる
    if structure_close_iter > 0 and np.count_nonzero(structure_mask) > 0:
        ck = max(3, structure_close_kernel if structure_close_kernel % 2 else structure_close_kernel + 1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        structure_mask = cv2.morphologyEx(
            structure_mask, cv2.MORPH_CLOSE, close_kernel, iterations=structure_close_iter
        )
        structure_mask = cv2.bitwise_and(structure_mask, structure_zone)

    info["structure_pixels"] = int(np.count_nonzero(structure_mask))

    # ========================================
    # 第2段階: 内部差分の分類
    # ========================================

    # --- 平坦面検出 ---
    flat_zone = np.zeros((h, w), dtype=np.uint8)
    if image_gray is not None:
        flat_zone = _detect_flat_zone(
            image_gray,
            internal_zone,
            canny_low=flat_canny_low,
            canny_high=flat_canny_high,
            edge_dilate_iter=flat_edge_dilate_iter,
            density_block_size=flat_density_block_size,
            density_threshold=flat_density_threshold,
        )
    info["flat_zone_pixels"] = int(np.count_nonzero(flat_zone))

    # --- 傷検出（pipeline_diff、平坦面のみ）---
    scratch_candidate = cv2.bitwise_and(pipeline_binary, flat_zone)

    if np.count_nonzero(scratch_candidate) > 0:
        scratch_mask, scratch_count = _filter_scratch_diff(
            scratch_candidate,
            diff_gray=pipeline_diff,
            min_aspect=scratch_min_aspect,
            min_area=scratch_min_area,
            max_area=int(total_area * scratch_max_area_ratio),
            min_mean_intensity=scratch_min_mean_intensity,
        )
        info["scratch_pixels"] = int(np.count_nonzero(scratch_mask))
        info["scratch_count"] = scratch_count

    # --- ナット検出（raw_diff、内部全体）---
    nut_zone_mask = np.zeros((h, w), dtype=np.uint8)
    if image_gray is not None:
        nut_zone_mask, nut_info = _detect_nut_zones(
            image_gray,
            internal_zone,
            adaptive_block_size=nut_adaptive_block_size,
            adaptive_c=nut_adaptive_c,
            min_hole_area=nut_min_hole_area,
            max_hole_area=nut_max_hole_area,
            min_circularity=nut_min_circularity,
            ring_width=nut_ring_width,
            min_edge_density=nut_min_edge_density,
            edge_ring_width=nut_edge_ring_width,
            canny_low=nut_canny_low,
            canny_high=nut_canny_high,
        )
        info["holes_detected"] = nut_info["holes_detected"]

    if np.count_nonzero(nut_zone_mask) > 0:
        nut_candidate = cv2.bitwise_and(raw_binary, nut_zone_mask)
        nut_mask, nut_count = _filter_nut_diff(
            nut_candidate, min_compactness=nut_min_compactness
        )
        info["nut_pixels"] = int(np.count_nonzero(nut_mask))
        info["nut_count"] = nut_count

    # --- その他差分（raw_diff、内部 - 傷 - ナット）---
    other_candidate = cv2.bitwise_and(raw_binary, internal_zone)
    if np.count_nonzero(scratch_mask) > 0:
        other_candidate = cv2.bitwise_and(other_candidate, cv2.bitwise_not(scratch_mask))
    if np.count_nonzero(nut_mask) > 0:
        other_candidate = cv2.bitwise_and(other_candidate, cv2.bitwise_not(nut_mask))

    other_count = 0
    if np.count_nonzero(other_candidate) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            other_candidate, connectivity=8
        )
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= other_min_area:
                other_mask[labels == lbl] = 255
                other_count += 1

    info["other_pixels"] = int(np.count_nonzero(other_mask))
    info["other_count"] = other_count

    logger.info(
        f"[DiffClassifier] 構造={info['structure_pixels']}px, "
        f"傷={info['scratch_pixels']}px({info['scratch_count']}件), "
        f"ナット={info['nut_pixels']}px({info['nut_count']}件), "
        f"その他={info['other_pixels']}px({info['other_count']}件), "
        f"穴検出={info['holes_detected']}個, "
        f"平坦面={info['flat_zone_pixels']}px"
    )

    return {
        "structure_mask": structure_mask,
        "scratch_mask": scratch_mask,
        "nut_mask": nut_mask,
        "other_mask": other_mask,
        "structure_zone": structure_zone,
        "internal_zone": internal_zone,
        "flat_zone": flat_zone,
        "nut_zone": nut_zone_mask,
        "info": info,
    }


def _define_zones(fg_mask, erode_iter, kernel_size):
    """
    前景マスクから構造領域・内部領域の2ゾーンを定義する。

    1. structure_zone = fg_mask - erode(fg_mask) -- 外周エッジ帯
    2. internal_zone  = erode(fg_mask)           -- 内部領域

    Args:
        fg_mask: 前景マスク (uint8, 0/255)
        erode_iter: 構造帯の幅（erosion回数）
        kernel_size: erosionカーネルサイズ

    Returns:
        (structure_zone, internal_zone) の2タプル
    """
    if fg_mask is None or np.count_nonzero(fg_mask) == 0:
        h, w = fg_mask.shape[:2] if fg_mask is not None else (1, 1)
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, np.ones((h, w), dtype=np.uint8) * 255

    ksize = max(3, kernel_size if kernel_size % 2 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(fg_mask, kernel, iterations=erode_iter)

    structure_zone = cv2.bitwise_and(fg_mask, cv2.bitwise_not(eroded))
    internal_zone = eroded.copy()

    logger.info(
        f"[Zones] structure={np.count_nonzero(structure_zone)}px, "
        f"internal={np.count_nonzero(internal_zone)}px"
    )

    return structure_zone, internal_zone


def _detect_flat_zone(
    image_gray,
    internal_zone,
    canny_low=50,
    canny_high=150,
    edge_dilate_iter=2,
    density_block_size=32,
    density_threshold=0.08,
):
    """
    内部領域から平坦面（エッジ密度が低い領域）を検出する。

    Args:
        image_gray: 元画像のグレースケール (uint8)
        internal_zone: 内部領域マスク (uint8, 0/255)
        canny_low: Cannyエッジ検出の低閾値
        canny_high: Cannyエッジ検出の高閾値
        edge_dilate_iter: エッジの膨張回数
        density_block_size: エッジ密度計算のブロックサイズ (px)
        density_threshold: 平坦と判定するエッジ密度の閾値 (0.0-1.0)

    Returns:
        flat_zone: 平坦面マスク (uint8, 0/255)。internal_zoneのサブセット。
    """
    h, w = image_gray.shape[:2]

    if internal_zone is None or np.count_nonzero(internal_zone) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Cannyエッジ検出（元画像に対して実行）
    edges = cv2.Canny(image_gray, canny_low, canny_high)

    # 内部領域内に制限
    edges = cv2.bitwise_and(edges, internal_zone)

    # エッジを膨張して密度計算を安定化
    if edge_dilate_iter > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, dilate_kernel, iterations=edge_dilate_iter)

    # boxFilterでローカルエッジ密度を算出
    edge_float = (edges / 255.0).astype(np.float32)
    density = cv2.boxFilter(edge_float, -1, (density_block_size, density_block_size), normalize=True)

    # 密度が閾値以下の領域を平坦面とする
    flat_zone = ((density < density_threshold) * 255).astype(np.uint8)
    flat_zone = cv2.bitwise_and(flat_zone, internal_zone)

    logger.info(
        f"[FlatZone] flat={np.count_nonzero(flat_zone)}px, "
        f"internal={np.count_nonzero(internal_zone)}px, "
        f"ratio={np.count_nonzero(flat_zone)/max(np.count_nonzero(internal_zone),1):.2f}"
    )

    return flat_zone


def _detect_nut_zones(
    image_gray,
    internal_zone,
    adaptive_block_size=51,
    adaptive_c=10,
    min_hole_area=100,
    max_hole_area=10000,
    min_circularity=0.5,
    ring_width=15,
    min_edge_density=0.15,
    edge_ring_width=10,
    canny_low=50,
    canny_high=150,
):
    """
    暗いブロブ（穴）検出でナット領域を特定し、リング状のゾーンマスクを返す。

    アルゴリズム:
      1. adaptiveThreshold(inverse) で暗い領域を二値化
      2. connectedComponents で個別の暗いブロブを抽出
      3. circularity (4*pi*area/perimeter^2) + サイズでフィルタ
      4. 各穴候補の周辺エッジ密度を確認（機械的特徴の検証）
      5. 確認済みの穴の周囲にリング状マスクを生成

    Returns:
        (nut_zone, info): nut_zoneマスク (uint8, 0/255) と検出情報辞書
    """
    h, w = image_gray.shape[:2]
    info = {"holes_detected": 0}
    nut_zone = np.zeros((h, w), dtype=np.uint8)

    if internal_zone is None or np.count_nonzero(internal_zone) == 0:
        return nut_zone, info

    # 内部領域内のみで検出
    masked_gray = cv2.bitwise_and(image_gray, internal_zone)

    # adaptive_block_sizeは奇数にする
    abs_ = max(3, adaptive_block_size if adaptive_block_size % 2 else adaptive_block_size + 1)

    # adaptive threshold (inverse) で暗い領域を検出
    binary = cv2.adaptiveThreshold(
        masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, abs_, adaptive_c
    )
    binary = cv2.bitwise_and(binary, internal_zone)

    # エッジ（穴周辺の機械的特徴チェック用）
    edges = cv2.Canny(image_gray, int(canny_low), int(canny_high))

    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    holes_found = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_hole_area or area > max_hole_area:
            continue

        # 真円度チェック
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        # 周辺エッジ密度チェック
        cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
        r = int(np.sqrt(area / np.pi))
        inner_r = r
        outer_r = r + edge_ring_width

        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), outer_r, 255, -1)
        cv2.circle(ring_mask, (cx, cy), inner_r, 0, -1)
        ring_mask = cv2.bitwise_and(ring_mask, internal_zone)

        ring_area = np.count_nonzero(ring_mask)
        if ring_area == 0:
            continue

        edge_in_ring = np.count_nonzero(cv2.bitwise_and(edges, ring_mask))
        edge_density = edge_in_ring / ring_area

        if edge_density < min_edge_density:
            continue

        # 穴確定: リング状ゾーンマスクを生成
        cv2.circle(nut_zone, (cx, cy), r + ring_width, 255, -1)
        holes_found += 1

        logger.debug(
            f"[NutDetect] hole: cx={cx}, cy={cy}, r={r}, "
            f"circ={circularity:.2f}, edge_dens={edge_density:.2f}"
        )

    # 内部領域でクリップ
    nut_zone = cv2.bitwise_and(nut_zone, internal_zone)
    info["holes_detected"] = holes_found

    logger.info(f"[NutDetect] 穴検出数={holes_found}")

    return nut_zone, info


def _filter_nut_diff(nut_candidate, min_compactness=0.3):
    """
    ナット領域内の差分から、塊状（コンパクト）な成分だけを抽出する。
    """
    if np.count_nonzero(nut_candidate) == 0:
        return nut_candidate, 0

    # 連結成分解析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nut_candidate, connectivity=8
    )

    result = np.zeros_like(nut_candidate)
    count = 0

    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl]
        if area < 10:
            continue

        # コンパクトネス（面積 / BBOX面積）
        bbox_area = max(bw * bh, 1)
        compactness = area / bbox_area

        if compactness >= min_compactness:
            result[labels == lbl] = 255
            count += 1

    return result, count


def _filter_scratch_diff(
    scratch_candidate, diff_gray=None, min_aspect=2.5, min_area=30,
    max_area=10000, min_mean_intensity=30,
):
    """
    内部差分から線状（高アスペクト比）の成分だけを抽出する。

    Args:
        scratch_candidate: 二値マスク (uint8, 0/255)
        diff_gray: 元の差分画像（強度フィルタ用）。Noneの場合は強度フィルタをスキップ。
        min_aspect: 最小アスペクト比
        min_area: 最小面積 (px)
        max_area: 最大面積 (px)
        min_mean_intensity: 各成分の差分平均強度の最小値。
            微小ノイズ由来の成分を除外する（正常画像の誤検出対策）。
    """
    if np.count_nonzero(scratch_candidate) == 0:
        return scratch_candidate, 0

    # 連結成分解析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        scratch_candidate, connectivity=8
    )

    result = np.zeros_like(scratch_candidate)
    count = 0

    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl]
        if area < min_area or area > max_area:
            continue

        # アスペクト比
        long_side = max(bw, bh)
        short_side = max(min(bw, bh), 1)
        aspect = long_side / short_side

        if aspect < min_aspect:
            continue

        # 差分強度フィルタ: 微小ノイズ由来の成分を除外
        if diff_gray is not None and min_mean_intensity > 0:
            component_mask = (labels == lbl)
            mean_intensity = float(np.mean(diff_gray[component_mask]))
            if mean_intensity < min_mean_intensity:
                logger.debug(
                    f"[ScratchFilter] 除外: area={area}, aspect={aspect:.1f}, "
                    f"mean_intensity={mean_intensity:.1f} < {min_mean_intensity}"
                )
                continue

        result[labels == lbl] = 255
        count += 1

    return result, count


def masks_to_bboxes(mask, min_area=50, connectivity=8, close_kernel_size=0):
    """
    マスクから BBOX リストを生成する。

    Args:
        mask: 二値マスク (uint8, 0/255)
        min_area: 最小面積
        connectivity: 連結成分の近傍
        close_kernel_size: BBOX生成前のclosingカーネルサイズ (px)。
            0=無効。近接する断片を結合して1つの特徴=1BBOXにする。

    Returns:
        bboxes: [(x, y, w, h), ...] のリスト
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return []

    work = mask
    if close_kernel_size > 0:
        ks = max(3, close_kernel_size if close_kernel_size % 2 else close_kernel_size + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        work = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        work, connectivity=connectivity
    )

    bboxes = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area >= min_area:
            bboxes.append((int(x), int(y), int(w), int(h)))

    return bboxes
