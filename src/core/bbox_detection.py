"""
BBOX検出モジュール

差分画像から変化領域のBBOX（境界ボックス）を検出する機能を提供します。
"""
import os

import cv2
from src.utils.image_utils import safe_imwrite
import numpy as np


def merge_nearby_bboxes(bboxes, distance_thresh=30, iou_thresh=0.1):
    """近接・重複するBBOXをマージして1要素1BBOXにする

    Args:
        bboxes: [(x, y, w, h), ...] のリスト
        distance_thresh: この距離以内のBBOXをマージ候補とする（px）
        iou_thresh: IoUがこの値以上のBBOXをマージする

    Returns:
        merged: [(x, y, w, h), ...] マージ後のリスト
    """
    if len(bboxes) <= 1:
        return list(bboxes)

    def _overlap_or_near(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        # 拡張BBOXで近接判定
        ex = distance_thresh
        a_x1, a_y1 = ax - ex, ay - ex
        a_x2, a_y2 = ax + aw + ex, ay + ah + ex
        b_x1, b_y1 = bx, by
        b_x2, b_y2 = bx + bw, by + bh
        # 拡張BBOXが重なるか
        if a_x1 >= b_x2 or b_x1 >= a_x2 or a_y1 >= b_y2 or b_y1 >= a_y2:
            return False
        # IoU計算（元のBBOXで）
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = aw * ah + bw * bh - inter
            if union > 0 and inter / union >= iou_thresh:
                return True
            # 元BBOXが重複しているがIoUが低い → 別の検出対象
            return False
        # 元BBOXは重なっていないが距離が近い → マージ
        return True

    def _merge_pair(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return (x1, y1, x2 - x1, y2 - y1)

    # Union-Find方式でクラスタリング
    n = len(bboxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap_or_near(bboxes[i], bboxes[j]):
                union(i, j)

    # クラスタごとにマージ
    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(bboxes[i])

    merged = []
    for group in clusters.values():
        result = group[0]
        for bb in group[1:]:
            result = _merge_pair(result, bb)
        merged.append(result)

    return merged


def diff_to_bboxes(
    diff_gray,
    thresh=13,
    min_area=1500,
    min_width=0,
    drop_band_aspect=0,
    drop_band_max_fill=0,
    drop_band_aspect_high=0,
    drop_band_max_fill_high=0,
    drop_sparse_large_cover=0,
    drop_sparse_large_max_fill=0,
    drop_band_min_area_ratio=0,
    morph_kernel=21,
    close_iter=4,
    open_iter=1,
    max_boxes=6,
    edge_ignore_ratio=None,
    min_area_relax_ratio=None,
    connectivity=8,
    edge_min_fill_ratio=None,
    drop_band_fill_ratio=None,
    drop_band_width_ratio=None,
    drop_band_height_ratio=None,
    inner_mask=None,
    inner_overlap_ratio=None,
    debug_path=None,
    debug_image_prefix=None,
):
    """
    差分(diff)画像から「白が密集した塊」だけをBBOX化（板金向け）

    仕組み：
      1) 二値化（閾値以上を白）
      2) Openで小ノイズ除去
      3) Closeで近い白を結合＆穴埋め → "塊"にする
      4) 連結成分を面積順に並べ、上位max_boxesだけ残す

    Args:
        diff_gray: 差分画像（グレースケール, 0-255）
        thresh: 差分閾値（小さいほど拾う）
        min_area: 最小面積(px)（小さい塊を捨てる）
        morph_kernel: まとめる強さ（大きいほど塊になる）
        close_iter: 近接領域の結合＆穴埋めの強さ
        open_iter: 小ノイズ除去の強さ
        max_boxes: 表示するBBOX上限（例：5〜6）

    Returns:
        mask: 二値マスク（処理後）
        bboxes: [(x,y,w,h), ...]（面積の大きい順、最大max_boxes）
    """
    # 1) 二値化
    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)

    # 2) モルフォロジー（ノイズ除去→結合）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # 3) 連結成分（塊）抽出
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=int(connectivity))
    # stats: [label, x, y, w, h, area] ではなく (x,y,w,h,area) が入る
    h, w = diff_gray.shape[:2]
    image_area = float(h * w) if h and w else 0.0

    debug_lines = []
    reason_map = {}
    rescue_map = {}
    inner_mask_bin = None
    overlap_cache = {}

    if inner_mask is not None:
        try:
            if inner_mask.ndim == 3:
                inner_gray = cv2.cvtColor(inner_mask, cv2.COLOR_BGR2GRAY)
            else:
                inner_gray = inner_mask
            _, inner_mask_bin = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY)
        except Exception:
            inner_mask_bin = None

    def _log(line):
        if debug_path is not None:
            debug_lines.append(line)

    def _inner_overlap(label, area):
        if inner_mask_bin is None or inner_overlap_ratio is None or area <= 0:
            return None, None
        cached = overlap_cache.get(label)
        if cached is not None:
            return cached
        component = labels == label
        overlap = int(np.count_nonzero(component & (inner_mask_bin > 0)))
        ratio = overlap / float(area) if area > 0 else 0.0
        overlap_cache[label] = (overlap, ratio)
        return overlap, ratio

    def _collect(min_area_thresh, pass_name):
        collected = []
        _log(f"[{pass_name}] min_area={min_area_thresh}")
        for label in range(1, num_labels):  # 0は背景
            x, y, bw, bh, area = stats[label]
            fill_ratio = float(area) / float(bw * bh) if bw > 0 and bh > 0 else 0.0
            if drop_band_fill_ratio is not None and drop_band_width_ratio is not None and drop_band_height_ratio is not None and image_area > 0:
                width_ratio = bw / float(diff_gray.shape[1])
                height_ratio = bh / float(diff_gray.shape[0])
                if width_ratio >= float(drop_band_width_ratio) and height_ratio <= float(drop_band_height_ratio) and fill_ratio <= float(drop_band_fill_ratio):
                    overlap, overlap_ratio = _inner_overlap(label, area)
                    if overlap is not None and overlap_ratio is not None and overlap_ratio >= float(inner_overlap_ratio):
                        _log(
                            f"  rescue_band label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                            f"width={width_ratio:.2f} height={height_ratio:.2f} fill={fill_ratio:.2f} "
                            f"inner_overlap={overlap} ratio={overlap_ratio:.2f}"
                        )
                        rescue_map[label] = "band"
                    else:
                        _log(
                            f"  drop_band label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                            f"width={width_ratio:.2f} height={height_ratio:.2f} fill={fill_ratio:.2f}"
                        )
                        reason_map[label] = "band"
                        continue
            if edge_ignore_ratio is not None and image_area > 0:
                touches_edge = x == 0 or y == 0 or (x + bw) >= diff_gray.shape[1] or (y + bh) >= diff_gray.shape[0]
                bbox_cover_ratio = (bw * bh) / image_area
                if touches_edge and bbox_cover_ratio >= float(edge_ignore_ratio):
                    if edge_min_fill_ratio is not None and bw > 0 and bh > 0:
                        if fill_ratio < float(edge_min_fill_ratio):
                            # edge-sparse成分は常にdrop → symmetry.pyのedge-cut MORPH_OPENで分離・救済
                            _log(
                                f"  drop_edge_sparse label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                f"cover={bbox_cover_ratio:.2f} fill={fill_ratio:.2f}"
                            )
                            reason_map[label] = "edge"
                            continue
                    else:
                        overlap, overlap_ratio = _inner_overlap(label, area)
                        if overlap is not None and overlap_ratio is not None and overlap_ratio >= float(inner_overlap_ratio):
                            _log(
                                f"  rescue_edge label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                f"cover={bbox_cover_ratio:.2f} inner_overlap={overlap} ratio={overlap_ratio:.2f}"
                            )
                            rescue_map[label] = "edge"
                        else:
                            _log(f"  drop_edge label={label} area={area} bbox=({x},{y},{bw},{bh}) cover={bbox_cover_ratio:.2f}")
                            reason_map[label] = "edge"
                            continue
            if min_width > 0 and min(bw, bh) < min_width:
                _log(f"  drop_min_width label={label} area={area} bbox=({x},{y},{bw},{bh}) min_side={min(bw, bh)}")
                reason_map[label] = "min_width"
                continue
            # 帯状・低密度大領域フィルタ（位置ずれによる帯状差分を除外）
            if bw > 0 and bh > 0:
                bbox_aspect = max(bw, bh) / float(min(bw, bh))
                bbox_cover = (bw * bh) / image_area if image_area > 0 else 0.0
                # 面積比による帯状フィルタ例外（大きな構造差分を保護）
                _area_ratio = area / image_area if image_area > 0 else 0.0
                _band_area_protected = (drop_band_min_area_ratio > 0 and _area_ratio >= float(drop_band_min_area_ratio))
                # 条件1: 高アスペクト比 + 低fill（帯状の位置ずれ差分）
                if drop_band_aspect > 0 and drop_band_max_fill > 0:
                    if bbox_aspect >= float(drop_band_aspect) and fill_ratio < float(drop_band_max_fill):
                        if _band_area_protected:
                            _log(f"  keep_band_large label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                 f"aspect={bbox_aspect:.1f} fill={fill_ratio:.2f} area_ratio={_area_ratio:.4f} (protected)")
                        else:
                            _log(f"  drop_band label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                 f"aspect={bbox_aspect:.1f} fill={fill_ratio:.2f}")
                            reason_map[label] = "band"
                            continue
                # 条件1b: 超高アスペクト比（≥20等）+ やや低fill（フレーム/レール構造差分）
                if drop_band_aspect_high > 0 and drop_band_max_fill_high > 0:
                    if bbox_aspect >= float(drop_band_aspect_high) and fill_ratio < float(drop_band_max_fill_high):
                        if _band_area_protected:
                            _log(f"  keep_band_high_large label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                 f"aspect={bbox_aspect:.1f} fill={fill_ratio:.2f} area_ratio={_area_ratio:.4f} (protected)")
                        else:
                            _log(f"  drop_band_high label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                                 f"aspect={bbox_aspect:.1f} fill={fill_ratio:.2f}")
                            reason_map[label] = "band"
                            continue
                # 条件2: 大面積カバー + 極低fill（大きなスカスカ領域）
                if drop_sparse_large_cover > 0 and drop_sparse_large_max_fill > 0:
                    if bbox_cover >= float(drop_sparse_large_cover) and fill_ratio < float(drop_sparse_large_max_fill):
                        _log(f"  drop_sparse_large label={label} area={area} bbox=({x},{y},{bw},{bh}) "
                             f"cover={bbox_cover:.2f} fill={fill_ratio:.2f}")
                        reason_map[label] = "sparse_large"
                        continue
            if area < min_area_thresh:
                _log(f"  drop_min_area label={label} area={area} bbox=({x},{y},{bw},{bh})")
                reason_map[label] = "min_area"
                continue
            if bw > 0 and bh > 0:
                _log(f"  keep label={label} area={area} bbox=({x},{y},{bw},{bh}) cover={(bw*bh)/image_area:.2f} fill={fill_ratio:.2f}")
            reason_map[label] = "keep"
            collected.append((label, x, y, bw, bh, area))
        return collected

    if debug_path is not None:
        _log(f"params: thresh={thresh} morph_kernel={morph_kernel} close_iter={close_iter} open_iter={open_iter} max_boxes={max_boxes}")
        _log(f"edge_ignore_ratio={edge_ignore_ratio} edge_min_fill_ratio={edge_min_fill_ratio} connectivity={connectivity}")
        _log(
            f"drop_band_fill_ratio={drop_band_fill_ratio} drop_band_width_ratio={drop_band_width_ratio} "
            f"drop_band_height_ratio={drop_band_height_ratio}"
        )
        if inner_mask_bin is not None and inner_overlap_ratio is not None:
            inner_nz = int(np.count_nonzero(inner_mask_bin))
            _log(f"inner_overlap_ratio={inner_overlap_ratio} inner_mask_nonzero={inner_nz}")

    reason_map = {}
    rescue_map = {}
    bboxes = _collect(min_area, "pass1")
    if not bboxes and min_area_relax_ratio is not None and min_area_relax_ratio > 0 and min_area > 0:
        relaxed = max(10, int(min_area * float(min_area_relax_ratio)))
        if relaxed < min_area:
            reason_map = {}
            rescue_map = {}
            bboxes = _collect(relaxed, "pass2")

    if debug_path is not None and rescue_map:
        rescue_edge = sum(1 for r in rescue_map.values() if r == "edge")
        rescue_band = sum(1 for r in rescue_map.values() if r == "band")
        _log(f"rescue_edge={rescue_edge} rescue_band={rescue_band}")

    if debug_path is not None:
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines))
        except Exception:
            pass

    # 4) 面積の大きい順に上位だけ残す
    bboxes = sorted(bboxes, key=lambda b: b[5], reverse=True)
    kept_bboxes = bboxes[:max_boxes]
    dropped_bboxes = bboxes[max_boxes:]
    kept_labels = {label for (label, x, y, w, h, area) in kept_bboxes}
    for label, reason in list(reason_map.items()):
        if reason == "keep" and label not in kept_labels:
            reason_map[label] = "max_boxes"

    # (x,y,w,h) に戻す
    bboxes_xywh = [(x, y, w, h) for (label, x, y, w, h, area) in kept_bboxes]
    if debug_path is not None and dropped_bboxes:
        try:
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"\n[drop_max_boxes] kept={len(kept_bboxes)} dropped={len(dropped_bboxes)} max_boxes={max_boxes}\n")
                h, w = diff_gray.shape[:2]
                image_area = float(h * w) if h and w else 0.0
                for label, x, y, bw, bh, area in dropped_bboxes:
                    fill_ratio = float(area) / float(bw * bh) if bw > 0 and bh > 0 else 0.0
                    cover = (bw * bh) / image_area if image_area > 0 else 0.0
                    f.write(f"  drop bbox=({x},{y},{bw},{bh}) area={area} cover={cover:.3f} fill={fill_ratio:.2f}\n")
        except Exception:
            pass
    if debug_image_prefix is not None:
        try:
            os.makedirs(os.path.dirname(debug_image_prefix), exist_ok=True)
            dropped_min_area = np.zeros_like(mask)
            dropped_edge = np.zeros_like(mask)
            dropped_band = np.zeros_like(mask)
            dropped_max_boxes = np.zeros_like(mask)
            dropped_all = np.zeros_like(mask)
            rescued_edge = np.zeros_like(mask)
            rescued_band = np.zeros_like(mask)
            rescued_all = np.zeros_like(mask)
            for label, reason in reason_map.items():
                if reason == "keep":
                    continue
                component = labels == label
                if reason == "min_area":
                    dropped_min_area[component] = 255
                elif reason == "edge":
                    dropped_edge[component] = 255
                elif reason in ("band", "sparse_large"):
                    dropped_band[component] = 255
                elif reason == "max_boxes":
                    dropped_max_boxes[component] = 255
                dropped_all[component] = 255
            for label, rescue in rescue_map.items():
                component = labels == label
                if rescue == "edge":
                    rescued_edge[component] = 255
                elif rescue == "band":
                    rescued_band[component] = 255
                rescued_all[component] = 255
            safe_imwrite(f"{debug_image_prefix}_dropped_min_area.png", dropped_min_area)
            safe_imwrite(f"{debug_image_prefix}_dropped_edge.png", dropped_edge)
            safe_imwrite(f"{debug_image_prefix}_dropped_band.png", dropped_band)
            safe_imwrite(f"{debug_image_prefix}_dropped_max_boxes.png", dropped_max_boxes)
            safe_imwrite(f"{debug_image_prefix}_dropped_all.png", dropped_all)
            safe_imwrite(f"{debug_image_prefix}_rescued_edge.png", rescued_edge)
            safe_imwrite(f"{debug_image_prefix}_rescued_band.png", rescued_band)
            safe_imwrite(f"{debug_image_prefix}_rescued_all.png", rescued_all)
        except Exception:
            pass
    return mask, bboxes_xywh
