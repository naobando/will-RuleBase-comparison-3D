"""
エッジ差分分離モジュール

前景マスク境界付近のずれ差分と内部の本当の差分を、
距離マップを使って分離する機能を提供します。
"""
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def separate_edge_from_interior(
    edge_drop_mask,
    fg_mask,
    diff_gray,
    min_dist=3,
    max_search_dist=30,
    search_step=1,
    min_interior_area=100,
    connectivity=8,
):
    """
    エッジに繋がった差分成分から、内部の本当の差分だけを分離して救済する。

    前景マスクの境界からの距離マップを使い、境界に近いピクセル（エッジずれ由来）
    と内部ピクセル（本当の差分）を切り分ける。

    仕組み:
      1) 前景マスクの境界からの距離マップを計算（distanceTransform）
      2) edge_drop_mask 内のピクセルを、距離閾値で段階的にフィルタリング
      3) 距離が小さい（境界付近）ピクセルを除外していき、
         途切れて独立した内部成分が出現したら → 分離成功
      4) 画像端に接触せず、面積が十分な成分を内部差分として返す

    Args:
        edge_drop_mask: エッジ接触で除外された成分のマスク (uint8, 0/255)
        fg_mask: 前景マスク (uint8, 0/255, 255=前景)
        diff_gray: 差分画像 (uint8, 0-255)
        min_dist: 内部とみなす最小境界距離 (px)
        max_search_dist: 距離探索の上限 (px)
        search_step: 距離探索のステップ (px)
        min_interior_area: 内部成分として認める最小面積 (px)
        connectivity: 連結成分の接続性 (4 or 8)

    Returns:
        interior_mask: 内部成分のマスク (uint8, 0/255)。分離できない場合はNone
        info: 診断情報の辞書
    """
    info = {
        "method": "distance_separation",
        "separated": False,
        "separation_dist": None,
        "interior_components": 0,
        "interior_total_area": 0,
    }

    if edge_drop_mask is None or fg_mask is None:
        return None, info

    if np.count_nonzero(edge_drop_mask) == 0:
        return None, info

    # 前景マスクの境界からの距離マップを計算
    # distanceTransform は前景（白）ピクセルの最寄りの背景（黒）からの距離を返す
    fg_dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

    h_img, w_img = edge_drop_mask.shape[:2]

    # 距離閾値を段階的に上げながら、内部成分が分離されるか探索
    best_interior = None
    best_dist = None
    best_components = 0
    best_area = 0

    for d in range(min_dist, max_search_dist + 1, search_step):
        # 境界からの距離が d 以上かつ edge_drop_mask 内のピクセルだけ残す
        dist_filter = (fg_dist >= float(d)).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(edge_drop_mask, dist_filter)

        if np.count_nonzero(candidate) == 0:
            # 全ピクセルがフィルタされた → これ以上探索しても無駄
            break

        # 連結成分を抽出
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=int(connectivity)
        )

        # 画像端に接触しない、面積が十分な成分を収集
        interior = np.zeros_like(candidate)
        n_interior = 0
        total_area = 0

        for lbl in range(1, num_labels):
            x, y, bw, bh, area = stats[lbl]
            touches_edge = (
                x == 0 or y == 0
                or (x + bw) >= w_img
                or (y + bh) >= h_img
            )
            if touches_edge:
                continue
            if area < min_interior_area:
                continue
            interior[labels == lbl] = 255
            n_interior += 1
            total_area += area

        if n_interior > 0:
            # 内部成分が見つかった
            if best_interior is None or total_area > best_area:
                best_interior = interior.copy()
                best_dist = d
                best_components = n_interior
                best_area = total_area
            # 最初に分離できた距離を採用（最小距離で分離 = 最大の内部領域を保持）
            break

    if best_interior is not None and np.count_nonzero(best_interior) > 0:
        info["separated"] = True
        info["separation_dist"] = best_dist
        info["interior_components"] = best_components
        info["interior_total_area"] = best_area
        logger.info(
            f"エッジ差分分離成功: dist={best_dist}px, "
            f"内部成分={best_components}個, 面積={best_area}px"
        )
        return best_interior, info

    logger.info("エッジ差分分離: 内部成分が見つかりませんでした")
    return None, info
