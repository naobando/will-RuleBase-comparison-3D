"""
ピンアート外形照合モジュール

3Dピンアートの比喩に基づく外形形状比較。
4辺から「ピン」を押し込み、物体に最初に当たった停止位置プロファイルを
マスターとテストで比較することでバリ・欠けなどの外周異常を検出する。

用語:
  プロファイル: 各行/列における最初の接触x/y座標の配列
  アンカー   : 最初に物体に接触した「帯域」の代表点（位置合わせ基準）
  接触帯     : 最初に接触した連続行/列の集合（ノイズ1点に依存しないため）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# データ構造
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PinProfile:
    """4方向ピン停止位置プロファイル（マスターまたはテスト）"""

    # 停止位置プロファイル（-1 = その行/列に物体なし）
    left:   np.ndarray  # shape (h,)  各行の左端x
    right:  np.ndarray  # shape (h,)  各行の右端x
    top:    np.ndarray  # shape (w,)  各列の上端y
    bottom: np.ndarray  # shape (w,)  各列の下端y

    # 有効マスク
    valid_rows: np.ndarray  # bool (h,)  物体が存在する行
    valid_cols: np.ndarray  # bool (w,)  物体が存在する列

    # アンカー: 各辺で最初に物体に接触した帯域の代表点 (x, y)
    # 接触帯が存在しない場合は None
    anchor_left:   tuple[int, int] | None = None
    anchor_right:  tuple[int, int] | None = None
    anchor_top:    tuple[int, int] | None = None
    anchor_bottom: tuple[int, int] | None = None

    # 元画像サイズ
    image_h: int = 0
    image_w: int = 0


@dataclass
class PinCompareResult:
    """ピンプロファイル比較結果"""

    # 各方向の差分配列（アンカー整列済み）
    # 正値 = テストが外に出っ張り（バリ候補）
    # 負値 = テストが内側に引っ込み（欠け候補）
    diff_left:   np.ndarray  # shape (h,)
    diff_right:  np.ndarray  # shape (h,)
    diff_top:    np.ndarray  # shape (w,)
    diff_bottom: np.ndarray  # shape (w,)

    # 有効マスク（マスターとテスト両方に物体がある行/列）
    valid_rows: np.ndarray
    valid_cols: np.ndarray

    # 整列に使ったオフセット (dx, dy)
    align_offset: tuple[int, int] = (0, 0)

    # スコアリング
    scores: dict = field(default_factory=dict)

    # 異常領域（バリ/欠けの行/列インデックス）
    burr_rows_left:   np.ndarray = field(default_factory=lambda: np.array([], int))
    burr_rows_right:  np.ndarray = field(default_factory=lambda: np.array([], int))
    burr_cols_top:    np.ndarray = field(default_factory=lambda: np.array([], int))
    burr_cols_bottom: np.ndarray = field(default_factory=lambda: np.array([], int))
    chip_rows_left:   np.ndarray = field(default_factory=lambda: np.array([], int))
    chip_rows_right:  np.ndarray = field(default_factory=lambda: np.array([], int))
    chip_cols_top:    np.ndarray = field(default_factory=lambda: np.array([], int))
    chip_cols_bottom: np.ndarray = field(default_factory=lambda: np.array([], int))


# ──────────────────────────────────────────────────────────────────────────────
# プロファイル抽出
# ──────────────────────────────────────────────────────────────────────────────

def extract_pin_profile(
    binary_mask: np.ndarray,
    noise_erode: int = 1,
    anchor_band_min_length: int = 5,
) -> PinProfile:
    """
    二値マスクから4方向ピン停止位置プロファイルを抽出する。

    Parameters
    ----------
    binary_mask : np.ndarray
        物体領域が255（前景）の二値マスク。uint8。
    noise_erode : int
        プロファイル計算前の収縮量（小ノイズ除去）。0で無効。
    anchor_band_min_length : int
        アンカー帯域として認める最小連続行/列数。

    Returns
    -------
    PinProfile
    """
    mask = binary_mask.copy()
    if noise_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * noise_erode + 1, 2 * noise_erode + 1))
        mask = cv2.erode(mask, k, iterations=1)

    h, w = mask.shape
    bool_mask = mask > 0  # (h, w) bool

    valid_rows = bool_mask.any(axis=1)  # (h,)
    valid_cols = bool_mask.any(axis=0)  # (w,)

    # ── 各方向の停止位置 ─────────────────────────────────────────────────────

    # 左から: 各行で最初にTrueになる列インデックス
    left = np.where(valid_rows,
                    np.argmax(bool_mask, axis=1),
                    -1).astype(np.int32)

    # 右から: 反転してargmax
    right = np.where(valid_rows,
                     w - 1 - np.argmax(bool_mask[:, ::-1], axis=1),
                     -1).astype(np.int32)

    # 上から: 各列で最初にTrueになる行インデックス
    top = np.where(valid_cols,
                   np.argmax(bool_mask, axis=0),
                   -1).astype(np.int32)

    # 下から: 反転してargmax
    bottom = np.where(valid_cols,
                      h - 1 - np.argmax(bool_mask[::-1, :], axis=0),
                      -1).astype(np.int32)

    # ── アンカー計算 ─────────────────────────────────────────────────────────
    anchor_left   = _compute_anchor_horizontal(left,   valid_rows, anchor_band_min_length, side="left")
    anchor_right  = _compute_anchor_horizontal(right,  valid_rows, anchor_band_min_length, side="right")
    anchor_top    = _compute_anchor_vertical(top,      valid_cols, anchor_band_min_length, side="top")
    anchor_bottom = _compute_anchor_vertical(bottom,   valid_cols, anchor_band_min_length, side="bottom")

    return PinProfile(
        left=left, right=right, top=top, bottom=bottom,
        valid_rows=valid_rows, valid_cols=valid_cols,
        anchor_left=anchor_left, anchor_right=anchor_right,
        anchor_top=anchor_top, anchor_bottom=anchor_bottom,
        image_h=h, image_w=w,
    )


def _compute_anchor_horizontal(
    profile: np.ndarray,
    valid: np.ndarray,
    min_length: int,
    side: str,
) -> tuple[int, int] | None:
    """
    左/右方向プロファイルのアンカーを計算する。

    「最も外側に出ている接触帯」の代表点を返す。
    左辺なら最小x（最も左に出た帯）、右辺なら最大x（最も右に出た帯）の
    連続領域中心。
    """
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < min_length:
        return None

    vals = profile[valid_idx]
    if side == "left":
        extreme = np.min(vals)
        # extreme ±2px 以内の行を候補に
        cand = valid_idx[np.abs(vals - extreme) <= 2]
    else:
        extreme = np.max(vals)
        cand = valid_idx[np.abs(vals - extreme) <= 2]

    if len(cand) == 0:
        return None

    # 最長連続帯を探す
    band = _longest_consecutive(cand)
    if len(band) < min_length:
        band = cand  # 連続でなくても全候補使う

    cx = int(round(np.median(profile[band])))
    cy = int(round(np.median(band)))
    return (cx, cy)


def _compute_anchor_vertical(
    profile: np.ndarray,
    valid: np.ndarray,
    min_length: int,
    side: str,
) -> tuple[int, int] | None:
    """上/下方向プロファイルのアンカーを計算する。"""
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < min_length:
        return None

    vals = profile[valid_idx]
    if side == "top":
        extreme = np.min(vals)
        cand = valid_idx[np.abs(vals - extreme) <= 2]
    else:
        extreme = np.max(vals)
        cand = valid_idx[np.abs(vals - extreme) <= 2]

    if len(cand) == 0:
        return None

    band = _longest_consecutive(cand)
    if len(band) < min_length:
        band = cand

    cx = int(round(np.median(band)))
    cy = int(round(np.median(profile[band])))
    return (cx, cy)


def _longest_consecutive(indices: np.ndarray) -> np.ndarray:
    """インデックス配列から最長の連続区間を返す。"""
    if len(indices) == 0:
        return indices
    sorted_idx = np.sort(indices)
    gaps = np.diff(sorted_idx)
    breaks = np.where(gaps > 1)[0] + 1
    segments = np.split(sorted_idx, breaks)
    return max(segments, key=len)


# ──────────────────────────────────────────────────────────────────────────────
# 比較・スコアリング
# ──────────────────────────────────────────────────────────────────────────────

def compare_pin_profiles(
    master: PinProfile,
    test: PinProfile,
    burr_threshold: int = 3,
    chip_threshold: int = 3,
    use_anchor_align: bool = True,
) -> PinCompareResult:
    """
    マスターとテストのピンプロファイルを比較する。

    Parameters
    ----------
    master, test : PinProfile
        extract_pin_profile() の結果。
    burr_threshold : int
        バリ（外側への突出）と判定する差分px閾値。
    chip_threshold : int
        欠け（内側への引っ込み）と判定する差分px閾値。
    use_anchor_align : bool
        Trueならアンカーを使ってテストをマスターに位置合わせしてから比較。

    Returns
    -------
    PinCompareResult
    """
    dx, dy = 0, 0
    if use_anchor_align:
        dx, dy = _estimate_offset(master, test)

    # 有効マスクの共通部分（両方に物体がある行/列）
    h = min(master.image_h, test.image_h)
    w = min(master.image_w, test.image_w)

    valid_rows = master.valid_rows[:h] & test.valid_rows[:h]
    valid_cols = master.valid_cols[:w] & test.valid_cols[:w]

    # ── 各方向の差分（アンカーオフセット補正後）─────────────────────────────
    # 左辺: 正 = テストが外（左）に出ている = バリ候補
    diff_left = np.full(h, 0, dtype=np.int32)
    mask_r = valid_rows
    diff_left[mask_r] = (master.left[:h][mask_r]
                         - (test.left[:h][mask_r] - dx))

    # 右辺: 正 = テストが外（右）に出ている = バリ候補
    diff_right = np.full(h, 0, dtype=np.int32)
    diff_right[mask_r] = ((test.right[:h][mask_r] - dx)
                          - master.right[:h][mask_r])

    # 上辺: 正 = テストが外（上）に出ている = バリ候補
    diff_top = np.full(w, 0, dtype=np.int32)
    mask_c = valid_cols
    diff_top[mask_c] = (master.top[:w][mask_c]
                        - (test.top[:w][mask_c] - dy))

    # 下辺: 正 = テストが外（下）に出ている = バリ候補
    diff_bottom = np.full(w, 0, dtype=np.int32)
    diff_bottom[mask_c] = ((test.bottom[:w][mask_c] - dy)
                           - master.bottom[:w][mask_c])

    # ── 異常箇所検出 ─────────────────────────────────────────────────────────
    burr_rows_left   = np.where(mask_r & (diff_left   >  burr_threshold))[0]
    chip_rows_left   = np.where(mask_r & (diff_left   < -chip_threshold))[0]
    burr_rows_right  = np.where(mask_r & (diff_right  >  burr_threshold))[0]
    chip_rows_right  = np.where(mask_r & (diff_right  < -chip_threshold))[0]
    burr_cols_top    = np.where(mask_c & (diff_top    >  burr_threshold))[0]
    chip_cols_top    = np.where(mask_c & (diff_top    < -chip_threshold))[0]
    burr_cols_bottom = np.where(mask_c & (diff_bottom >  burr_threshold))[0]
    chip_cols_bottom = np.where(mask_c & (diff_bottom < -chip_threshold))[0]

    # ── スコアリング ──────────────────────────────────────────────────────────
    scores = _score(
        diff_left, diff_right, diff_top, diff_bottom,
        valid_rows, valid_cols,
        burr_threshold, chip_threshold,
    )

    return PinCompareResult(
        diff_left=diff_left, diff_right=diff_right,
        diff_top=diff_top,   diff_bottom=diff_bottom,
        valid_rows=valid_rows, valid_cols=valid_cols,
        align_offset=(dx, dy),
        scores=scores,
        burr_rows_left=burr_rows_left,   chip_rows_left=chip_rows_left,
        burr_rows_right=burr_rows_right, chip_rows_right=chip_rows_right,
        burr_cols_top=burr_cols_top,     chip_cols_top=chip_cols_top,
        burr_cols_bottom=burr_cols_bottom, chip_cols_bottom=chip_cols_bottom,
    )


def _estimate_offset(master: PinProfile, test: PinProfile) -> tuple[int, int]:
    """
    アンカー点を使ってテスト→マスターへの平行移動オフセット (dx, dy) を推定。

    複数のアンカーが存在する場合は中央値を使う。
    """
    dxs, dys = [], []

    pairs = [
        (master.anchor_left,   test.anchor_left),
        (master.anchor_right,  test.anchor_right),
        (master.anchor_top,    test.anchor_top),
        (master.anchor_bottom, test.anchor_bottom),
    ]
    for m_anc, t_anc in pairs:
        if m_anc is not None and t_anc is not None:
            dxs.append(m_anc[0] - t_anc[0])
            dys.append(m_anc[1] - t_anc[1])

    if not dxs:
        return 0, 0
    return int(round(np.median(dxs))), int(round(np.median(dys)))


def _score(
    diff_left, diff_right, diff_top, diff_bottom,
    valid_rows, valid_cols,
    burr_thr, chip_thr,
) -> dict:
    """一致率・最大偏差・連続異常長などをスコア化する。"""
    all_diffs = np.concatenate([
        diff_left[valid_rows], diff_right[valid_rows],
        diff_top[valid_cols],  diff_bottom[valid_cols],
    ])
    if len(all_diffs) == 0:
        return {"match_rate": 0.0, "max_deviation": 0, "mean_deviation": 0.0,
                "max_consecutive_anomaly": 0}

    thr = max(burr_thr, chip_thr)
    within = np.abs(all_diffs) <= thr
    match_rate = float(np.mean(within))
    max_dev = int(np.max(np.abs(all_diffs)))
    mean_dev = float(np.mean(np.abs(all_diffs)))

    # 最長連続異常（どの辺でも）
    anomaly = np.abs(all_diffs) > thr
    max_consec = _max_consecutive_true(anomaly)

    return {
        "match_rate":              match_rate,   # 0〜1 (高いほど良い)
        "max_deviation":           max_dev,       # px
        "mean_deviation":          mean_dev,      # px
        "max_consecutive_anomaly": max_consec,    # 行/列数
    }


def _max_consecutive_true(arr: np.ndarray) -> int:
    """bool配列中の最長連続Trueの長さ。"""
    if not arr.any():
        return 0
    max_len = cur = 0
    for v in arr:
        if v:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


# ──────────────────────────────────────────────────────────────────────────────
# 可視化
# ──────────────────────────────────────────────────────────────────────────────

def visualize_pin_compare(
    image: np.ndarray,
    master: PinProfile,
    result: PinCompareResult,
    burr_color:  tuple[int, int, int] = (0, 0, 255),    # 赤: バリ
    chip_color:  tuple[int, int, int] = (255, 165, 0),  # オレンジ: 欠け
    master_color: tuple[int, int, int] = (0, 255, 0),   # 緑: マスター輪郭
    alpha: float = 0.5,
) -> np.ndarray:
    """
    比較結果を元画像上に重ねて可視化する。

    - 緑線: マスターの外形プロファイル（期待される輪郭）
    - 赤領域: バリ（外側へ突出）
    - オレンジ領域: 欠け（内側へ引っ込み）
    """
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    h, w = vis.shape[:2]
    dx, dy = result.align_offset

    # ── マスター輪郭を緑線で描画 ────────────────────────────────────────────
    for y in np.where(master.valid_rows[:h])[0]:
        xl = min(max(int(master.left[y]),  0), w - 1)
        xr = min(max(int(master.right[y]), 0), w - 1)
        cv2.circle(overlay, (xl, y), 1, master_color, -1)
        cv2.circle(overlay, (xr, y), 1, master_color, -1)
    for x in np.where(master.valid_cols[:w])[0]:
        yt = min(max(int(master.top[x]),    0), h - 1)
        yb = min(max(int(master.bottom[x]), 0), h - 1)
        cv2.circle(overlay, (x, yt), 1, master_color, -1)
        cv2.circle(overlay, (x, yb), 1, master_color, -1)

    # ── バリ領域（赤）────────────────────────────────────────────────────────
    for y in result.burr_rows_left:
        xl = min(max(int(master.left[y]) + dx, 0), w - 1)
        cv2.line(overlay, (0, y), (xl, y), burr_color, 2)
    for y in result.burr_rows_right:
        xr = min(max(int(master.right[y]) + dx, 0), w - 1)
        cv2.line(overlay, (xr, y), (w - 1, y), burr_color, 2)
    for x in result.burr_cols_top:
        yt = min(max(int(master.top[x]) + dy, 0), h - 1)
        cv2.line(overlay, (x, 0), (x, yt), burr_color, 2)
    for x in result.burr_cols_bottom:
        yb = min(max(int(master.bottom[x]) + dy, 0), h - 1)
        cv2.line(overlay, (x, yb), (x, h - 1), burr_color, 2)

    # ── 欠け領域（オレンジ）──────────────────────────────────────────────────
    for y in result.chip_rows_left:
        xl_m = min(max(int(master.left[y]),          0), w - 1)
        xl_t = min(max(int(master.left[y]) + dx + result.diff_left[y], 0), w - 1)
        if xl_m != xl_t:
            cv2.line(overlay, (min(xl_m, xl_t), y), (max(xl_m, xl_t), y), chip_color, 2)
    for y in result.chip_rows_right:
        xr_m = min(max(int(master.right[y]),          0), w - 1)
        xr_t = min(max(int(master.right[y]) + dx - result.diff_right[y], 0), w - 1)
        if xr_m != xr_t:
            cv2.line(overlay, (min(xr_m, xr_t), y), (max(xr_m, xr_t), y), chip_color, 2)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis


# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ: バイナリマスク生成
# ──────────────────────────────────────────────────────────────────────────────

def image_to_binary_mask(
    image: np.ndarray,
    method: str = "otsu",
    invert: bool | None = None,
) -> np.ndarray:
    """
    BGR/グレー画像から二値マスクを生成する。

    Parameters
    ----------
    image : np.ndarray
        入力画像（BGR or グレースケール）。
    method : str
        "otsu" (大津法) または "fg_mask" (既存fg_maskをそのまま使う場合はそちらを直接渡す)。
    invert : bool | None
        Noneなら自動（白ピクセル多数なら反転）。

    Returns
    -------
    np.ndarray
        uint8 二値マスク（物体=255, 背景=0）。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert is None:
        invert = np.mean(binary) > 127
    if invert:
        binary = cv2.bitwise_not(binary)

    # 小ノイズ除去
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    return binary
