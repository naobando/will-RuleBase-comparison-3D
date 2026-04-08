from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .config_loader import AppConfig


def imread_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def otsu_largest_mask(gray: np.ndarray, top_exclude_ratio: float = 0.0) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = th.shape
    if top_exclude_ratio > 0:
        ycut = int(h * top_exclude_ratio)
        th[:ycut, :] = 0

    if th[h // 2, w // 2] == 0:
        th = cv2.bitwise_not(th)

    # stabilize boundary
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    # fill holes
    ff = th.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 255)
    ff_inv = cv2.bitwise_not(ff)
    th = cv2.bitwise_or(th, ff_inv)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num <= 1:
        return th

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8) * 255


def erode_mask(mask: np.ndarray, ksize: int, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(mask, kernel, iterations=iters)


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_with_pad(img: np.ndarray, rect: Tuple[int, int, int, int], out_size: Tuple[int, int], pad_value: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = rect
    h, w = img.shape[:2]

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - (w - 1))
    pad_bottom = max(0, y1 - (h - 1))

    if pad_left or pad_top or pad_right or pad_bottom:
        if img.ndim == 2:
            imgp = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(pad_value))
        else:
            imgp = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(pad_value),) * 3)
        x0p = x0 + pad_left
        x1p = x1 + pad_left
        y0p = y0 + pad_top
        y1p = y1 + pad_top
    else:
        imgp = img
        x0p, y0p, x1p, y1p = x0, y0, x1, y1

    crop = imgp[y0p:y1p + 1, x0p:x1p + 1]

    ow, oh = out_size
    if crop.ndim == 2:
        canvas = np.full((oh, ow), pad_value, dtype=crop.dtype)
    else:
        canvas = np.full((oh, ow, crop.shape[2]), pad_value, dtype=crop.dtype)

    ch, cw = crop.shape[:2]
    yoff = max(0, (oh - ch) // 2)
    xoff = max(0, (ow - cw) // 2)
    crop2 = crop

    if ch > oh:
        ys = (ch - oh) // 2
        crop2 = crop2[ys:ys + oh, :]
        ch = oh
        yoff = 0
    if cw > ow:
        xs = (cw - ow) // 2
        crop2 = crop2[:, xs:xs + ow]
        cw = ow
        xoff = 0

    canvas[yoff:yoff + ch, xoff:xoff + cw] = crop2
    return canvas


def crop_to_canvas(img: np.ndarray, rect: Tuple[int, int, int, int], out_size: Tuple[int, int], pad_value: int = 0) -> np.ndarray:
    return crop_with_pad(img, rect, out_size, pad_value=pad_value)


def coarse_crop_ref_and_test(cfg: AppConfig, ref_bgr: np.ndarray, test_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    out_size = cfg.alignment.coarse_crop.out_size
    top_ex = cfg.alignment.coarse_crop.top_exclude_ratio
    pad_ratio_x = float(getattr(cfg.alignment.coarse_crop, "pad_ratio", 0.18))
    pad_ratio_y = float(getattr(cfg.alignment.coarse_crop, "pad_ratio_y", pad_ratio_x))

    ref_gray = to_gray(ref_bgr)
    test_gray = to_gray(test_bgr)

    ref_mask = otsu_largest_mask(ref_gray, top_exclude_ratio=0.0)
    test_mask = otsu_largest_mask(test_gray, top_exclude_ratio=top_ex)

    rx0, ry0, rx1, ry1 = mask_bbox(ref_mask)
    tx0, ty0, tx1, ty1 = mask_bbox(test_mask)

    bw = rx1 - rx0 + 1
    bh = ry1 - ry0 + 1
    padx = int(bw * pad_ratio_x)
    pady = int(bh * pad_ratio_y)

    rx0 -= padx; rx1 += padx
    ry0 -= pady; ry1 += pady

    bw = rx1 - rx0 + 1
    bh = ry1 - ry0 + 1

    ref_crop = crop_with_pad(ref_bgr, (rx0, ry0, rx1, ry1), out_size, pad_value=0)

    cx = (tx0 + tx1) // 2
    cy = (ty0 + ty1) // 2
    nx0 = cx - bw // 2
    ny0 = cy - bh // 2
    nx1 = nx0 + bw - 1
    ny1 = ny0 + bh - 1

    test_crop = crop_with_pad(test_bgr, (nx0, ny0, nx1, ny1), out_size, pad_value=0)

    meta = {
        "ref_rect": [int(rx0), int(ry0), int(rx1), int(ry1)],
        "ref_bbox_wh": [int(bw), int(bh)],
        "test_rect": [int(nx0), int(ny0), int(nx1), int(ny1)],
        "out_size": [int(out_size[0]), int(out_size[1])],
        "top_exclude_ratio": float(top_ex),
        "pad_ratio": float(pad_ratio_x),
        "pad_ratio_y": float(pad_ratio_y),
    }
    return ref_crop, test_crop, meta


def measure_scale(mask: np.ndarray, method: str) -> float:
    if method == "area":
        return float(np.sum(mask > 0))
    x0, y0, x1, y1 = mask_bbox(mask)
    return float(y1 - y0 + 1)


def ecc_align(cfg: AppConfig, ref_gray: np.ndarray, test_gray: np.ndarray, input_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    scale = cfg.alignment.ecc.scale
    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    test_small = cv2.resize(test_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ref_f = ref_small.astype(np.float32) / 255.0
    test_f = test_small.astype(np.float32) / 255.0

    if cfg.alignment.ecc.motion == "translation":
        motion_type = cv2.MOTION_TRANSLATION
    elif cfg.alignment.ecc.motion == "euclidean":
        motion_type = cv2.MOTION_EUCLIDEAN
    else:
        motion_type = cv2.MOTION_AFFINE

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.alignment.ecc.num_iter, cfg.alignment.ecc.eps)

    # mask（ref座標系）を縮小してECCに渡す
    ecc_mask = None
    if input_mask is not None:
        m = (input_mask > 0).astype(np.uint8) * 255
        ecc_mask = cv2.resize(m, (ref_small.shape[1], ref_small.shape[0]), interpolation=cv2.INTER_NEAREST)

    cc, warp = cv2.findTransformECC(ref_f, test_f, warp, motion_type, criteria, ecc_mask, 1)

    warp_full = warp.copy()
    warp_full[0, 2] /= scale
    warp_full[1, 2] /= scale

    aligned = cv2.warpAffine(
        test_gray, warp_full, (ref_gray.shape[1], ref_gray.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    return aligned, warp_full, float(cc)


def outline_from_mask(mask: np.ndarray, ksize: int = 3, iters: int = 0) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, cnts, -1, 255, thickness=1)
    if iters > 0:
        out = cv2.dilate(out, np.ones((ksize, ksize), np.uint8), iterations=iters)
    return out


def red_outline_from_overlay(cfg: AppConfig, overlay_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2HSV)
    rr = cfg.template.overlay_red_hsv
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in rr.h_ranges:
        m = cv2.inRange(hsv, (int(lo), rr.s_min, rr.v_min), (int(hi), 255, 255))
        mask = cv2.bitwise_or(mask, m)
    k = rr.morph_close_ksize
    it = rr.morph_close_iter
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=it)
    return (mask > 0).astype(np.uint8) * 255


def band_match(ref_outline: np.ndarray, test_outline: np.ndarray, band_width: int) -> Dict[str, float]:
    kernel = np.ones((band_width, band_width), np.uint8)
    ref_band = cv2.dilate(ref_outline, kernel, iterations=1)
    test_band = cv2.dilate(test_outline, kernel, iterations=1)

    test_pix = int(np.sum(test_outline > 0))
    ref_pix = int(np.sum(ref_outline > 0))
    miss = int(np.sum((test_outline > 0) & (ref_band == 0)))
    extra = int(np.sum((ref_outline > 0) & (test_band == 0)))

    miss_ratio = float(miss / max(1, test_pix))
    extra_ratio = float(extra / max(1, ref_pix))
    return {
        "miss_ratio": miss_ratio, "extra_ratio": extra_ratio,
        "miss_pixels": miss, "extra_pixels": extra,
        "test_outline_pixels": test_pix, "ref_outline_pixels": ref_pix
    }

def draw_contour_overlay(
    base_gray: np.ndarray,
    ref_outline: np.ndarray,
    test_outline: np.ndarray,
    miss_map: Optional[np.ndarray]=None,
    extra_map: Optional[np.ndarray]=None,
    band_width: Optional[int]=None,
) -> np.ndarray:
    """
    表示ポリシー:
      - マスター輪郭(ref) = 青
      - 一致(ref & test) = 緑
      - ズレ(マスター側) extra_map = 黄
      - ズレ(検査側)   miss_map  = 赤
    """
    if base_gray.ndim != 2:
        raise ValueError("base_gray must be single-channel(gray)")

    bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    ref = (ref_outline > 0)
    tst = (test_outline > 0)
    match = ref & tst

    # master = 青
    bgr[ref] = (255, 0, 0)

    # match = 緑（上書き）
    bgr[match] = (0, 255, 0)

    # extra = 黄
    if extra_map is not None:
        bgr[extra_map > 0] = (0, 255, 255)

    # miss = 赤
    if miss_map is not None:
        bgr[miss_map > 0] = (0, 0, 255)

    return bgr

def perimeter_edge_outline(
    gray: np.ndarray,
    part_mask: np.ndarray,
    canny1: int = 60,
    canny2: int = 160,
    band_ksize: int = 31,
    band_iter: int = 1
) -> np.ndarray:
    """
    外周近傍（リング帯）に限定したCannyエッジを「外周輪郭」として返す。
    マスク輪郭が段差を吸収する問題を避ける。
    """
    e = cv2.Canny(gray, int(canny1), int(canny2))

    outline = outline_from_mask(part_mask, ksize=3, iters=0)  # 1px
    band = cv2.dilate(outline, np.ones((int(band_ksize), int(band_ksize)), np.uint8), iterations=int(band_iter))

    out = cv2.bitwise_and(e, e, mask=band)
    return (out > 0).astype(np.uint8) * 255

def perimeter_edge_outline(
    gray: np.ndarray,
    part_mask: np.ndarray,
    canny1: int = 30,
    canny2: int = 100,
    band_ksize: int = 31,
    band_iter: int = 1,
    blur_ksize: int = 5,
    thicken_ksize: int = 3,
    thicken_iter: int = 1,
) -> np.ndarray:
    """
    外周付近の帯(band)の中だけで Canny を取り、輪郭線を作る。
    - part_mask: 部品マスク（255=有効）
    - band_ksize/band_iter: 外周帯の幅を決める（dilate-erode）
    - canny: エッジ感度
    - thicken: 線が細すぎると band_match が破綻するので太らせる
    """
    if gray.ndim != 2:
        raise ValueError("gray must be single-channel")
    if part_mask is None:
        raise ValueError("part_mask is required")

    m = (part_mask > 0).astype(np.uint8) * 255

    k = np.ones((int(band_ksize), int(band_ksize)), np.uint8)
    dil = cv2.dilate(m, k, iterations=int(band_iter))
    ero = cv2.erode(m, k, iterations=int(band_iter))
    band = cv2.absdiff(dil, ero)  # 外周帯（リング）
    band = (band > 0).astype(np.uint8) * 255

    g = gray
    if blur_ksize and int(blur_ksize) >= 3:
        kk = int(blur_ksize)
        if kk % 2 == 0:
            kk += 1
        g = cv2.GaussianBlur(gray, (kk, kk), 0)

    edge = cv2.Canny(g, int(canny1), int(canny2))
    edge = cv2.bitwise_and(edge, edge, mask=band)

    # 線を太らせて密度差を減らす（重要）
    if int(thicken_iter) > 0 and int(thicken_ksize) > 1:
        kt = np.ones((int(thicken_ksize), int(thicken_ksize)), np.uint8)
        edge = cv2.dilate(edge, kt, iterations=int(thicken_iter))

    return (edge > 0).astype(np.uint8) * 255
