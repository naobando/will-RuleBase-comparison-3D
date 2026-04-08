from __future__ import annotations
import os
import json
import time
import argparse
import cv2
import numpy as np
import yaml

from .config_loader import load_config
from . import pipeline as P

def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_raw_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _roi_rect_from_ratio(shape_hw, roi_ratio):
    h, w = shape_hw
    x0 = int(w * float(roi_ratio[0])); y0 = int(h * float(roi_ratio[1]))
    x1 = int(w * float(roi_ratio[2])); y1 = int(h * float(roi_ratio[3]))
    x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
    if x1 <= x0: x1 = min(w - 1, x0 + 1)
    if y1 <= y0: y1 = min(h - 1, y0 + 1)
    return x0, y0, x1, y1

def compute_roi_diff(raw_cfg: dict, ref_gray: np.ndarray, test_aligned: np.ndarray, ref_part_mask: np.ndarray):
    """
    ROI差分（学習なし）:
    - method=edge: Canny差分
    - method=abs: 画素差分
    - ref_part_maskで有効領域を限定し、ROI内の差分比率 + 最大ブロブ面積でNG判定
    """
    det = (raw_cfg.get("detection") or {})
    roi_cfg = (det.get("roi_diff") or {})
    enabled = bool(roi_cfg.get("enabled", False))
    if not enabled:
        return None

    method = str(roi_cfg.get("method", "edge")).lower()
    rois = roi_cfg.get("rois") or []
    if not rois:
        # ROIが無ければ全体を1つのROIとして扱う
        rois = [[0.0, 0.0, 1.0, 1.0]]

    canny1 = int(roi_cfg.get("canny1", 60))
    canny2 = int(roi_cfg.get("canny2", 160))
    dilate_ksize = int(roi_cfg.get("dilate_ksize", 5))
    dilate_iter = int(roi_cfg.get("dilate_iter", 1))
    diff_th = int(roi_cfg.get("diff_th", 25))
    min_area = int(roi_cfg.get("min_area", 120))
    sum_ratio_th = float(roi_cfg.get("sum_ratio_th", 0.0010))

    if method == "edge":
        ref_rep = cv2.Canny(ref_gray, canny1, canny2)
        tst_rep = cv2.Canny(test_aligned, canny1, canny2)
    else:
        ref_rep = ref_gray
        tst_rep = test_aligned

    diff = cv2.absdiff(ref_rep, tst_rep)

    if dilate_iter > 0 and dilate_ksize > 1:
        k = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        diff = cv2.dilate(diff, k, iterations=dilate_iter)

    _, diff_bin = cv2.threshold(diff, diff_th, 255, cv2.THRESH_BINARY)

    # 有効領域（ref_part_mask）で限定
    if ref_part_mask is not None:
        diff_bin = cv2.bitwise_and(diff_bin, diff_bin, mask=ref_part_mask)

    h, w = ref_gray.shape[:2]
    per_roi = []
    any_ng = False

    # ROI確認用の可視化ベース
    roi_vis = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)

    for roi in rois:
        x0, y0, x1, y1 = _roi_rect_from_ratio((h, w), roi)

        roi_bin = diff_bin[y0:y1+1, x0:x1+1]
        roi_mask = None
        if ref_part_mask is not None:
            roi_mask = ref_part_mask[y0:y1+1, x0:x1+1]
            valid = int(np.sum(roi_mask > 0))
        else:
            valid = int(roi_bin.size)

        diff_pixels = int(np.sum(roi_bin > 0))
        diff_ratio = float(diff_pixels / max(1, valid))

        # 最大ブロブ面積
        num, labels, stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8)
        max_blob = 0
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_blob = int(np.max(areas)) if len(areas) else 0

        ng = (diff_ratio >= sum_ratio_th) and (max_blob >= min_area)
        any_ng = any_ng or ng

        per_roi.append({
            "roi_ratio": roi,
            "rect": [int(x0), int(y0), int(x1), int(y1)],
            "valid_pixels": int(valid),
            "diff_pixels": int(diff_pixels),
            "diff_ratio": float(diff_ratio),
            "max_blob_area": int(max_blob),
            "ng": bool(ng),
        })

        # ROI枠描画（NGなら赤、OKなら緑）
        color = (0, 0, 255) if ng else (0, 255, 0)
        cv2.rectangle(roi_vis, (x0, y0), (x1, y1), color, 2)

    return {
        "enabled": True,
        "method": method,
        "params": {
            "canny1": canny1, "canny2": canny2,
            "dilate_ksize": dilate_ksize, "dilate_iter": dilate_iter,
            "diff_th": diff_th,
            "min_area": min_area,
            "sum_ratio_th": sum_ratio_th,
        },
        "per_roi": per_roi,
        "ng": bool(any_ng),
        "diff_bin": diff_bin,
        "roi_vis": roi_vis,
    }

def make_template(cfg_path: str, name: str, image_path: str) -> None:
    cfg = load_config(cfg_path)
    tpl_dir = os.path.join(cfg.io.templates_dir, name)
    P.ensure_dir(tpl_dir)

    bgr = P.imread_bgr(image_path)
    ref_crop, _, meta_crop = P.coarse_crop_ref_and_test(cfg, bgr, bgr)
    ref_gray = P.to_gray(ref_crop)

    if cfg.template.from_overlay:
        ref_outline = P.red_outline_from_overlay(cfg, ref_crop)
        ref_part_mask = np.ones(ref_gray.shape, dtype=np.uint8)*255
        ref_measure_mask = ref_part_mask.copy()
    else:
        pm = P.otsu_largest_mask(ref_gray, top_exclude_ratio=0.0)
        ref_measure_mask = pm.copy()
        if cfg.detection.part_mask.enabled:
            pm = P.erode_mask(pm, cfg.detection.part_mask.erode_ksize, cfg.detection.part_mask.erode_iter)
        ref_part_mask = pm
        ref_outline = P.outline_from_mask(ref_part_mask, ksize=3, iters=0)

    ref_measure = P.measure_scale(ref_measure_mask, cfg.alignment.scale_normalize.method)

    cv2.imwrite(os.path.join(tpl_dir, "ref_crop.png"), ref_crop)
    cv2.imwrite(os.path.join(tpl_dir, "ref_part_mask.png"), ref_part_mask)
    cv2.imwrite(os.path.join(tpl_dir, "ref_outline.png"), ref_outline)

    save_json(os.path.join(tpl_dir, "template_meta.json"), {
        "name": name,
        "source_image": image_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "coarse_crop": meta_crop,
        "scale_method": cfg.alignment.scale_normalize.method,
        "ref_measure": float(ref_measure),
        "template_from_overlay": bool(cfg.template.from_overlay),
    })

    print(json.dumps({"status":"ok","template":name,"dir":tpl_dir}, ensure_ascii=False, indent=2))

def inspect(cfg_path: str, template_name: str, image_path: str) -> None:
    cfg = load_config(cfg_path)
    raw_cfg = load_raw_yaml(cfg_path)

    tpl_dir = os.path.join(cfg.io.templates_dir, template_name)
    meta_path = os.path.join(tpl_dir, "template_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ref_crop = cv2.imread(os.path.join(tpl_dir, "ref_crop.png"), cv2.IMREAD_COLOR)
    ref_part_mask = cv2.imread(os.path.join(tpl_dir, "ref_part_mask.png"), cv2.IMREAD_GRAYSCALE)
    ref_outline = cv2.imread(os.path.join(tpl_dir, "ref_outline.png"), cv2.IMREAD_GRAYSCALE)

    test_bgr = P.imread_bgr(image_path)

    out_size = tuple(meta["coarse_crop"]["out_size"])
    bw, bh = meta["coarse_crop"]["ref_bbox_wh"]
    top_ex = meta["coarse_crop"]["top_exclude_ratio"]

    # center-stable coarse crop (pad handles out-of-bound)
    test_gray_full = P.to_gray(test_bgr)
    test_mask_full = P.otsu_largest_mask(test_gray_full, top_exclude_ratio=top_ex)
    tx0, ty0, tx1, ty1 = P.mask_bbox(test_mask_full)
    cx = (tx0 + tx1) // 2
    cy = (ty0 + ty1) // 2

    nx0 = cx - bw // 2
    ny0 = cy - bh // 2
    nx1 = nx0 + bw - 1
    ny1 = ny0 + bh - 1

    test_crop = P.crop_to_canvas(test_bgr, (nx0, ny0, nx1, ny1), out_size, pad_value=0)

    ref_gray = P.to_gray(ref_crop)
    test_gray = P.to_gray(test_crop)

    # scale normalize using stored ref_measure
    if cfg.alignment.scale_normalize.enabled:
        test_mask_c = P.otsu_largest_mask(test_gray, top_exclude_ratio=0.0)
        test_measure = P.measure_scale(test_mask_c, meta["scale_method"])
        if test_measure <= 1e-6:
            scale_factor = 1.0
            test_scaled = test_gray.copy()
        else:
            scale_factor = float(meta["ref_measure"] / test_measure)
            scale_factor = max(cfg.alignment.scale_normalize.min_scale, min(cfg.alignment.scale_normalize.max_scale, scale_factor))
            scaled = cv2.resize(test_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            h, w = ref_gray.shape
            canvas = np.full((h, w), cfg.alignment.scale_normalize.pad_value, dtype=np.uint8)
            sh, sw = scaled.shape
            yoff = (h - sh)//2
            xoff = (w - sw)//2
            if yoff >= 0 and xoff >= 0:
                canvas[yoff:yoff+sh, xoff:xoff+sw] = scaled
            else:
                ys = max(0, -yoff); xs = max(0, -xoff)
                ye = ys + min(h, sh); xe = xs + min(w, sw)
                canvas[:, :] = scaled[ys:ye, xs:xe]
            test_scaled = canvas
    else:
        scale_factor = 1.0
        test_scaled = test_gray.copy()

    if cfg.alignment.ecc.enabled:
        test_aligned, warp, ecc_cc = P.ecc_align(cfg, ref_gray, test_scaled, input_mask=ref_part_mask)
    else:
        test_aligned, warp, ecc_cc = test_scaled, np.eye(2,3, dtype=np.float32), 1.0

    # test mask & outline
    test_mask = P.otsu_largest_mask(test_aligned, top_exclude_ratio=0.0)
    if cfg.detection.part_mask.enabled:
        test_mask = P.erode_mask(test_mask, cfg.detection.part_mask.erode_ksize, cfg.detection.part_mask.erode_iter)

    # ref/test の「外周輪郭」を perimeter-edge で作る（同一パラメータで揃える）
    ref_outline2 = P.perimeter_edge_outline(
    ref_gray, ref_part_mask,
    canny1=30, canny2=100,
    band_ksize=31, band_iter=1,
    thicken_ksize=3, thicken_iter=1,
    )
    test_outline = P.perimeter_edge_outline(
        test_aligned, test_mask,
        canny1=30, canny2=100,
        band_ksize=31, band_iter=1,
        thicken_ksize=3, thicken_iter=1,
    )

    # 比較領域を ref_part_mask で限定（ref_outline2を維持することが重要）
    if cfg.detection.reference_mask_cutout.enabled:
        test_outline = cv2.bitwise_and(test_outline, test_outline, mask=ref_part_mask)
        ref_outline2 = cv2.bitwise_and(ref_outline2, ref_outline2, mask=ref_part_mask)

    else:
        ref_outline2 = ref_outline

    cd = cfg.detection.contour_diff
    result = {
        "template": template_name,
        "image": image_path,
        "scale_factor": float(scale_factor),
        "ecc_cc": float(ecc_cc),
        "warp": warp.tolist(),
    }

    miss_map = None
    extra_map = None

    # contour judge
    contour_judge = "UNKNOWN"
    contour_ng = False
    if cd.enabled and cd.method == "band_match":
        sc = P.band_match(ref_outline2, test_outline, cd.band_width)
        kernel = np.ones((cd.band_width, cd.band_width), np.uint8)
        ref_band = cv2.dilate(ref_outline2, kernel, iterations=1)
        test_band = cv2.dilate(test_outline, kernel, iterations=1)
        miss_map = ((test_outline>0) & (ref_band==0)).astype(np.uint8)*255
        extra_map = ((ref_outline2>0) & (test_band==0)).astype(np.uint8)*255
        contour_ng = (sc["miss_ratio"] >= cd.miss_ratio_th) or (sc["extra_ratio"] >= cd.extra_ratio_th)
        contour_judge = "NG" if contour_ng else "OK"
        result.update(sc)
    result["contour_judge"] = contour_judge

    # ROI diff judge (NG should override)
    roi_res = compute_roi_diff(raw_cfg, ref_gray, test_aligned, ref_part_mask)
    roi_ng = False
    if roi_res is not None:
        roi_ng = bool(roi_res.get("ng", False))
        # diff_bin / roi_vis are images; exclude from json
        diff_bin = roi_res.pop("diff_bin")
        roi_vis = roi_res.pop("roi_vis")
        result["roi_diff"] = roi_res
    else:
        diff_bin = None
        roi_vis = None

    # final judge
    final_ng = contour_ng or roi_ng
    result["judge"] = "NG" if final_ng else "OK"

    run_dir = os.path.join(cfg.io.outputs_dir, f"{template_name}_{now_id()}")
    P.ensure_dir(run_dir)

    if cfg.visualization.save_intermediate:
        cv2.imwrite(os.path.join(run_dir, "ref_crop.png"), ref_crop)
        cv2.imwrite(os.path.join(run_dir, "test_crop.png"), test_crop)
        cv2.imwrite(os.path.join(run_dir, "test_scaled.png"), test_scaled)
        cv2.imwrite(os.path.join(run_dir, "test_aligned.png"), test_aligned)
        cv2.imwrite(os.path.join(run_dir, "ref_part_mask.png"), ref_part_mask)
        cv2.imwrite(os.path.join(run_dir, "test_part_mask.png"), test_mask)
        cv2.imwrite(os.path.join(run_dir, "ref_outline.png"), ref_outline2)
        cv2.imwrite(os.path.join(run_dir, "test_outline.png"), test_outline)
        if miss_map is not None:
            cv2.imwrite(os.path.join(run_dir, "miss_map.png"), miss_map)
        if extra_map is not None:
            cv2.imwrite(os.path.join(run_dir, "extra_map.png"), extra_map)
        if diff_bin is not None:
            cv2.imwrite(os.path.join(run_dir, "roi_diff_bin.png"), diff_bin)
        if roi_vis is not None:
            cv2.imwrite(os.path.join(run_dir, "roi_boxes.png"), roi_vis)

    base_mode = getattr(cfg.visualization, "overlay_base", "ref")

    if base_mode == "test_aligned":
        base_gray_for_overlay = test_aligned
    elif base_mode == "test_crop":
        base_gray_for_overlay = test_gray
    else:
        base_gray_for_overlay = ref_gray

    # ---- overlay is drawn on INPUT (aligned) image, not on master ----
    # base = test_aligned (aligned input)
    overlay = P.draw_contour_overlay(
        test_aligned,              # ← ここを ref_gray から変更
        ref_outline2,
        test_outline,
        miss_map=miss_map,
        extra_map=extra_map,
        band_width=cfg.detection.contour_diff.band_width
    )

    # 保存名も分かりやすくしておく（必要なければ従来名でもOK）
    cv2.imwrite(os.path.join(run_dir, f"overlay_on_input_{result['judge']}.png"), overlay)

    save_json(os.path.join(run_dir, "result.json"), result)

    print(json.dumps({"status":"ok","outputs":run_dir, "result":result}, ensure_ascii=False, indent=2))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-template")
    a.add_argument("--config", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--image", required=True)
    b = sub.add_parser("inspect")
    b.add_argument("--config", required=True)
    b.add_argument("--template", required=True)
    b.add_argument("--image", required=True)
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "make-template":
        make_template(args.config, args.name, args.image)
    else:
        inspect(args.config, args.template, args.image)

if __name__ == "__main__":
    main()
