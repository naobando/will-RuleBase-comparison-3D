"""解析処理をバックグラウンドスレッドで実行"""
from __future__ import annotations

import io
import os
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.pipeline.symmetry import compare_images
from src.core.auto_thresh import calculate_auto_diff_thresh
from src.core.calibration import calibrate_master
from src.core.segmentation import get_foreground_mask


class AnalysisWorker(QThread):
    """compare_images をバックグラウンドで実行し、結果をシグナルで返す"""

    result_ready = Signal(object)       # tuple: (mse, ssim, diff, mask, bboxes, fig_bytes, ...)
    error_occurred = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        master_frame: np.ndarray,
        test_frame: np.ndarray,
        params: dict,
        config: dict,
        calibration_result: dict | None,
        parent=None,
    ):
        super().__init__(parent)
        self.master_frame = master_frame
        self.test_frame = test_frame
        self.params = params
        self.config = config
        self.calibration_result = calibration_result

    def run(self) -> None:
        try:
            self.progress.emit("解析中...")
            result = self._run_analysis()
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _run_analysis(self) -> tuple:
        cfg = self.config["analysis"]
        master = self.master_frame
        test = self.test_frame
        params = dict(self.params)  # shallow copy

        diff_thresh = params["diff_thresh"]
        min_area = params["min_area"]
        morph_kernel = params["morph_kernel"]

        # 自動計算
        if cfg.get("auto_diff_thresh"):
            diff_thresh, _ = calculate_auto_diff_thresh(
                master, test,
                method=cfg.get("diff_thresh_method", "hybrid"),
                percentile=cfg.get("diff_thresh_percentile", 98),
                contrast_low=cfg.get("diff_thresh_contrast_low", 8),
                contrast_mid=cfg.get("diff_thresh_contrast_mid", 14),
                contrast_high=cfg.get("diff_thresh_contrast_high", 20),
                thresh_min=cfg.get("diff_thresh_min", 13),
                thresh_max=cfg.get("diff_thresh_max", 51),
            )
            params["diff_thresh"] = diff_thresh

        if cfg.get("auto_min_area"):
            h, w = master.shape[:2]
            min_area = int(h * w * cfg.get("min_area_ratio", 0.008))
            params["min_area"] = min_area

        if cfg.get("auto_morph_kernel"):
            h, w = master.shape[:2]
            area = h * w
            if area < 200_000:
                coeff = cfg.get("morph_kernel_coefficient_small", 60)
            elif area < 800_000:
                coeff = cfg.get("morph_kernel_coefficient_medium", 65)
            else:
                coeff = cfg.get("morph_kernel_coefficient_large", 95)
            mk = max(3, int((area ** 0.5) / coeff))
            if mk % 2 == 0:
                mk += 1
            params["morph_kernel"] = mk

        # pipeline_params を config["analysis"] をベースに構築
        pipeline_params = dict(cfg)

        _key_renames = {
            "foreground_mask": "use_foreground_mask",
            "background_overlap_check": "use_background_overlap_check",
            "crop_to_master_fov": "use_crop_to_master_fov",
            "flip_compare": "use_flip_compare",
            "roi_enabled": "use_roi",
        }
        for old_key, new_key in _key_renames.items():
            if old_key in pipeline_params:
                pipeline_params[new_key] = pipeline_params.pop(old_key)

        base_dir = self.config["output"]["base_dir"]
        if pipeline_params.get("debug_crop_log"):
            pipeline_params["debug_crop_log_path"] = os.path.join(
                base_dir, pipeline_params.get("debug_crop_log_path", "crop_debug.log"))
        if pipeline_params.get("debug_crop_save_images"):
            pipeline_params["debug_crop_save_images_dir"] = os.path.join(
                base_dir, pipeline_params.get("debug_crop_save_images_dir", "crop_debug_images"))
        if pipeline_params.get("debug_save_pipeline_stages"):
            pipeline_params["debug_save_pipeline_dir"] = os.path.join(
                base_dir, pipeline_params.get("debug_save_pipeline_dir", "debug_pipeline"))

        pipeline_params.update({
            "title": "板金比較: 基準 vs 比較",
            "show_plot": False,
        })
        # crop_fixed_params はconfigトップレベルにあるので注入
        _crop_fixed = self.config.get("crop_fixed_params")
        if _crop_fixed:
            pipeline_params["crop_fixed_params"] = _crop_fixed
        pipeline_params.update(params)

        # キャリブレーション結果を注入
        cal = self.calibration_result
        if cal and cal.get("calibrated"):
            pipeline_params["ensemble_thresh"] = cal["ensemble_thresh"]
            pipeline_params["ensemble_bbox_min_mean"] = cal["ensemble_bbox_min_mean"]

        # ── ピンアート型取りモード ──────────────────────────────────
        crop_method = params.get("crop_method", "conventional")
        if crop_method == "pin_art":
            raw = self._run_pin_art_analysis(
                master, test, params, pipeline_params,
            )
        else:
            raw = compare_images(master, test, **pipeline_params)

        # 結果を13要素タプルに正規化
        mse = raw[0]
        ssim_score = raw[1]
        diff = raw[2]
        mask = raw[3]
        bboxes = raw[4]
        fig = raw[5]
        align_info = raw[6] if len(raw) > 6 else None
        crop_info = raw[7] if len(raw) > 7 else None
        roi_info = raw[8] if len(raw) > 8 else None
        flip_info = raw[9] if len(raw) > 9 else None
        quality_info = raw[10] if len(raw) > 10 else None
        structural_info = raw[11] if len(raw) > 11 else None
        scratch_info = raw[12] if len(raw) > 12 else None
        pin_info = raw[14] if len(raw) > 14 else None

        # Matplotlib figure → PNG bytes（全パネル・デバッグ用）
        fig_buffer = io.BytesIO()
        fig.savefig(fig_buffer, format="png", dpi=150, bbox_inches="tight", pad_inches=0.8)
        fig_buffer.seek(0)
        fig_bytes = fig_buffer.getvalue()

        # 2番目のパネル（比較画像+BBOX）をピクセルデータとして抽出（ユーザーモード用）
        # matplotlib の白背景を含まない純粋な cv2 BGR 画像として取り出す
        visB_img = None
        try:
            if len(fig.axes) >= 2:
                ax2 = fig.axes[1]
                imgs = ax2.get_images()
                if imgs:
                    arr = imgs[0].get_array()
                    if arr is not None and arr.ndim == 3:
                        visB_img = cv2.cvtColor(
                            np.array(arr, dtype=np.uint8), cv2.COLOR_RGB2BGR
                        )
        except Exception:
            pass  # フォールバックは user_window 側で fig_bytes を使う

        return (
            mse, ssim_score, diff, mask, bboxes, fig_bytes,
            align_info, crop_info, roi_info,
            flip_info, quality_info, structural_info, scratch_info,
            params["diff_thresh"], params["min_area"],
            params.get("morph_kernel", pipeline_params.get("morph_kernel")),
            params.get("max_boxes", pipeline_params.get("max_boxes")),
            visB_img,   # index 17: 比較画像+BBOX の cv2 BGR 画像（ユーザーモード用）
            pin_info,   # index 18: ピンアート照合結果 dict or None
        )


    # ── ピンアート専用パイプライン ────────────────────────────────

    def _run_pin_art_analysis(
        self,
        master: np.ndarray,
        test: np.ndarray,
        params: dict,
        pipeline_params: dict,
    ) -> tuple:
        """
        ピンアート方式の解析:
        1. 従来パイプラインで位置合わせ（背景付きのまま）
        2. crop_info から位置合わせ済み画像ペアを取得
        3. 位置合わせ済みマスターからピンアートマスクを生成
        4. マスク内ピクセルだけで SSIM / MSE を算出
        """
        from skimage.metrics import structural_similarity as skimage_ssim
        from src.core.pin_profile import image_to_binary_mask

        # Step 1: 従来パイプラインで位置合わせ + BBOX 等を取得
        raw = compare_images(master, test, **pipeline_params)

        # Step 2: 位置合わせ済み画像ペアを取得
        # crop_info['preview_pass1'] = (aligned_master, aligned_test, scale, ...)
        crop_info = raw[7] if len(raw) > 7 else None
        aligned_master = None
        aligned_test = None

        if crop_info and crop_info.get("adopted"):
            for key in ("preview_pass2", "preview_pass1"):
                preview = crop_info.get(key)
                if preview is not None:
                    aligned_master = preview[0]
                    aligned_test = preview[1]
                    break

        # フォールバック: crop_info がない場合は diff サイズに合わせる
        if aligned_master is None or aligned_test is None:
            diff_img = raw[2]
            if diff_img is not None and diff_img.size > 0:
                th, tw = diff_img.shape[:2]
                aligned_master = cv2.resize(master, (tw, th))
                aligned_test = cv2.resize(test, (tw, th))
            else:
                return raw  # フォールバック不能

        # Step 3: 位置合わせ済みマスターからピンアートマスクを生成
        master_binary = image_to_binary_mask(aligned_master)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        master_binary = cv2.morphologyEx(
            master_binary, cv2.MORPH_CLOSE, kernel, iterations=2,
        )
        master_binary = cv2.morphologyEx(
            master_binary, cv2.MORPH_OPEN, kernel, iterations=1,
        )

        # Step 4: マスク内ピクセルだけで SSIM / MSE を再計算
        gray_a = cv2.cvtColor(aligned_master, cv2.COLOR_BGR2GRAY) if aligned_master.ndim == 3 else aligned_master
        gray_b = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY) if aligned_test.ndim == 3 else aligned_test

        mask_bool = master_binary > 0
        n_pixels = int(np.sum(mask_bool))

        if n_pixels > 0:
            pixels_a = gray_a[mask_bool].astype(np.float64)
            pixels_b = gray_b[mask_bool].astype(np.float64)
            masked_mse = float(np.mean((pixels_a - pixels_b) ** 2))

            _, ssim_map = skimage_ssim(
                gray_a, gray_b, full=True, data_range=255.0,
            )
            masked_ssim = float(np.mean(ssim_map[mask_bool]))
        else:
            masked_mse = raw[0]
            masked_ssim = raw[1]

        # Step 5: MSE/SSIM だけ差し替え、他はパイプライン結果のまま
        result = list(raw)
        result[0] = masked_mse
        result[1] = masked_ssim
        return tuple(result)


class CalibrationWorker(QThread):
    """マスター自動キャリブレーションをバックグラウンドで実行"""

    result_ready = Signal(object)   # dict
    error_occurred = Signal(str)

    def __init__(
        self,
        master_bgr: np.ndarray,
        config: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.master_bgr = master_bgr
        self.config = config

    def run(self) -> None:
        try:
            cfg = self.config["analysis"]
            master = self.master_bgr

            fg_mask = None
            if cfg.get("foreground_mask", False):
                fg_mask = get_foreground_mask(
                    master,
                    keep_largest_ratio=float(cfg.get("foreground_mask_keep_ratio", 0.0)),
                    preclose_kernel=int(cfg.get("foreground_mask_kernel", 15)),
                )
                if fg_mask is not None:
                    dilate_iter = int(cfg.get("foreground_mask_dilate_iter", 0))
                    if dilate_iter > 0:
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        fg_mask = cv2.dilate(fg_mask, k, iterations=dilate_iter)

            result = calibrate_master(
                master,
                preprocess_mode=cfg.get("preprocess_mode", "luminance"),
                preprocess_blur_ksize=int(cfg.get("preprocess_blur_ksize", 3)),
                ensemble_2scale_min_side=int(cfg.get("ensemble_2scale_min_side", 250)),
                ensemble_edge_suppress_ratio=float(cfg.get("ensemble_edge_suppress_ratio", 0.04)),
                ensemble_edge_suppress_max=int(cfg.get("ensemble_edge_suppress_max", 8)),
                ensemble_bg_brightness_max=int(cfg.get("ensemble_bg_brightness_max", 30)),
                fg_mask=fg_mask,
                config_fallback_thresh=int(cfg.get("ensemble_thresh", 70)),
                config_fallback_min_mean=int(cfg.get("ensemble_bbox_min_mean", 60)),
                calibration_margin_ratio=float(cfg.get("calibration_margin_ratio", 0.3)),
                calibration_min_thresh=int(cfg.get("calibration_min_thresh", 30)),
                calibration_max_thresh=int(cfg.get("calibration_max_thresh", 200)),
            )
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
