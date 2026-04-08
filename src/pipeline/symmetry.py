"""
対称性検査パイプライン

compare_images関数をクラスベースのパイプラインに分割したもの。
"""

import math

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from datetime import datetime

# src.coreからの関数インポート
from src.utils.image_utils import safe_imwrite
from src.core import (
    preprocess_image,
    calculate_mse,
    check_image_quality,
    diff_to_bboxes,
    merge_nearby_bboxes,
    auto_align_images,
    align_edge_based,
    align_bbox_based,
    align_fg_icp,
    align_ecc_refine,
    template_match_crop,
)


class SymmetryPipeline:
    """
    対称性検査のメインパイプライン

    compare_images関数の処理を段階的に実行するクラス。
    """

    def __init__(self, config):
        """
        設定を受け取る

        Args:
            config: パラメータ辞書（compare_imagesの引数と同じ）
        """
        self.config = config

    def run(self, master, test, **params):
        """
        メインパイプライン - compare_images相当

        Args:
            master: 基準画像（マスター画像）
            test: 比較画像（テスト画像）
            **params: その他のパラメータ（configをオーバーライド可能）

        Returns:
            (mse, ssim, diff, mask, bboxes, fig, align_info, crop_info, roi_info, flip_info)
        """
        # パラメータをマージ
        config = {**self.config, **params}

        # パラメータを取り出す
        title = config.get("title", "板金画像比較")
        diff_thresh = config.get("diff_thresh", 13)
        min_area = config.get("min_area", 1500)
        morph_kernel = config.get("morph_kernel", 20)
        max_boxes = config.get("max_boxes", 6)
        show_plot = config.get("show_plot", True)
        detect_bbox = config.get("detect_bbox", True)
        use_foreground_mask = config.get("use_foreground_mask", config.get("foreground_mask", False))
        use_background_overlap_check = config.get("use_background_overlap_check", config.get("background_overlap_check", False))
        background_overlap_max_ratio = config.get("background_overlap_max_ratio", 0.5)
        background_overlap_use_diff = config.get("background_overlap_use_diff", False)
        foreground_mask_dilate_iter = config.get("foreground_mask_dilate_iter", 0)
        foreground_mask_kernel = config.get("foreground_mask_kernel", 15)
        foreground_mask_keep_ratio = config.get("foreground_mask_keep_ratio", 0.0)
        foreground_xor_add = config.get("foreground_xor_add", False)
        fg_edge_exclude_enabled = config.get("fg_edge_exclude_enabled", False)
        fg_edge_exclude_iter = config.get("fg_edge_exclude_iter", 2)
        fg_edge_overlap_exclude_enabled = config.get("fg_edge_overlap_exclude_enabled", False)
        fg_edge_raise_thresh_enabled = config.get("fg_edge_raise_thresh_enabled", False)
        fg_edge_raise_thresh_offset = config.get("fg_edge_raise_thresh_offset", 0)
        fg_contour_mask_enabled = config.get("fg_contour_mask_enabled", False)
        fg_contour_mask_canny_low = config.get("fg_contour_mask_canny_low", 50)
        fg_contour_mask_canny_high = config.get("fg_contour_mask_canny_high", 150)
        fg_contour_mask_kernel = config.get("fg_contour_mask_kernel", 5)
        fg_contour_mask_close_iter = config.get("fg_contour_mask_close_iter", 3)
        fg_contour_mask_dilate_iter = config.get("fg_contour_mask_dilate_iter", 1)
        fg_contour_mask_erode_iter = config.get("fg_contour_mask_erode_iter", 5)
        fg_contour_mask_min_area_ratio = config.get("fg_contour_mask_min_area_ratio", 0.05)
        fg_contour_mask_max_area_ratio = config.get("fg_contour_mask_max_area_ratio", 0.98)
        fg_contour_mask_fallback_to_fg = config.get("fg_contour_mask_fallback_to_fg", True)
        use_crop_to_master_fov = config.get("use_crop_to_master_fov", False)
        crop_compare_mode = config.get("crop_compare_mode", True)
        crop_two_pass = config.get("crop_two_pass", False)
        crop_scale_min = config.get("crop_scale_min", 0.3)
        crop_scale_step = config.get("crop_scale_step", 0.05)
        crop_two_pass_min_matches = config.get("crop_two_pass_min_matches", 10)
        crop_ssim_degradation_threshold = config.get("crop_ssim_degradation_threshold", -0.02)
        crop_ssim_min = config.get("crop_ssim_min", 0.3)
        crop_ssim_min_skip_matches = config.get("crop_ssim_min_skip_matches", 20)
        crop_adopt_min_matches = config.get("crop_adopt_min_matches", 8)
        crop_mse_degradation_threshold = config.get("crop_mse_degradation_threshold", 0.05)
        crop_min_area_ratio = config.get("crop_min_area_ratio", 1.5)
        crop_max_rotation_deg = config.get("crop_max_rotation_deg", 15.0)
        crop_min_inlier_ratio = config.get("crop_min_inlier_ratio", 0.3)
        crop_try_rotations = config.get("crop_try_rotations", False)
        crop_rotation_search_step_deg = config.get("crop_rotation_search_step_deg", 90)
        crop_master_padding_ratio = config.get("crop_master_padding_ratio", 0)
        crop_template_match_fallback = config.get("crop_template_match_fallback", True)
        crop_template_match_min_score = config.get("crop_template_match_min_score", 0.3)
        crop_template_match_padding = config.get("crop_template_match_padding", 0.05)
        debug_crop_log = config.get("debug_crop_log", False)
        debug_crop_log_path = config.get("debug_crop_log_path", None)
        debug_crop_save_images = config.get("debug_crop_save_images", False)
        debug_crop_save_images_dir = config.get("debug_crop_save_images_dir", None)
        auto_align = config.get("auto_align", False)
        align_mode = config.get("align_mode", "feature_points")
        align_method = config.get("align_method", "orb")
        align_max_features = config.get("align_max_features", 500)
        align_edge_canny_low = config.get("align_edge_canny_low", 50)
        align_edge_canny_high = config.get("align_edge_canny_high", 150)
        align_match_ratio = config.get("align_match_ratio", 0.75)
        align_compare_mode = config.get("align_compare_mode", True)
        align_rematch_check = config.get("align_rematch_check", True)
        align_fallback_mode = config.get("align_fallback_mode", None)
        align_fallback_on_reject = config.get("align_fallback_on_reject", True)
        align_match_improvement_threshold = config.get("align_match_improvement_threshold", 0.20)
        align_inlier_improvement_threshold = config.get("align_inlier_improvement_threshold", 0.10)
        align_ssim_improvement_threshold = config.get("align_ssim_improvement_threshold", 0.03)
        align_ssim_degradation_threshold = config.get("align_ssim_degradation_threshold", -0.02)
        align_mse_improvement_threshold = config.get("align_mse_improvement_threshold", 0.10)
        align_min_rotation = config.get("align_min_rotation", 3.0)
        align_min_translation = config.get("align_min_translation", 30)
        align_ecc_refine_enabled = config.get("align_ecc_refine_enabled", False)
        align_ecc_blur_ksize = config.get("align_ecc_blur_ksize", 5)
        align_ecc_max_iter = config.get("align_ecc_max_iter", 1000)
        align_ecc_eps = config.get("align_ecc_eps", 1e-6)
        align_ecc_warp_mode = config.get("align_ecc_warp_mode", "affine")
        preprocess_mode = config.get("preprocess_mode", "luminance")
        preprocess_blur_ksize = config.get("preprocess_blur_ksize", 3)
        preprocess_edge_method = config.get("preprocess_edge_method", "canny")
        preprocess_edge_low = config.get("preprocess_edge_low", 50)
        preprocess_edge_high = config.get("preprocess_edge_high", 150)
        preprocess_blackhat_kernel = config.get("preprocess_blackhat_kernel", 15)
        preprocess_contrast_gamma = config.get("preprocess_contrast_gamma", 1.2)
        quality_gate_enabled = config.get("quality_gate_enabled", False)
        quality_gate_block = config.get("quality_gate_block", False)
        quality_gate_white_ratio = config.get("quality_gate_white_ratio", 0.05)
        quality_gate_blur_threshold = config.get("quality_gate_blur_threshold", 100)
        quality_gate_high_brightness = config.get("quality_gate_high_brightness", 250)
        quality_gate_dark_ratio = config.get("quality_gate_dark_ratio", None)
        quality_gate_low_brightness = config.get("quality_gate_low_brightness", 5)
        quality_gate_low_contrast = config.get("quality_gate_low_contrast", None)
        quality_gate_specular_ratio = config.get("quality_gate_specular_ratio", None)
        quality_gate_specular_high_v = config.get("quality_gate_specular_high_v", 240)
        quality_gate_specular_low_s = config.get("quality_gate_specular_low_s", 40)
        use_roi = config.get("use_roi", False)
        roi_x = config.get("roi_x", 0)
        roi_y = config.get("roi_y", 0)
        roi_w = config.get("roi_w", 0)
        roi_h = config.get("roi_h", 0)
        bbox_edge_ignore_ratio = config.get("bbox_edge_ignore_ratio", None)
        bbox_min_area_relax_ratio = config.get("bbox_min_area_relax_ratio", None)
        bbox_min_width = int(config.get("bbox_min_width", 0))
        bbox_drop_band_aspect = float(config.get("bbox_drop_band_aspect", 0))
        bbox_drop_band_max_fill = float(config.get("bbox_drop_band_max_fill", 0))
        bbox_drop_band_aspect_high = float(config.get("bbox_drop_band_aspect_high", 0))
        bbox_drop_band_max_fill_high = float(config.get("bbox_drop_band_max_fill_high", 0))
        bbox_drop_sparse_large_cover = float(config.get("bbox_drop_sparse_large_cover", 0))
        bbox_drop_sparse_large_max_fill = float(config.get("bbox_drop_sparse_large_max_fill", 0))
        bbox_drop_band_min_area_ratio = float(config.get("bbox_drop_band_min_area_ratio", 0))
        bbox_merge_distance = int(config.get("bbox_merge_distance", 0))
        bbox_merge_distance_ratio = float(config.get("bbox_merge_distance_ratio", 0))
        bbox_fg_edge_suppress_enabled = config.get("bbox_fg_edge_suppress_enabled", False)
        bbox_fg_edge_suppress_width = int(config.get("bbox_fg_edge_suppress_width", 15))
        bbox_fg_edge_suppress_offset = int(config.get("bbox_fg_edge_suppress_offset", 30))
        bbox_close_iter = config.get("bbox_close_iter", 4)
        bbox_open_iter = config.get("bbox_open_iter", 1)
        bbox_connectivity = config.get("bbox_connectivity", 8)
        bbox_edge_min_fill_ratio = config.get("bbox_edge_min_fill_ratio", None)
        bbox_edge_cut_enabled = config.get("bbox_edge_cut_enabled", False)
        bbox_edge_cut_kernel = config.get("bbox_edge_cut_kernel", 3)
        bbox_edge_cut_iter = config.get("bbox_edge_cut_iter", 1)
        edge_sep_enabled = config.get("edge_sep_enabled", True)
        edge_sep_min_dist = config.get("edge_sep_min_dist", 3)
        edge_sep_max_dist = config.get("edge_sep_max_dist", 30)
        edge_sep_step = config.get("edge_sep_step", 1)
        bbox_drop_band_fill_ratio = config.get("bbox_drop_band_fill_ratio", None)
        bbox_drop_band_width_ratio = config.get("bbox_drop_band_width_ratio", None)
        bbox_drop_band_height_ratio = config.get("bbox_drop_band_height_ratio", None)
        bbox_edge_roi_enabled = config.get("bbox_edge_roi_enabled", False)
        bbox_edge_roi_canny_low = config.get("bbox_edge_roi_canny_low", 50)
        bbox_edge_roi_canny_high = config.get("bbox_edge_roi_canny_high", 150)
        bbox_edge_roi_kernel = config.get("bbox_edge_roi_kernel", 5)
        bbox_edge_roi_close_iter = config.get("bbox_edge_roi_close_iter", 2)
        bbox_edge_roi_dilate_iter = config.get("bbox_edge_roi_dilate_iter", 1)
        bbox_edge_roi_min_area_ratio = config.get("bbox_edge_roi_min_area_ratio", 0.1)
        bbox_edge_roi_max_area_ratio = config.get("bbox_edge_roi_max_area_ratio", 0.95)
        bbox_edge_roi_require_closed = config.get("bbox_edge_roi_require_closed", True)
        bbox_edge_roi_border_margin = config.get("bbox_edge_roi_border_margin", 2)
        bbox_inner_erode_iter = config.get("bbox_inner_erode_iter", 0)
        bbox_inner_overlap_ratio = config.get("bbox_inner_overlap_ratio", None)
        bbox_rescue_drop_band_fill_ratio = config.get("bbox_rescue_drop_band_fill_ratio", None)
        bbox_rescue_drop_band_width_ratio = config.get("bbox_rescue_drop_band_width_ratio", None)
        bbox_rescue_drop_band_height_ratio = config.get("bbox_rescue_drop_band_height_ratio", None)
        bbox_rescue_thresh_offset = config.get("bbox_rescue_thresh_offset", 0)
        bbox_rescue_morph_kernel = config.get("bbox_rescue_morph_kernel", None)
        bbox_rescue_close_iter = config.get("bbox_rescue_close_iter", None)
        bbox_rescue_open_iter = config.get("bbox_rescue_open_iter", None)
        bbox_rescue_edge_erode_iter = config.get("bbox_rescue_edge_erode_iter", 1)
        bbox_rescue_use_fg_edge_band = config.get("bbox_rescue_use_fg_edge_band", False)
        bbox_rescue_min_area_ratio = config.get("bbox_rescue_min_area_ratio", None)
        hole_mask_enabled = config.get("hole_mask_enabled", False)
        hole_mask_source = config.get("hole_mask_source", "edge_roi_or_foreground")
        hole_mask_apply_mode = config.get("hole_mask_apply_mode", "master")
        hole_mask_method = config.get("hole_mask_method", "auto")
        hole_bg_border_ratio = config.get("hole_bg_border_ratio", 0.05)
        hole_bg_dist_percentile = config.get("hole_bg_dist_percentile", 95.0)
        hole_bg_dist_scale = config.get("hole_bg_dist_scale", 1.5)
        hole_bg_dist_min = config.get("hole_bg_dist_min", 6.0)
        hole_bg_dist_max = config.get("hole_bg_dist_max", 60.0)
        hole_inner_erode_iter = config.get("hole_inner_erode_iter", 1)
        hole_expand_iter = config.get("hole_expand_iter", 0)
        hole_edge_exclude_enabled = config.get("hole_edge_exclude_enabled", False)
        hole_edge_exclude_iter = config.get("hole_edge_exclude_iter", 2)
        fg_hole_edge_exclude_enabled = config.get("fg_hole_edge_exclude_enabled", False)
        fg_hole_edge_exclude_dilate = int(config.get("fg_hole_edge_exclude_dilate", 25))
        fg_hole_edge_exclude_min_area_ratio = float(config.get("fg_hole_edge_exclude_min_area_ratio", 0.05))
        hole_bbox_filter_enabled = config.get("hole_bbox_filter_enabled", False)
        hole_bbox_filter_dilate_iter = config.get("hole_bbox_filter_dilate_iter", 0)
        hole_bbox_filter_overlap_ratio = config.get("hole_bbox_filter_overlap_ratio", 0.2)
        hole_bbox_filter_max_area_ratio = config.get("hole_bbox_filter_max_area_ratio", 0.01)
        hole_bbox_filter_use_center = config.get("hole_bbox_filter_use_center", True)
        hole_min_area_ratio = config.get("hole_min_area_ratio", 0.0002)
        hole_max_area_ratio = config.get("hole_max_area_ratio", 0.08)
        hole_min_fill_ratio = config.get("hole_min_fill_ratio", 0.5)
        hole_aspect_min = config.get("hole_aspect_min", 0.6)
        hole_aspect_max = config.get("hole_aspect_max", 1.4)
        hole_shrink_iter = config.get("hole_shrink_iter", 1)
        hole_max_total_ratio = config.get("hole_max_total_ratio", 0.15)
        strong_bbox_edge_roi_enabled = config.get("strong_bbox_edge_roi_enabled", False)
        strong_bbox_edge_roi_canny_low = config.get("strong_bbox_edge_roi_canny_low", 50)
        strong_bbox_edge_roi_canny_high = config.get("strong_bbox_edge_roi_canny_high", 150)
        strong_bbox_edge_roi_kernel = config.get("strong_bbox_edge_roi_kernel", 5)
        strong_bbox_edge_roi_close_iter = config.get("strong_bbox_edge_roi_close_iter", 2)
        strong_bbox_edge_roi_dilate_iter = config.get("strong_bbox_edge_roi_dilate_iter", 1)
        strong_bbox_edge_roi_min_area_ratio = config.get("strong_bbox_edge_roi_min_area_ratio", 0.1)
        strong_bbox_edge_roi_max_area_ratio = config.get("strong_bbox_edge_roi_max_area_ratio", 0.95)
        strong_bbox_edge_roi_require_closed = config.get("strong_bbox_edge_roi_require_closed", True)
        strong_bbox_edge_roi_border_margin = config.get("strong_bbox_edge_roi_border_margin", 2)
        strong_bbox_drop_band_fill_ratio = config.get("strong_bbox_drop_band_fill_ratio", None)
        strong_bbox_drop_band_width_ratio = config.get("strong_bbox_drop_band_width_ratio", None)
        strong_bbox_drop_band_height_ratio = config.get("strong_bbox_drop_band_height_ratio", None)
        strong_bbox_min_diff_max = config.get("strong_bbox_min_diff_max", 0)
        strong_bbox_edge_margin = config.get("strong_bbox_edge_margin", 0)
        strong_bbox_min_fill = config.get("strong_bbox_min_fill", 0)
        use_flip_compare = config.get("use_flip_compare", False)
        flip_compare_min_ssim = config.get("flip_compare_min_ssim", None)
        flip_compare_min_improvement = config.get("flip_compare_min_improvement", None)
        # --- 構造比較パラメータ ---
        structural_comparison_enabled = config.get("structural_comparison_enabled", False)
        structural_ssim_normal_thresh = float(config.get("structural_ssim_normal_thresh", 0.70))
        structural_ssim_flip_thresh = float(config.get("structural_ssim_flip_thresh", 0.65))
        structural_ssim_flip_min_improvement = float(config.get("structural_ssim_flip_min_improvement", 0.10))
        structural_iou_different_part = float(config.get("structural_iou_different_part", 0.30))
        structural_iou_deficient = float(config.get("structural_iou_deficient", 0.60))
        structural_overlap_deficiency = float(config.get("structural_overlap_deficiency", 0.50))
        structural_block_bbox_on_fail = config.get("structural_block_bbox_on_fail", True)
        # --- 傷検出パラメータ ---
        scratch_detection_enabled = config.get("scratch_detection_enabled", False)
        scratch_blur_ksize = int(config.get("scratch_blur_ksize", 5))
        scratch_blackhat_kernel = int(config.get("scratch_blackhat_kernel", 15))
        scratch_diff_thresh = int(config.get("scratch_diff_thresh", 40))
        scratch_min_area = int(config.get("scratch_min_area", 50))
        scratch_min_area_ratio = float(config.get("scratch_min_area_ratio", 0))
        scratch_min_width = int(config.get("scratch_min_width", 0))
        scratch_min_aspect_ratio = float(config.get("scratch_min_aspect_ratio", 2.0))
        scratch_max_area_ratio = float(config.get("scratch_max_area_ratio", 5.0))
        scratch_morph_kernel = int(config.get("scratch_morph_kernel", 3))
        scratch_morph_open_iter = int(config.get("scratch_morph_open_iter", 1))
        scratch_morph_close_iter = int(config.get("scratch_morph_close_iter", 1))
        scratch_max_count = int(config.get("scratch_max_count", 20))
        scratch_fg_mask_erode_iter = int(config.get("scratch_fg_mask_erode_iter", 10))
        # --- ピンアート外形照合パラメータ ---
        pin_profile_enabled = bool(config.get("pin_profile_enabled", False))
        pin_burr_threshold = int(config.get("pin_burr_threshold", 5))
        pin_chip_threshold = int(config.get("pin_chip_threshold", 5))
        pin_noise_erode = int(config.get("pin_noise_erode", 1))
        pin_anchor_band_min_length = int(config.get("pin_anchor_band_min_length", 5))
        # --- 輪郭差分検出パラメータ ---
        contour_detection_enabled = config.get("contour_detection_enabled", False)
        contour_blur_ksize = int(config.get("contour_blur_ksize", 5))
        contour_canny_low = int(config.get("contour_canny_low", 50))
        contour_canny_high = int(config.get("contour_canny_high", 150))
        contour_diff_thresh = int(config.get("contour_diff_thresh", 0))
        contour_min_area = int(config.get("contour_min_area", 500))
        contour_min_length = int(config.get("contour_min_length", 50))
        contour_morph_kernel = int(config.get("contour_morph_kernel", 5))
        contour_morph_close_iter = int(config.get("contour_morph_close_iter", 3))
        contour_max_count = int(config.get("contour_max_count", 5))
        contour_fg_mask_dilate_iter = int(config.get("contour_fg_mask_dilate_iter", 20))
        contour_edge_band_width = int(config.get("contour_edge_band_width", 30))
        # --- 内部平面差分検出パラメータ ---
        surface_detection_enabled = bool(config.get("surface_detection_enabled", False))
        surface_inner_erode_iter = int(config.get("surface_inner_erode_iter", 12))
        surface_diff_thresh = int(config.get("surface_diff_thresh", 50))
        surface_min_area = int(config.get("surface_min_area", 150))
        surface_min_area_ratio = float(config.get("surface_min_area_ratio", 0.0))  # >0で画像面積比から自動計算
        surface_max_area_ratio = float(config.get("surface_max_area_ratio", 8.0))
        surface_max_count = int(config.get("surface_max_count", 20))
        surface_flat_exclude_enabled = bool(config.get("surface_flat_exclude_enabled", False))
        surface_flat_canny_low = int(config.get("surface_flat_canny_low", 30))
        surface_flat_canny_high = int(config.get("surface_flat_canny_high", 80))
        surface_flat_edge_dilate_iter = int(config.get("surface_flat_edge_dilate_iter", 8))
        # structural_comparison が有効なら flip_compare は無効化（構造比較がflipを管理する）
        if structural_comparison_enabled and use_flip_compare:
            print("[StructuralComparison] structural_comparison_enabled=true のため flip_compare は無効化されます")
            use_flip_compare = False
        debug_save_pipeline_stages = config.get("debug_save_pipeline_stages", False)
        debug_save_pipeline_dir = config.get("debug_save_pipeline_dir", None)

        # 左右反転比較: 元画像と反転画像の両方で比較し、SSIMが高い方の結果を採用（線対称での誤判定低減）
        if use_flip_compare:
            return self._handle_flip_compare(
                master,
                test,
                config,
                flip_compare_min_ssim=flip_compare_min_ssim,
                flip_compare_min_improvement=flip_compare_min_improvement,
            )

        imageA = master
        imageB = test

        # パイプライン全段階で画像を保存（原因切り分け用）
        _pipe_dir = None
        bbox_debug_log_path = None
        strong_bbox_debug_log_path = None
        if debug_save_pipeline_stages and debug_save_pipeline_dir:
            try:
                _pipe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _pipe_dir = os.path.join(debug_save_pipeline_dir, _pipe_ts)
                os.makedirs(_pipe_dir, exist_ok=True)
                bbox_debug_log_path = os.path.join(_pipe_dir, "11_mask_components.txt")
                strong_bbox_debug_log_path = os.path.join(_pipe_dir, "11_mask_strong_components.txt")
                print(f"  [パイプライン段階保存] {os.path.abspath(_pipe_dir)}")
                try:
                    snap_path = os.path.join(_pipe_dir, "00_config_snapshot.txt")
                    with open(snap_path, "w", encoding="utf-8") as f:
                        f.write(f"diff_thresh={diff_thresh}\n")
                        f.write(f"min_area={min_area}\n")
                        f.write(f"morph_kernel={morph_kernel}\n")
                        f.write(f"max_boxes={max_boxes}\n")
                        f.write(f"bbox_edge_ignore_ratio={bbox_edge_ignore_ratio}\n")
                        f.write(f"bbox_edge_min_fill_ratio={bbox_edge_min_fill_ratio}\n")
                        f.write(f"bbox_close_iter={bbox_close_iter}\n")
                        f.write(f"bbox_open_iter={bbox_open_iter}\n")
                        f.write(f"bbox_connectivity={bbox_connectivity}\n")
                        f.write(f"bbox_edge_cut_enabled={bbox_edge_cut_enabled}\n")
                        f.write(f"bbox_edge_cut_kernel={bbox_edge_cut_kernel}\n")
                        f.write(f"bbox_edge_cut_iter={bbox_edge_cut_iter}\n")
                        f.write(f"bbox_drop_band_fill_ratio={bbox_drop_band_fill_ratio}\n")
                        f.write(f"bbox_drop_band_width_ratio={bbox_drop_band_width_ratio}\n")
                        f.write(f"bbox_drop_band_height_ratio={bbox_drop_band_height_ratio}\n")
                        f.write(f"bbox_inner_erode_iter={bbox_inner_erode_iter}\n")
                        f.write(f"bbox_inner_overlap_ratio={bbox_inner_overlap_ratio}\n")
                        f.write(f"bbox_rescue_drop_band_fill_ratio={bbox_rescue_drop_band_fill_ratio}\n")
                        f.write(f"bbox_rescue_drop_band_width_ratio={bbox_rescue_drop_band_width_ratio}\n")
                        f.write(f"bbox_rescue_drop_band_height_ratio={bbox_rescue_drop_band_height_ratio}\n")
                        f.write(f"bbox_rescue_thresh_offset={bbox_rescue_thresh_offset}\n")
                        f.write(f"bbox_rescue_morph_kernel={bbox_rescue_morph_kernel}\n")
                        f.write(f"bbox_rescue_close_iter={bbox_rescue_close_iter}\n")
                        f.write(f"bbox_rescue_open_iter={bbox_rescue_open_iter}\n")
                        f.write(f"bbox_rescue_edge_erode_iter={bbox_rescue_edge_erode_iter}\n")
                        f.write(f"bbox_rescue_use_fg_edge_band={bbox_rescue_use_fg_edge_band}\n")
                        f.write(f"bbox_rescue_min_area_ratio={bbox_rescue_min_area_ratio}\n")
                        f.write(f"hole_mask_enabled={hole_mask_enabled}\n")
                        f.write(f"hole_mask_apply_mode={hole_mask_apply_mode}\n")
                        f.write(f"hole_mask_method={hole_mask_method}\n")
                        f.write(f"hole_edge_exclude_enabled={hole_edge_exclude_enabled}\n")
                        f.write(f"hole_edge_exclude_iter={hole_edge_exclude_iter}\n")
                        f.write(f"hole_bbox_filter_enabled={hole_bbox_filter_enabled}\n")
                        f.write(f"strong_bbox_enabled={config.get('strong_bbox_enabled', False)}\n")
                        f.write(f"strong_bbox_drop_band_fill_ratio={strong_bbox_drop_band_fill_ratio}\n")
                        f.write(f"strong_bbox_drop_band_width_ratio={strong_bbox_drop_band_width_ratio}\n")
                        f.write(f"strong_bbox_drop_band_height_ratio={strong_bbox_drop_band_height_ratio}\n")
                except Exception:
                    pass
            except Exception as _e:
                _pipe_dir = None
                bbox_debug_log_path = None
                strong_bbox_debug_log_path = None
                print(f"  [パイプライン段階保存] フォルダ作成失敗: {_e}")

        def _save_stage(name, img_a=None, img_b=None, single=None):
            if _pipe_dir is None:
                return
            try:
                if img_a is not None:
                    fp = os.path.join(_pipe_dir, f"{name}_A.png")
                    safe_imwrite(fp, img_a)
                if img_b is not None:
                    fp = os.path.join(_pipe_dir, f"{name}_B.png")
                    safe_imwrite(fp, img_b)
                if single is not None:
                    fp = os.path.join(_pipe_dir, f"{name}.png")
                    safe_imwrite(fp, single)
            except Exception as _e:
                print(f"  [パイプライン段階保存] {name} 保存失敗: {_e}")

        _save_stage("01_initial", imageA, imageB)

        def _center_crop(image, target_h, target_w):
            h, w = image.shape[:2]
            if h == target_h and w == target_w:
                return image
            if h < target_h or w < target_w:
                return None
            y0 = (h - target_h) // 2
            x0 = (w - target_w) // 2
            return image[y0 : y0 + target_h, x0 : x0 + target_w]

        def _pad_to_size(image, target_h, target_w):
            h, w = image.shape[:2]
            if h == target_h and w == target_w:
                return image
            # 画像が target より大きい場合は中央クロップ（pad だと broadcast エラーになる）
            if h > target_h or w > target_w:
                cropped = _center_crop(image, target_h, target_w)
                if cropped is not None:
                    return cropped
            if image.ndim == 2:
                result = np.zeros((target_h, target_w), dtype=image.dtype)
            else:
                result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            y0 = max(0, (target_h - h) // 2)
            x0 = max(0, (target_w - w) // 2)
            result[y0 : y0 + h, x0 : x0 + w] = image
            return result

        def _preprocess(img):
            return preprocess_image(
                img,
                blur_ksize=preprocess_blur_ksize,
                mode=preprocess_mode,
                edge_method=preprocess_edge_method,
                edge_low=preprocess_edge_low,
                edge_high=preprocess_edge_high,
                blackhat_kernel=preprocess_blackhat_kernel,
                contrast_gamma=preprocess_contrast_gamma,
            )

        # 画角切り出しの結果をUI・JSON用に保持
        crop_info = None

        # 画角をマスターに合わせる：比較画像からマスターと同じ範囲だけ切り出す（オンオフ可）
        if use_crop_to_master_fov:
            imageA, imageB, crop_info = self._crop_to_fov(
                imageA, imageB,
                align_method=align_method,
                align_max_features=align_max_features,
                align_match_ratio=align_match_ratio,
                crop_compare_mode=crop_compare_mode,
                crop_two_pass=crop_two_pass,
                crop_scale_min=crop_scale_min,
                crop_scale_step=crop_scale_step,
                crop_two_pass_min_matches=crop_two_pass_min_matches,
                crop_ssim_degradation_threshold=crop_ssim_degradation_threshold,
                crop_ssim_min=crop_ssim_min,
                crop_ssim_min_skip_matches=crop_ssim_min_skip_matches,
                crop_adopt_min_matches=crop_adopt_min_matches,
                crop_mse_degradation_threshold=crop_mse_degradation_threshold,
                crop_min_area_ratio=crop_min_area_ratio,
                crop_max_rotation_deg=crop_max_rotation_deg,
                crop_min_inlier_ratio=crop_min_inlier_ratio,
                crop_try_rotations=crop_try_rotations,
                crop_rotation_search_step_deg=crop_rotation_search_step_deg,
                crop_master_padding_ratio=crop_master_padding_ratio,
                crop_template_match_fallback=crop_template_match_fallback,
                crop_template_match_min_score=crop_template_match_min_score,
                crop_template_match_padding=crop_template_match_padding,
                debug_crop_log=debug_crop_log,
                debug_crop_log_path=debug_crop_log_path,
                debug_crop_save_images=debug_crop_save_images,
                debug_crop_save_images_dir=debug_crop_save_images_dir,
                _preprocess=_preprocess,
                try_flip=not structural_comparison_enabled,
            )

        _save_stage("02_after_crop", imageA, imageB)
        # 段階2の失敗切り分け用: 画角合わせの結果をテキストで残す
        if _pipe_dir is not None:
            try:
                lines = []
                if crop_info is None:
                    lines.append("use_crop_to_master_fov=False (画角合わせなし)")
                else:
                    lines.append(f"enabled={crop_info.get('enabled', '')} compare_mode={crop_info.get('compare_mode', '')} adopted={crop_info.get('adopted', '')}")
                    if crop_info.get("reason") is not None:
                        lines.append(f"reason={crop_info.get('reason', '')}")
                    if crop_info.get("adopted_scale") is not None:
                        lines.append(f"adopted_scale={crop_info.get('adopted_scale')}")
                    if crop_info.get("matches") is not None:
                        lines.append(f"matches={crop_info.get('matches')} matches_orig={crop_info.get('matches_orig', '')}")
                    if crop_info.get("match_ratio") is not None:
                        lines.append(f"match_ratio={crop_info.get('match_ratio')} match_ratio_min={crop_info.get('match_ratio_min')}")
                    if crop_info.get("ssim") is not None or crop_info.get("ssim_orig") is not None:
                        lines.append(f"ssim={crop_info.get('ssim')} ssim_orig={crop_info.get('ssim_orig')}")
                    if crop_info.get("mse") is not None or crop_info.get("mse_orig") is not None:
                        lines.append(f"mse={crop_info.get('mse')} mse_orig={crop_info.get('mse_orig')}")
                    if crop_info.get("match_ok") is not None:
                        lines.append(
                            "flags="
                            + ",".join(
                                [
                                    f"match_ok={int(crop_info.get('match_ok'))}",
                                    f"ssim_ok={int(crop_info.get('ssim_ok'))}",
                                    f"mse_ok={int(crop_info.get('mse_ok'))}",
                                    f"ssim_min_ok={int(crop_info.get('ssim_min_ok'))}",
                                    f"adopt_match_ok={int(crop_info.get('adopt_match_ok'))}",
                                ]
                            )
                        )
                    if crop_info.get("no_crop_resized") is not None:
                        lines.append(f"no_crop_resized={int(crop_info.get('no_crop_resized'))} resize_shape={crop_info.get('no_crop_resize_shape')}")
                fp_info = os.path.join(_pipe_dir, "02_after_crop_info.txt")
                with open(fp_info, "w", encoding="utf-8") as _f:
                    _f.write("\n".join(lines))
            except Exception as _e:
                print(f"  [パイプライン段階保存] 02_after_crop_info.txt 保存失敗: {_e}")

        # サイズ合わせ: 処理には同じサイズが必要なので、基準画像サイズに比較画像を揃える（1ピクセル差も解消）
        target_h, target_w = imageA.shape[:2]
        if imageB.shape[:2] != (target_h, target_w):
            imageB = cv2.resize(imageB, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        _save_stage("03_after_size_align", imageA, imageB)

        # マッチング情報を格納する辞書（UI表示用）
        align_info = None

        # 自動位置合わせ（角度・位置ずれ補正）
        if auto_align:
            imageB, align_info = self._align_images(
                imageA, imageB,
                align_mode=align_mode,
                align_method=align_method,
                align_max_features=align_max_features,
                align_edge_canny_low=align_edge_canny_low,
                align_edge_canny_high=align_edge_canny_high,
                align_match_ratio=align_match_ratio,
                align_min_rotation=align_min_rotation,
                align_min_translation=align_min_translation,
                align_compare_mode=align_compare_mode,
                align_rematch_check=align_rematch_check,
                align_fallback_mode=align_fallback_mode,
                align_fallback_on_reject=align_fallback_on_reject,
                align_match_improvement_threshold=align_match_improvement_threshold,
                align_inlier_improvement_threshold=align_inlier_improvement_threshold,
                align_ssim_improvement_threshold=align_ssim_improvement_threshold,
                align_ssim_degradation_threshold=align_ssim_degradation_threshold,
                align_mse_improvement_threshold=align_mse_improvement_threshold,
                align_ecc_refine_enabled=align_ecc_refine_enabled,
                align_ecc_blur_ksize=align_ecc_blur_ksize,
                align_ecc_max_iter=align_ecc_max_iter,
                align_ecc_eps=align_ecc_eps,
                align_ecc_warp_mode=align_ecc_warp_mode,
                _preprocess=_preprocess,
            )

        _save_stage("04_after_auto_align", imageA, imageB)
        if _pipe_dir is not None:
            try:
                lines = []
                if not align_info:
                    lines.append("align_info=none")
                else:
                    lines.append(f"enabled={align_info.get('enabled')}")
                    lines.append(f"success={align_info.get('success')}")
                    lines.append(f"applied={align_info.get('applied')}")
                    if align_info.get("decision") is not None:
                        lines.append(f"decision={align_info.get('decision')}")
                    if align_info.get("decision_step") is not None:
                        lines.append(f"decision_step={align_info.get('decision_step')}")
                    if align_info.get("rotation_deg") is not None:
                        lines.append(f"rotation_deg={align_info.get('rotation_deg')}")
                    if align_info.get("translation_px") is not None:
                        lines.append(f"translation_px={align_info.get('translation_px')}")
                    if align_info.get("matches_before") is not None:
                        lines.append(f"matches_before={align_info.get('matches_before')}")
                    if align_info.get("inlier_ratio_before") is not None:
                        lines.append(f"inlier_ratio_before={align_info.get('inlier_ratio_before')}")
                    if align_info.get("matches_after") is not None:
                        lines.append(f"matches_after={align_info.get('matches_after')}")
                    if align_info.get("inlier_ratio_after") is not None:
                        lines.append(f"inlier_ratio_after={align_info.get('inlier_ratio_after')}")
                    if align_info.get("fallback_used") is not None:
                        lines.append(f"fallback_used={align_info.get('fallback_used')}")
                    if align_info.get("fallback_failure_reason") is not None:
                        lines.append(f"fallback_failure_reason={align_info.get('fallback_failure_reason')}")
                    cmp_metrics = align_info.get("compare_metrics")
                    if isinstance(cmp_metrics, dict):
                        lines.append("compare_metrics:")
                        for k in ("ssim_original", "ssim_aligned", "mse_original", "mse_aligned", "ssim_improvement", "mse_improvement"):
                            if k in cmp_metrics:
                                lines.append(f"  {k}={cmp_metrics.get(k)}")
                fp_info = os.path.join(_pipe_dir, "04_after_auto_align_info.txt")
                with open(fp_info, "w", encoding="utf-8") as _f:
                    _f.write("\n".join(lines))
            except Exception as _e:
                print(f"  [パイプライン段階保存] 04_after_auto_align_info.txt 保存失敗: {_e}")

        # 比較前にサイズを必ず揃える（broadcast エラー防止）
        target_h, target_w = imageA.shape[:2]
        if imageB.shape[:2] != (target_h, target_w):
            imageB = cv2.resize(imageB, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        _save_stage("05_after_size_align2", imageA, imageB)

        # --- 比率ベース自動スケーリング（画像サイズ確定後） ---
        _img_h, _img_w = imageA.shape[:2]
        _image_area = _img_h * _img_w
        _img_short_side = min(_img_h, _img_w)

        if scratch_min_area_ratio > 0 and _image_area > 0:
            scratch_min_area = max(1, int(_image_area * scratch_min_area_ratio))
            print(f"  [auto] scratch_min_area: ratio={scratch_min_area_ratio} -> {scratch_min_area}px (image_area={_image_area})")

        if bbox_merge_distance_ratio > 0 and _img_short_side > 0:
            bbox_merge_distance = max(1, int(_img_short_side * bbox_merge_distance_ratio))
            print(f"  [auto] bbox_merge_distance: ratio={bbox_merge_distance_ratio} -> {bbox_merge_distance}px (short_side={_img_short_side})")

        # --- auto_morph_kernel: 画像面積に応じてmorph_kernelを自動計算 ---
        _auto_morph = config.get("auto_morph_kernel", False)
        if _auto_morph and _image_area > 0:
            import math
            if _image_area < 200000:
                _mk_coeff = float(config.get("morph_kernel_coefficient_small", 60))
            elif _image_area < 800000:
                _mk_coeff = float(config.get("morph_kernel_coefficient_medium", 65))
            else:
                _mk_coeff = float(config.get("morph_kernel_coefficient_large", 95))
            _mk = max(3, int(round(math.sqrt(_image_area) / _mk_coeff)))
            if _mk % 2 == 0:
                _mk += 1
            print(f"  [auto] morph_kernel: {morph_kernel} -> {_mk} (area={_image_area}, coeff={_mk_coeff})")
            morph_kernel = _mk

        # --- auto_min_area: 画像面積に応じてmin_areaを自動計算 ---
        _auto_min = config.get("auto_min_area", False)
        _min_area_ratio = float(config.get("min_area_ratio", 0))
        if _auto_min and _min_area_ratio > 0 and _image_area > 0:
            _old_min_area = min_area
            min_area = max(10, int(_image_area * _min_area_ratio))
            print(f"  [auto] min_area: {_old_min_area} -> {min_area} (ratio={_min_area_ratio}, area={_image_area})")

        # ROI指定検査（ON時のみ）: 位置合わせ・サイズ揃えの後、両画像を同じ矩形でクロップしてから差分・BBOX
        roi_info = None
        if use_roi and roi_w > 0 and roi_h > 0:
            imageA, imageB, roi_info = self._apply_roi(imageA, imageB, roi_x, roi_y, roi_w, roi_h)

        _save_stage("06_after_roi", imageA, imageB)

        # 前処理直前でサイズを再確認（1ピクセル差の broadcast エラー防止）
        if imageA.shape[:2] != imageB.shape[:2]:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 画質ゲーティング（BBOXより前に品質チェック）
        quality_info = {"enabled": False}
        if quality_gate_enabled:
            ok_a, reason_a, metrics_a = check_image_quality(
                imageA,
                white_ratio_threshold=quality_gate_white_ratio,
                blur_threshold=quality_gate_blur_threshold,
                high_brightness=quality_gate_high_brightness,
                dark_ratio_threshold=quality_gate_dark_ratio,
                low_brightness=quality_gate_low_brightness,
                low_contrast_threshold=quality_gate_low_contrast,
                specular_ratio_threshold=quality_gate_specular_ratio,
                specular_high_v=quality_gate_specular_high_v,
                specular_low_s=quality_gate_specular_low_s,
                return_details=True,
            )
            ok_b, reason_b, metrics_b = check_image_quality(
                imageB,
                white_ratio_threshold=quality_gate_white_ratio,
                blur_threshold=quality_gate_blur_threshold,
                high_brightness=quality_gate_high_brightness,
                dark_ratio_threshold=quality_gate_dark_ratio,
                low_brightness=quality_gate_low_brightness,
                low_contrast_threshold=quality_gate_low_contrast,
                specular_ratio_threshold=quality_gate_specular_ratio,
                specular_high_v=quality_gate_specular_high_v,
                specular_low_s=quality_gate_specular_low_s,
                return_details=True,
            )
            passed = bool(ok_a and ok_b)
            reasons = []
            if not ok_a:
                reasons.append(f"マスター: {reason_a or 'NG'}")
            if not ok_b:
                reasons.append(f"比較: {reason_b or 'NG'}")
            quality_info = {
                "enabled": True,
                "passed": passed,
                "blocked": bool(quality_gate_block and not passed),
                "reasons": reasons,
                "thresholds": {
                    "white_ratio": quality_gate_white_ratio,
                    "blur": quality_gate_blur_threshold,
                    "high_brightness": quality_gate_high_brightness,
                    "dark_ratio": quality_gate_dark_ratio,
                    "low_brightness": quality_gate_low_brightness,
                    "low_contrast": quality_gate_low_contrast,
                    "specular_ratio": quality_gate_specular_ratio,
                    "specular_high_v": quality_gate_specular_high_v,
                    "specular_low_s": quality_gate_specular_low_s,
                },
                "master": metrics_a,
                "test": metrics_b,
            }
            print(f"[QualityGate] master={int(ok_a)} test={int(ok_b)} blocked={int(quality_info['blocked'])} reasons={reasons}")
            if _pipe_dir is not None:
                try:
                    q_path = os.path.join(_pipe_dir, "06_quality_gate.txt")
                    with open(q_path, "w", encoding="utf-8") as f:
                        f.write(f"quality_gate_enabled={quality_gate_enabled}\n")
                        f.write(f"quality_gate_block={quality_gate_block}\n")
                        f.write(f"passed={passed}\n")
                        f.write(f"reasons={reasons}\n")
                        f.write("thresholds:\n")
                        for k, v in quality_info["thresholds"].items():
                            f.write(f"  {k}={v}\n")
                        f.write("master_metrics:\n")
                        for k, v in metrics_a.items():
                            f.write(f"  {k}={v}\n")
                        f.write("test_metrics:\n")
                        for k, v in metrics_b.items():
                            f.write(f"  {k}={v}\n")
                except Exception as _e:
                    print(f"  [QualityGate] ログ保存失敗: {_e}")

        # --- 構造比較（通常→flip→構造判定の3段階フロー） ---
        structural_info = None
        if structural_comparison_enabled:
            imageA, imageB, structural_info = self._structural_comparison(
                imageA, imageB,
                ssim_normal_thresh=structural_ssim_normal_thresh,
                ssim_flip_thresh=structural_ssim_flip_thresh,
                ssim_flip_min_improvement=structural_ssim_flip_min_improvement,
                iou_different_part=structural_iou_different_part,
                iou_deficient=structural_iou_deficient,
                overlap_deficiency=structural_overlap_deficiency,
                block_bbox_on_fail=structural_block_bbox_on_fail,
                _pipe_dir=_pipe_dir,
            )
            # 構造比較の結果をデバッグ保存
            if _pipe_dir is not None:
                try:
                    sc_path = os.path.join(_pipe_dir, "05b_structural_info.txt")
                    with open(sc_path, "w", encoding="utf-8") as f:
                        for k, v in structural_info.items():
                            f.write(f"{k}={v}\n")
                except Exception:
                    pass
            # flip採用時にサイズ再確認
            if imageA.shape[:2] != imageB.shape[:2]:
                imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 前処理（選択した画像処理項目で差分検知）
        processedA = _preprocess(imageA)
        processedB = _preprocess(imageB)

        _save_stage("07_preprocessed", processedA, processedB)

        # 指標
        mse_value = calculate_mse(processedA, processedB)
        # 前処理はuint8(0-255)なのでdata_rangeを明示。floatの場合は値域から算出
        _dr = 255.0 if np.issubdtype(processedA.dtype, np.integer) else float(max(processedA.max() - processedA.min(), processedB.max() - processedB.min(), 1e-6))
        ssim_score, ssim_map = ssim(processedA, processedB, full=True, data_range=_dr)

        # 5枚目表示の切り分け用: 計算結果が正しいか確認（NaN/定数/範囲）
        _ssim_map = np.asarray(ssim_map, dtype=np.float64)
        _nan = np.isnan(_ssim_map).sum()
        _inf = np.isinf(_ssim_map).sum()
        _min = np.nanmin(_ssim_map) if _ssim_map.size else float("nan")
        _max = np.nanmax(_ssim_map) if _ssim_map.size else float("nan")
        _mean = np.nanmean(_ssim_map) if _ssim_map.size else float("nan")
        print(f"[SSIM map] shape={_ssim_map.shape}, min={_min:.4f}, max={_max:.4f}, mean={_mean:.4f}, NaN={_nan}, Inf={_inf}, data_range={_dr}")
        if ssim_score < 0.1:
            print(f"[SSIM low] score={ssim_score:.4f} → processedA: shape={processedA.shape}, dtype={processedA.dtype}; processedB: shape={processedB.shape}, dtype={processedB.dtype}")

        # --- 差分信号の選択 ---
        _diff_method = config.get("diff_method", "ssim")
        if _diff_method == "ssim":
            # SSIMマップベース差分（低SSIM=高anomaly → 0-255にスケーリング）
            _ssim_anomaly = (1.0 - np.clip(ssim_map, 0.0, 1.0)) * 255.0
            diff = _ssim_anomaly.astype(np.uint8)
        else:
            # 従来の絶対差分
            diff = cv2.absdiff(processedA, processedB)
        diff_before_fg_mask = diff.copy() if use_background_overlap_check else None
        diff_original = diff.copy()  # strong BBOX用（fg_mask/erode/suppress前の元差分）

        _save_stage("08_diff", single=diff)

        # 前景マスク、背景重なりチェック、BBOX検出（ゲーティングNG or 構造判定ブロックならBBOXはスキップ）
        _structural_blocked = structural_info is not None and structural_info.get("block_bbox", False)
        if _structural_blocked:
            print(f"[StructuralBlock] classification={structural_info.get('classification')} → BBOX検出は続行（情報のみ）")
        if quality_info.get("blocked"):
            mask = np.zeros_like(diff, dtype=np.uint8)
            bboxes = []
            fg_mask = None
            strong_bboxes = []
            type_masks = {"shape": None, "surface": None}
        else:
            diff, mask, bboxes, fg_mask, strong_bboxes, type_masks = self._detect_differences(
                diff, imageA, imageB,
                diff_before_fg_mask=diff_before_fg_mask,
                use_foreground_mask=use_foreground_mask,
                use_background_overlap_check=use_background_overlap_check,
                background_overlap_max_ratio=background_overlap_max_ratio,
                background_overlap_use_diff=background_overlap_use_diff,
                foreground_mask_dilate_iter=foreground_mask_dilate_iter,
                foreground_mask_kernel=foreground_mask_kernel,
                foreground_mask_keep_ratio=foreground_mask_keep_ratio,
                foreground_xor_add=foreground_xor_add,
                fg_edge_exclude_enabled=fg_edge_exclude_enabled,
                fg_edge_exclude_iter=fg_edge_exclude_iter,
                fg_edge_overlap_exclude_enabled=fg_edge_overlap_exclude_enabled,
                fg_edge_raise_thresh_enabled=fg_edge_raise_thresh_enabled,
                fg_edge_raise_thresh_offset=fg_edge_raise_thresh_offset,
                fg_contour_mask_enabled=fg_contour_mask_enabled,
                fg_contour_mask_canny_low=fg_contour_mask_canny_low,
                fg_contour_mask_canny_high=fg_contour_mask_canny_high,
                fg_contour_mask_kernel=fg_contour_mask_kernel,
                fg_contour_mask_close_iter=fg_contour_mask_close_iter,
                fg_contour_mask_dilate_iter=fg_contour_mask_dilate_iter,
                fg_contour_mask_erode_iter=fg_contour_mask_erode_iter,
                fg_contour_mask_min_area_ratio=fg_contour_mask_min_area_ratio,
                fg_contour_mask_max_area_ratio=fg_contour_mask_max_area_ratio,
                fg_contour_mask_fallback_to_fg=fg_contour_mask_fallback_to_fg,
                detect_bbox=detect_bbox,
                diff_thresh=diff_thresh,
                min_area=min_area,
                morph_kernel=morph_kernel,
                max_boxes=max_boxes,
                bbox_edge_ignore_ratio=bbox_edge_ignore_ratio,
                bbox_min_area_relax_ratio=bbox_min_area_relax_ratio,
                bbox_min_width=bbox_min_width,
                bbox_drop_band_aspect=bbox_drop_band_aspect,
                bbox_drop_band_max_fill=bbox_drop_band_max_fill,
                bbox_drop_band_aspect_high=bbox_drop_band_aspect_high,
                bbox_drop_band_max_fill_high=bbox_drop_band_max_fill_high,
                bbox_drop_sparse_large_cover=bbox_drop_sparse_large_cover,
                bbox_drop_sparse_large_max_fill=bbox_drop_sparse_large_max_fill,
                bbox_fg_edge_suppress_enabled=bbox_fg_edge_suppress_enabled,
                bbox_fg_edge_suppress_width=bbox_fg_edge_suppress_width,
                bbox_fg_edge_suppress_offset=bbox_fg_edge_suppress_offset,
                bbox_close_iter=bbox_close_iter,
                bbox_open_iter=bbox_open_iter,
                bbox_connectivity=bbox_connectivity,
                bbox_edge_min_fill_ratio=bbox_edge_min_fill_ratio,
                bbox_edge_cut_enabled=bbox_edge_cut_enabled,
                bbox_edge_cut_kernel=bbox_edge_cut_kernel,
                bbox_edge_cut_iter=bbox_edge_cut_iter,
                bbox_drop_band_fill_ratio=bbox_drop_band_fill_ratio,
                bbox_drop_band_width_ratio=bbox_drop_band_width_ratio,
                bbox_drop_band_height_ratio=bbox_drop_band_height_ratio,
                bbox_edge_roi_enabled=bbox_edge_roi_enabled,
            bbox_edge_roi_canny_low=bbox_edge_roi_canny_low,
            bbox_edge_roi_canny_high=bbox_edge_roi_canny_high,
            bbox_edge_roi_kernel=bbox_edge_roi_kernel,
            bbox_edge_roi_close_iter=bbox_edge_roi_close_iter,
            bbox_edge_roi_dilate_iter=bbox_edge_roi_dilate_iter,
            bbox_edge_roi_min_area_ratio=bbox_edge_roi_min_area_ratio,
            bbox_edge_roi_max_area_ratio=bbox_edge_roi_max_area_ratio,
            bbox_edge_roi_require_closed=bbox_edge_roi_require_closed,
            bbox_edge_roi_border_margin=bbox_edge_roi_border_margin,
            bbox_inner_erode_iter=bbox_inner_erode_iter,
            bbox_inner_overlap_ratio=bbox_inner_overlap_ratio,
            bbox_rescue_drop_band_fill_ratio=bbox_rescue_drop_band_fill_ratio,
            bbox_rescue_drop_band_width_ratio=bbox_rescue_drop_band_width_ratio,
            bbox_rescue_drop_band_height_ratio=bbox_rescue_drop_band_height_ratio,
            bbox_rescue_thresh_offset=bbox_rescue_thresh_offset,
            bbox_rescue_morph_kernel=bbox_rescue_morph_kernel,
            bbox_rescue_close_iter=bbox_rescue_close_iter,
            bbox_rescue_open_iter=bbox_rescue_open_iter,
            bbox_rescue_edge_erode_iter=bbox_rescue_edge_erode_iter,
            bbox_rescue_use_fg_edge_band=bbox_rescue_use_fg_edge_band,
            bbox_rescue_min_area_ratio=bbox_rescue_min_area_ratio,
            hole_mask_enabled=hole_mask_enabled,
            hole_mask_source=hole_mask_source,
            hole_mask_apply_mode=hole_mask_apply_mode,
            hole_mask_method=hole_mask_method,
            hole_bg_border_ratio=hole_bg_border_ratio,
            hole_bg_dist_percentile=hole_bg_dist_percentile,
            hole_bg_dist_scale=hole_bg_dist_scale,
            hole_bg_dist_min=hole_bg_dist_min,
            hole_bg_dist_max=hole_bg_dist_max,
            hole_inner_erode_iter=hole_inner_erode_iter,
            hole_expand_iter=hole_expand_iter,
            hole_bbox_filter_enabled=hole_bbox_filter_enabled,
            hole_bbox_filter_dilate_iter=hole_bbox_filter_dilate_iter,
            hole_bbox_filter_overlap_ratio=hole_bbox_filter_overlap_ratio,
            hole_bbox_filter_max_area_ratio=hole_bbox_filter_max_area_ratio,
            hole_bbox_filter_use_center=hole_bbox_filter_use_center,
            hole_edge_exclude_enabled=hole_edge_exclude_enabled,
            hole_edge_exclude_iter=hole_edge_exclude_iter,
            fg_hole_edge_exclude_enabled=fg_hole_edge_exclude_enabled,
            fg_hole_edge_exclude_dilate=fg_hole_edge_exclude_dilate,
            fg_hole_edge_exclude_min_area_ratio=fg_hole_edge_exclude_min_area_ratio,
            hole_min_area_ratio=hole_min_area_ratio,
            hole_max_area_ratio=hole_max_area_ratio,
            hole_min_fill_ratio=hole_min_fill_ratio,
            hole_aspect_min=hole_aspect_min,
            hole_aspect_max=hole_aspect_max,
            hole_shrink_iter=hole_shrink_iter,
            hole_max_total_ratio=hole_max_total_ratio,
            strong_bbox_enabled=config.get("strong_bbox_enabled", False),
                strong_bbox_percentile=config.get("strong_bbox_percentile", 99.5),
                strong_bbox_offset=config.get("strong_bbox_offset", 10),
                strong_bbox_min_area_ratio=config.get("strong_bbox_min_area_ratio", 0.3),
                strong_bbox_min_area=config.get("strong_bbox_min_area", 0),
                strong_bbox_morph_kernel=config.get("strong_bbox_morph_kernel", 5),
                strong_bbox_close_iter=config.get("strong_bbox_close_iter", 1),
                strong_bbox_open_iter=config.get("strong_bbox_open_iter", 0),
                strong_bbox_max_boxes=config.get("strong_bbox_max_boxes", 5),
                strong_bbox_use_edge_filter=config.get("strong_bbox_use_edge_filter", True),
                strong_bbox_edge_roi_enabled=strong_bbox_edge_roi_enabled,
                strong_bbox_edge_roi_canny_low=strong_bbox_edge_roi_canny_low,
                strong_bbox_edge_roi_canny_high=strong_bbox_edge_roi_canny_high,
                strong_bbox_edge_roi_kernel=strong_bbox_edge_roi_kernel,
                strong_bbox_edge_roi_close_iter=strong_bbox_edge_roi_close_iter,
                strong_bbox_edge_roi_dilate_iter=strong_bbox_edge_roi_dilate_iter,
                strong_bbox_edge_roi_min_area_ratio=strong_bbox_edge_roi_min_area_ratio,
                strong_bbox_edge_roi_max_area_ratio=strong_bbox_edge_roi_max_area_ratio,
                strong_bbox_edge_roi_require_closed=strong_bbox_edge_roi_require_closed,
                strong_bbox_edge_roi_border_margin=strong_bbox_edge_roi_border_margin,
                strong_bbox_drop_band_fill_ratio=strong_bbox_drop_band_fill_ratio,
                strong_bbox_drop_band_width_ratio=strong_bbox_drop_band_width_ratio,
                strong_bbox_drop_band_height_ratio=strong_bbox_drop_band_height_ratio,
                strong_bbox_debug_log_path=strong_bbox_debug_log_path,
                strong_bbox_min_diff_max=strong_bbox_min_diff_max,
                strong_bbox_edge_margin=strong_bbox_edge_margin,
                strong_bbox_min_fill=strong_bbox_min_fill,
                bbox_debug_log_path=bbox_debug_log_path,
                diff_original=diff_original,
                _save_stage=_save_stage,
                bbox_type_coloring=config.get("bbox_type_coloring", True),
            )

        # --- BBOX検出方式の分岐 ---
        _bbox_method = config.get("bbox_method", "legacy")

        if _bbox_method == "ensemble_sift" and processedA is not None and processedB is not None:
            # ============================================================
            # E+SIFT: Multi-Scale SSIM Ensemble + SIFT同一構造フィルタ
            # 従来の13ステップのワークアラウンドを5パラメータで置換
            # ============================================================
            _eh, _ew = processedA.shape[:2]
            _ethresh = config.get("ensemble_thresh", 70)
            _emin_area_cfg = config.get("ensemble_min_area", 200)
            _emin_area_ratio = float(config.get("ensemble_min_area_ratio", 0.003))
            _eimg_area = _eh * _ew
            _emin_area_scaled = max(30, int(_eimg_area * _emin_area_ratio))
            _emin_area = min(_emin_area_cfg, _emin_area_scaled)
            if _emin_area < _emin_area_cfg:
                print(f"  [E+SIFT AUTO] ensemble_min_area: {_emin_area_cfg} -> {_emin_area} "
                      f"(ratio={_emin_area_ratio}, area={_eimg_area})")
            _emin_dim = config.get("ensemble_min_dim", 6)

            # Phase 1: Multi-Scale SSIM ensemble
            # 短辺が小さい画像では1/4スケールがノイズになるため2スケール(1x, 1/2x)のみ使用
            _e2scale_min_side = int(config.get("ensemble_2scale_min_side", 250))
            _emin_side = min(_eh, _ew)
            _use_2scale = _emin_side < _e2scale_min_side

            _, _es1 = ssim(processedA, processedB, full=True, data_range=255.0)
            _ed1 = ((1.0 - np.clip(_es1, 0, 1)) * 255).astype(np.uint8)

            _ehA = cv2.resize(processedA, (_ew // 2, _eh // 2), interpolation=cv2.INTER_AREA)
            _ehB = cv2.resize(processedB, (_ew // 2, _eh // 2), interpolation=cv2.INTER_AREA)
            _ews2 = min(7, min(_ehA.shape[:2]) - 1)
            if _ews2 % 2 == 0:
                _ews2 -= 1
            _ews2 = max(3, _ews2)
            _, _es2 = ssim(_ehA, _ehB, full=True, win_size=_ews2, data_range=255.0)
            _ed2 = cv2.resize(((1.0 - np.clip(_es2, 0, 1)) * 255).astype(np.uint8),
                              (_ew, _eh), interpolation=cv2.INTER_LINEAR)

            if _use_2scale:
                _ensemble = np.minimum(_ed1, _ed2)
                print(f"  [E+SIFT 2-SCALE] min_side={_emin_side} < {_e2scale_min_side}, "
                      f"skipping 1/4 scale")
            else:
                _eqA = cv2.resize(processedA, (_ew // 4, _eh // 4), interpolation=cv2.INTER_AREA)
                _eqB = cv2.resize(processedB, (_ew // 4, _eh // 4), interpolation=cv2.INTER_AREA)
                _ews4 = min(7, min(_eqA.shape[:2]) - 1)
                if _ews4 % 2 == 0:
                    _ews4 -= 1
                _ews4 = max(3, _ews4)
                _, _es4 = ssim(_eqA, _eqB, full=True, win_size=_ews4, data_range=255.0)
                _ed4 = cv2.resize(((1.0 - np.clip(_es4, 0, 1)) * 255).astype(np.uint8),
                                  (_ew, _eh), interpolation=cv2.INTER_LINEAR)
                _ensemble = np.minimum(np.minimum(_ed1, _ed2), _ed4)
            if fg_mask is not None:
                _ensemble = cv2.bitwise_and(_ensemble, fg_mask)

            # 背景クロマキー: 画像端から連結する暗領域のみマスク（内部の穴は保護）
            _ebg_max = config.get("ensemble_bg_brightness_max", 30)
            if _ebg_max > 0:
                _ebg_dark = ((processedA <= _ebg_max) | (processedB <= _ebg_max)).astype(np.uint8) * 255
                _ebg_n, _ebg_labels, _ebg_stats, _ = cv2.connectedComponentsWithStats(
                    _ebg_dark, connectivity=8)
                _ebg_bg = np.zeros((_eh, _ew), dtype=np.uint8)
                for _ebi in range(1, _ebg_n):
                    _ebx, _eby, _ebbw, _ebbh, _ = _ebg_stats[_ebi]
                    # 画像端に接触する暗い成分 = 背景
                    if _ebx == 0 or _eby == 0 or (_ebx + _ebbw) >= _ew or (_eby + _ebbh) >= _eh:
                        _ebg_bg[_ebg_labels == _ebi] = 255
                # 境界の遷移帯をカバーするため軽くdilate
                _ebg_dilated = cv2.dilate(
                    _ebg_bg,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1)
                _ebg_count = int(np.count_nonzero(_ebg_dilated))
                if _ebg_count > 0:
                    _ensemble[_ebg_dilated > 0] = 0
                    print(f"  [E+SIFT BG CHROMA] {_ebg_count}px masked "
                          f"(brightness<={_ebg_max}, border-connected)")

            # 画像端帯のensemble差分を抑制（アライメント誤差対策）
            _edge_ratio = config.get("ensemble_edge_suppress_ratio", 0.0)
            _edge_max = config.get("ensemble_edge_suppress_max", 8)
            if _edge_ratio > 0:
                _edge_band = max(1, int(_edge_ratio * min(_eh, _ew)))
                if _edge_max > 0:
                    _edge_band = min(_edge_band, _edge_max)
                _ensemble[:_edge_band, :] = 0   # top
                _ensemble[-_edge_band:, :] = 0  # bottom
                _ensemble[:, :_edge_band] = 0   # left
                _ensemble[:, -_edge_band:] = 0  # right
                # 注: 表示用diff/ssim_mapには適用しない（ヒートマップの外周が不自然に消えるため）
                print(f"  [E+SIFT EDGE SUPPRESS] band={_edge_band}px "
                      f"(ratio={_edge_ratio}, base={min(_eh, _ew)}px)")

            _, _emask = cv2.threshold(_ensemble, _ethresh, 255, cv2.THRESH_BINARY)
            _ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _emask = cv2.morphologyEx(_emask, cv2.MORPH_CLOSE, _ek, iterations=1)

            # Phase 2: Connected components + アスペクト比分割
            _en, _elabels, _estats, _ = cv2.connectedComponentsWithStats(_emask, connectivity=4)
            _raw_bb = []
            for _ei in range(1, _en):
                _ebx, _eby, _ebw, _ebh, _earea = _estats[_ei]
                if _earea >= _emin_area and _ebw >= _emin_dim and _ebh >= _emin_dim:
                    _raw_bb.append((_ebx, _eby, _ebw, _ebh))

            _ebboxes = []
            for (_ebx, _eby, _ebw, _ebh) in _raw_bb:
                _easpect = max(_ebh / max(_ebw, 1), _ebw / max(_ebh, 1))
                if _easpect <= 2.0:
                    _ebboxes.append((_ebx, _eby, _ebw, _ebh))
                    continue
                # 縦長/横長BBOXを行/列スキャンで分割
                _eroi = _emask[_eby:_eby + _ebh, _ebx:_ebx + _ebw]
                _is_tall = _ebh > _ebw
                _efill = (np.count_nonzero(_eroi, axis=1) / max(_ebw, 1) if _is_tall
                          else np.count_nonzero(_eroi, axis=0) / max(_ebh, 1))
                _eactive = _efill > 0.3
                _egroups, _estart = [], None
                for _er in range(len(_eactive)):
                    if _eactive[_er]:
                        if _estart is None:
                            _estart = _er
                    else:
                        if _estart is not None:
                            _egroups.append((_estart, _er))
                            _estart = None
                if _estart is not None:
                    _egroups.append((_estart, len(_eactive)))
                for (_egs, _ege) in _egroups:
                    _esz = _ege - _egs
                    if _esz >= _emin_dim:
                        if _is_tall:
                            _esub = _eroi[_egs:_ege, :]
                            _ecols = np.any(_esub > 0, axis=0)
                            if np.any(_ecols):
                                _esx = int(np.argmax(_ecols))
                                _esw = len(_ecols) - int(np.argmax(_ecols[::-1])) - _esx
                                if _esw >= _emin_dim:
                                    _ebboxes.append((_ebx + _esx, _eby + _egs, _esw, _esz))
                        else:
                            _esub = _eroi[:, _egs:_ege]
                            _erows = np.any(_esub > 0, axis=1)
                            if np.any(_erows):
                                _esy = int(np.argmax(_erows))
                                _esh = len(_erows) - int(np.argmax(_erows[::-1])) - _esy
                                if _esh >= _emin_dim:
                                    _ebboxes.append((_ebx + _egs, _eby + _esy, _esz, _esh))

            # Phase 3: SIFT同一構造フィルタ
            _esift = cv2.SIFT_create(nfeatures=2000)
            _ekpA, _edesA = _esift.detectAndCompute(processedA, None)
            _ekpB, _edesB = _esift.detectAndCompute(processedB, None)
            _emin_kp = config.get("sift_min_kp", 5)
            _eratio_thresh = config.get("sift_ratio_thresh", 0.35)

            if (_edesA is not None and _edesB is not None
                    and len(_edesA) >= 2 and len(_edesB) >= 2):
                _ebf = cv2.BFMatcher(cv2.NORM_L2)
                _eraw_m = _ebf.knnMatch(_edesA, _edesB, k=2)
                _egood = [m for m, n2 in _eraw_m if m.distance < 0.75 * n2.distance]

                _efiltered = []
                for (_ebx, _eby, _ebw, _ebh) in _ebboxes:
                    _em = 10
                    _ex1, _ey1 = max(0, _ebx - _em), max(0, _eby - _em)
                    _ex2, _ey2 = _ebx + _ebw + _em, _eby + _ebh + _em
                    _ekA = set(i for i, kp in enumerate(_ekpA)
                               if _ex1 <= kp.pt[0] <= _ex2 and _ey1 <= kp.pt[1] <= _ey2)
                    _ekB = set(i for i, kp in enumerate(_ekpB)
                               if _ex1 <= kp.pt[0] <= _ex2 and _ey1 <= kp.pt[1] <= _ey2)
                    _em_both = [m for m in _egood if m.queryIdx in _ekA and m.trainIdx in _ekB]
                    _en_A = len(_ekA)
                    _er = len(_em_both) / max(_en_A, 1)
                    if _en_A >= _emin_kp and _er > _eratio_thresh:
                        print(f"  [SIFT DROP] bbox=({_ebx},{_eby},{_ebw},{_ebh}) kp={_en_A} ratio={_er:.2f}")
                    else:
                        _efiltered.append((_ebx, _eby, _ebw, _ebh))
                _ebboxes = _efiltered

            # Phase 4: BBOX内ensemble平均フィルタ（薄いアーティファクト除去）
            _ebbox_min_mean = config.get("ensemble_bbox_min_mean", 60)
            _efinal = []
            for (_ebx, _eby, _ebw, _ebh) in _ebboxes:
                _eroi = _ensemble[_eby:_eby + _ebh, _ebx:_ebx + _ebw]
                _emean = float(np.mean(_eroi))
                if _emean < _ebbox_min_mean:
                    print(f"  [ENS MEAN DROP] bbox=({_ebx},{_eby},{_ebw},{_ebh}) mean={_emean:.1f} < {_ebbox_min_mean}")
                else:
                    _efinal.append((_ebx, _eby, _ebw, _ebh))

            bboxes = _efinal
            strong_bboxes = []
            mask = _emask

            # E+SIFT用 性質別マスク生成（BBOX色分け用）
            _bbox_type_coloring = config.get("bbox_type_coloring", True)
            if _bbox_type_coloring:
                # エッジ差分 = shape（形状変化）、非エッジ差分 = surface（表面異常）
                _edge_A = cv2.Canny(processedA, bbox_edge_roi_canny_low, bbox_edge_roi_canny_high)
                _edge_B = cv2.Canny(processedB, bbox_edge_roi_canny_low, bbox_edge_roi_canny_high)
                _edge_diff = cv2.bitwise_or(_edge_A, _edge_B)
                _edge_dilated = cv2.dilate(
                    _edge_diff,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                # shape: ensemble差分 AND エッジ領域
                _etype_shape = cv2.bitwise_and(_emask, _edge_dilated)
                # surface: ensemble差分 AND 非エッジ領域
                _etype_surface = cv2.bitwise_and(_emask, cv2.bitwise_not(_edge_dilated))
                self._type_shape_mask = _etype_shape
                self._type_surface_mask = _etype_surface
            else:
                self._type_shape_mask = None
                self._type_surface_mask = None

            print(f"[E+SIFT] {len(bboxes)} BBOX detected")

        else:
            # ============================================================
            # Legacy: 従来のBBOX後処理パイプライン
            # ============================================================
            # ヒートマップ二値化: diff>200の強信号のみ表示（散在ノイズ除去、表示用）
            if diff is not None and mask is not None:
                _strong = (diff > 200).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask, _strong)

            # 近接BBOXのマージ（1要素1BBOXにする）— strong追加前に実行
            if bbox_merge_distance > 0 and len(bboxes) > 1:
                bboxes = merge_nearby_bboxes(bboxes, distance_thresh=bbox_merge_distance)

            if strong_bboxes:
                def _bbox_iou(a, b):
                    ax, ay, aw, ah = a
                    bx, by, bw, bh = b
                    ix1, iy1 = max(ax, bx), max(ay, by)
                    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
                    if ix1 >= ix2 or iy1 >= iy2:
                        return 0.0
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = aw * ah + bw * bh - inter
                    return inter / union if union > 0 else 0.0

                for bb in strong_bboxes:
                    covered = False
                    bx, by, bw, bh = bb
                    for existing in bboxes:
                        if _bbox_iou(bb, existing) >= 0.1:
                            covered = True
                            break
                        ex, ey, ew, eh = existing
                        d = bbox_merge_distance
                        if not (bx - d >= ex + ew or ex >= bx + bw + d or
                                by - d >= ey + eh or ey >= by + bh + d):
                            covered = True
                            break
                    if not covered and tuple(bb) not in {tuple(b) for b in bboxes}:
                        bboxes.append(bb)

            # BBOX上方拡張
            if diff is not None and len(bboxes) > 0:
                _grow_thresh = float(diff_thresh)
                _img_h = diff.shape[0]
                _grow_y_min = int(_img_h * 0.75)
                _grown = []
                for (bx, by, bw, bh) in bboxes:
                    if by >= _grow_y_min:
                        _min_px = max(5, int(bw * 0.25))
                        ny = by
                        for gy in range(by - 1, max(0, by - 40) - 1, -1):
                            row = diff[gy, bx:bx + bw]
                            if np.count_nonzero(row > _grow_thresh) >= _min_px:
                                ny = gy
                            else:
                                break
                        _grown.append((bx, ny, bw, bh + (by - ny)))
                    else:
                        _grown.append((bx, by, bw, bh))
                bboxes = _grown

            # BBOX tight-fit
            _strong_diff_thresh = 200
            if diff is not None and len(bboxes) > 0:
                _tight = []
                for (bx, by, bw, bh) in bboxes:
                    roi = diff[by:by + bh, bx:bx + bw]
                    strong_mask = roi > _strong_diff_thresh
                    if np.any(strong_mask):
                        ys, xs = np.where(strong_mask)
                        nx = bx + int(xs.min())
                        ny = by + int(ys.min())
                        nx2 = bx + int(xs.max())
                        ny2 = by + int(ys.max())
                        margin = max(2, int(min(bw, bh) * 0.1))
                        nx = max(0, nx - margin)
                        ny = max(0, ny - margin)
                        nx2 = min(diff.shape[1] - 1, nx2 + margin)
                        ny2 = min(diff.shape[0] - 1, ny2 + margin)
                        _tight.append((nx, ny, nx2 - nx + 1, ny2 - ny + 1))
                    else:
                        pass
                bboxes = _tight

            # 粗差分フィルタ
            _coarse_scale = 0.25
            if diff is not None and len(bboxes) > 0:
                _ch, _cw = processedA.shape[:2]
                _cnw, _cnh = int(_cw * _coarse_scale), int(_ch * _coarse_scale)
                if _cnw >= 8 and _cnh >= 8:
                    _csA = cv2.resize(processedA, (_cnw, _cnh), interpolation=cv2.INTER_AREA)
                    _csB = cv2.resize(processedB, (_cnw, _cnh), interpolation=cv2.INTER_AREA)
                    _cws = max(3, min(7, _cnh // 10 * 2 + 1))
                    if _cws % 2 == 0:
                        _cws += 1
                    _, _csmap = ssim(_csA, _csB, full=True, win_size=_cws, data_range=255.0)
                    _cdiff = ((1.0 - np.clip(_csmap, 0.0, 1.0)) * 255).astype(np.uint8)
                    _cdiff_up = cv2.resize(_cdiff, (_cw, _ch), interpolation=cv2.INTER_NEAREST)
                    _coarse_thresh = 80
                    _coarse_kept = []
                    for (bx, by, bw, bh) in bboxes:
                        _croi = _cdiff_up[by:by + bh, bx:bx + bw]
                        _cmean = float(_croi.mean()) if _croi.size > 0 else 0
                        if _cmean >= _coarse_thresh:
                            _coarse_kept.append((bx, by, bw, bh))
                    if len(_coarse_kept) > 0:
                        bboxes = _coarse_kept

        _save_stage("11_mask", single=mask)

        # BBOX描画（性質別色分け）
        visA = imageA.copy()
        visB = imageB.copy()
        _type_shape = type_masks.get("shape") if type_masks else None
        _type_surface = type_masks.get("surface") if type_masks else None

        def _classify_bbox(x, y, w, h):
            """BBOXの性質を判定: 'shape'(形状変化) / 'surface'(傷・汚れ) / 'mixed'(混合)"""
            if _type_shape is None or _type_surface is None:
                return "surface"
            roi_shape = _type_shape[y:y+h, x:x+w]
            roi_surface = _type_surface[y:y+h, x:x+w]
            shape_px = int(np.count_nonzero(roi_shape))
            surface_px = int(np.count_nonzero(roi_surface))
            if shape_px + surface_px == 0:
                return "surface"
            ratio = shape_px / (shape_px + surface_px)
            if ratio > 0.7:
                return "shape"
            if ratio < 0.3:
                return "surface"
            return "mixed"

        _type_colors = {
            "surface": (0, 0, 255),    # 赤: 傷・汚れ
            "shape":   (0, 255, 0),    # 緑: 形状変化
            "mixed":   (0, 255, 255),  # 黄: 混合
        }

        # BBOX描画 + アノテーション（マージ後の最終リストで描画）
        _strong_set = {tuple(b) for b in strong_bboxes} if strong_bboxes else set()
        _img_h, _img_w = visB.shape[:2]
        _ann_font = cv2.FONT_HERSHEY_SIMPLEX
        _ann_scale = max(0.35, min(_img_w / 800.0, 0.6))
        _ann_thick = max(1, int(_ann_scale * 2))
        for _bi, (x, y, w, h) in enumerate(bboxes):
            btype = _classify_bbox(x, y, w, h)
            color = _type_colors.get(btype, (0, 0, 255))
            thick = 5 if (x, y, w, h) in _strong_set else 3
            cv2.rectangle(visA, (x, y), (x + w, y + h), color, thick)
            cv2.rectangle(visB, (x, y), (x + w, y + h), color, thick)
            # 位置ラベル: #番号 L/R
            _cx = x + w / 2.0
            _side = "L" if _cx < _img_w / 2.0 else "R"
            _label = f"#{_bi}{_side}"
            (_ltw, _lth), _ = cv2.getTextSize(_label, _ann_font, _ann_scale, _ann_thick)
            # ラベル位置: BBOX上辺の上、はみ出す場合はBBOX内
            _lx = x
            _ly = y - 4 if y - _lth - 4 > 0 else y + _lth + 4
            cv2.rectangle(visB, (_lx - 1, _ly - _lth - 1), (_lx + _ltw + 1, _ly + 1), (0, 0, 0), -1)
            cv2.putText(visB, _label, (_lx, _ly), _ann_font, _ann_scale, (255, 255, 255), _ann_thick, cv2.LINE_AA)

        # 左右反転で画角合わせした場合、BBOX画像に注記を追加
        _crop_flipped = crop_info.get("flipped", False) if isinstance(crop_info, dict) else False
        if _crop_flipped:
            _flip_text = "FLIPPED (L/R)"
            _font = cv2.FONT_HERSHEY_SIMPLEX
            _font_scale = max(0.6, min(visB.shape[1] / 500.0, 1.5))
            _thickness = max(1, int(_font_scale * 2))
            (_tw, _th), _ = cv2.getTextSize(_flip_text, _font, _font_scale, _thickness)
            _tx = visB.shape[1] - _tw - 10
            _ty = _th + 10
            # 背景矩形
            cv2.rectangle(visB, (_tx - 5, _ty - _th - 5), (_tx + _tw + 5, _ty + 5), (0, 0, 0), -1)
            cv2.putText(visB, _flip_text, (_tx, _ty), _font, _font_scale, (0, 200, 255), _thickness, cv2.LINE_AA)

        # 構造比較の分類結果をBBOX画像に注記
        if structural_info is not None and structural_info.get("classification") not in (None, "normal"):
            _sc_class = structural_info["classification"]
            _sc_msg = structural_info.get("message", _sc_class)
            _sc_colors = {
                "symmetry": (255, 200, 0),       # シアン系
                "different_part": (0, 0, 255),    # 赤
                "structural_deficiency": (0, 128, 255),  # オレンジ
                "major_deformation": (0, 128, 255),      # オレンジ
                "poor_match": (0, 200, 255),      # 黄
            }
            _sc_color = _sc_colors.get(_sc_class, (255, 255, 255))
            _sc_font = cv2.FONT_HERSHEY_SIMPLEX
            _sc_font_scale = max(0.5, min(visB.shape[1] / 600.0, 1.2))
            _sc_thickness = max(1, int(_sc_font_scale * 2))
            _sc_label = f"[{_sc_class.upper()}]"
            (_sc_tw, _sc_th), _ = cv2.getTextSize(_sc_label, _sc_font, _sc_font_scale, _sc_thickness)
            _sc_tx = 10
            _sc_ty = visB.shape[0] - 10
            # 背景矩形
            cv2.rectangle(visB, (_sc_tx - 5, _sc_ty - _sc_th - 5), (_sc_tx + _sc_tw + 5, _sc_ty + 5), (0, 0, 0), -1)
            cv2.putText(visB, _sc_label, (_sc_tx, _sc_ty), _sc_font, _sc_font_scale, _sc_color, _sc_thickness, cv2.LINE_AA)
            # visAにも同じラベル
            cv2.rectangle(visA, (_sc_tx - 5, _sc_ty - _sc_th - 5), (_sc_tx + _sc_tw + 5, _sc_ty + 5), (0, 0, 0), -1)
            cv2.putText(visA, _sc_label, (_sc_tx, _sc_ty), _sc_font, _sc_font_scale, _sc_color, _sc_thickness, cv2.LINE_AA)

        # --- 内部平面差分検出（SSIMマップ + fg_mask内側 → 緑BBOX） ---
        surface_bboxes = []
        if surface_detection_enabled and fg_mask is not None and not quality_info.get("blocked"):
            _simg_area = diff_original.shape[0] * diff_original.shape[1]
            # min_area_ratio > 0 なら画像サイズ比で自動計算（scratch と同方式）
            _surf_min_area = surface_min_area
            if surface_min_area_ratio > 0 and _simg_area > 0:
                _surf_min_area = max(1, int(_simg_area * surface_min_area_ratio))
                print(f"  [Surface auto] min_area: ratio={surface_min_area_ratio} -> {_surf_min_area}px (area={_simg_area})")
            _isk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _inner_mask = cv2.erode(fg_mask, _isk, iterations=surface_inner_erode_iter)
            # マスター画像のエッジで非平面部（ブラケット枠・ネジ・突起）を除外 → 平面のみに絞る
            if surface_flat_exclude_enabled:
                _master_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) if len(imageA.shape) == 3 else imageA
                _master_blur = cv2.GaussianBlur(_master_gray, (5, 5), 0)
                _master_edges = cv2.Canny(_master_blur, surface_flat_canny_low, surface_flat_canny_high)
                _edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                _master_edges_dilated = cv2.dilate(_master_edges, _edge_k, iterations=surface_flat_edge_dilate_iter)
                # 平面マスク = 内側マスク AND NOT エッジ近傍
                _flat_mask = cv2.bitwise_and(_inner_mask, cv2.bitwise_not(_master_edges_dilated))
                _flat_px = int(np.count_nonzero(_flat_mask))
                print(f"  [Surface flat] edge除外後の平面マスク: {_flat_px}px")
            else:
                _flat_mask = _inner_mask
            # diff_original: fg_mask/erode適用前の元差分（グレースケール or BGR）
            _sdiff = diff_original if len(diff_original.shape) == 2 else cv2.cvtColor(diff_original, cv2.COLOR_BGR2GRAY)
            _sdiff = cv2.bitwise_and(_sdiff, _flat_mask)
            _, _sbin = cv2.threshold(_sdiff, surface_diff_thresh, 255, cv2.THRESH_BINARY)
            _smk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            _sbin = cv2.morphologyEx(_sbin, cv2.MORPH_OPEN, _smk, iterations=1)
            _sbin = cv2.morphologyEx(_sbin, cv2.MORPH_CLOSE, _smk, iterations=2)
            _smax_area = int(_simg_area * surface_max_area_ratio / 100.0)
            _sn, _, _sstats, _ = cv2.connectedComponentsWithStats(_sbin, connectivity=8)
            _slist = []
            for _li in range(1, _sn):
                _sx, _sy, _sw, _sh, _sarea = _sstats[_li]
                if _sarea < _surf_min_area:
                    continue
                if _sarea > _smax_area:
                    continue
                _slist.append((_sx, _sy, _sw, _sh, _sarea))
            _slist.sort(key=lambda s: s[4], reverse=True)
            if len(_slist) > surface_max_count:
                _slist = _slist[:surface_max_count]
            surface_bboxes = [(_sx, _sy, _sw, _sh) for (_sx, _sy, _sw, _sh, _) in _slist]
            for (sx, sy, sw, sh) in surface_bboxes:
                cv2.rectangle(visA, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.rectangle(visB, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            print(f"[Surface] 内部差分BBOX数: {len(surface_bboxes)}")

        _save_stage("12_bbox", visA, visB)

        print("=" * 60)
        print(f"【{title}】")
        print("=" * 60)
        if _crop_flipped:
            print("※ 比較画像は左右反転して画角合わせしています")
        if structural_info is not None and structural_info.get("classification") not in (None, "normal"):
            _sc_class = structural_info["classification"]
            _sc_msg = structural_info.get("message", _sc_class)
            print(f"★ 構造比較結果: {_sc_msg} (classification={_sc_class})")
            if structural_info.get("ssim_raw") is not None:
                print(f"  SSIM_raw={structural_info['ssim_raw']:.4f}", end="")
                if structural_info.get("ssim_raw_flipped") is not None:
                    print(f", SSIM_raw_flipped={structural_info['ssim_raw_flipped']:.4f}", end="")
                if structural_info.get("iou") is not None:
                    print(f", IoU={structural_info['iou']:.4f}", end="")
                print()
            if structural_info.get("supplementary"):
                print(f"  ⚠ {structural_info['supplementary']}")
            if structural_info.get("block_bbox"):
                print("  → BBOX検出はブロックされています")
        print(f"MSE: {mse_value:.2f}（0に近いほど類似）")
        print(f"SSIM: {ssim_score:.4f}（1に近いほど類似）")
        print(f"検出BBOX数（上限{max_boxes}）: {len(bboxes)}")
        print(f"diff_thresh={diff_thresh}, min_area={min_area}, morph_kernel={morph_kernel}")
        print("=" * 60)

        # --- ピンアート外形照合 ---
        pin_info = None
        if pin_profile_enabled and not quality_info.get("blocked"):
            from src.core.pin_profile import (
                extract_pin_profile, compare_pin_profiles, visualize_pin_compare,
            )
            from src.core.segmentation import get_foreground_mask as _get_fg

            # マスター前景マスク: fg_mask が既に計算済みならそれを使う
            _pin_master_mask = fg_mask
            if _pin_master_mask is None:
                _pin_master_mask = _get_fg(imageA, blur_ksize=5)
            # テスト前景マスク: 常に別途計算
            _pin_test_mask = _get_fg(imageB, blur_ksize=5)

            if _pin_master_mask is not None and _pin_test_mask is not None:
                master_prof = extract_pin_profile(
                    _pin_master_mask,
                    noise_erode=pin_noise_erode,
                    anchor_band_min_length=pin_anchor_band_min_length,
                )
                test_prof = extract_pin_profile(
                    _pin_test_mask,
                    noise_erode=pin_noise_erode,
                    anchor_band_min_length=pin_anchor_band_min_length,
                )
                pin_result = compare_pin_profiles(
                    master_prof, test_prof,
                    burr_threshold=pin_burr_threshold,
                    chip_threshold=pin_chip_threshold,
                )
                # visB にピンアート比較結果を重ねて描画（fig 生成前なので panel 2 に反映される）
                visB = visualize_pin_compare(visB, master_prof, pin_result)
                _save_stage("15_pin_profile", visA, visB)

                _burr_count = (
                    len(pin_result.burr_rows_left) + len(pin_result.burr_rows_right)
                    + len(pin_result.burr_cols_top) + len(pin_result.burr_cols_bottom)
                )
                _chip_count = (
                    len(pin_result.chip_rows_left) + len(pin_result.chip_rows_right)
                    + len(pin_result.chip_cols_top) + len(pin_result.chip_cols_bottom)
                )
                pin_info = {
                    "enabled": True,
                    "scores": pin_result.scores,
                    "align_offset": pin_result.align_offset,
                    "burr_count": _burr_count,
                    "chip_count": _chip_count,
                }
                print(
                    f"ピンアート照合: match_rate={pin_result.scores.get('match_rate', 0):.3f}, "
                    f"max_dev={pin_result.scores.get('max_deviation', 0)}px, "
                    f"バリ={_burr_count}行列, 欠け={_chip_count}行列"
                )

        # 結果の可視化
        fig = self._generate_result_figure(
            visA, visB, mask, diff, ssim_map, bboxes, title, ssim_score, mse_value, max_boxes, show_plot
        )

        # --- 傷検出 ---
        scratch_info = None
        if scratch_detection_enabled and not quality_info.get("blocked"):
            from src.core.scratch import detect_scratches
            _scratch_debug_dir = None
            if _pipe_dir is not None:
                _scratch_debug_dir = os.path.join(_pipe_dir, "scratch")
                os.makedirs(_scratch_debug_dir, exist_ok=True)
            scratch_result = detect_scratches(
                imageA, imageB,
                blur_ksize=scratch_blur_ksize,
                blackhat_kernel_size=scratch_blackhat_kernel,
                diff_thresh=scratch_diff_thresh,
                min_area=scratch_min_area,
                min_width=scratch_min_width,
                morph_kernel_size=scratch_morph_kernel,
                morph_open_iter=scratch_morph_open_iter,
                morph_close_iter=scratch_morph_close_iter,
                max_scratches=scratch_max_count,
                min_aspect_ratio=scratch_min_aspect_ratio,
                max_area_ratio=scratch_max_area_ratio,
                fg_mask=fg_mask,
                fg_mask_erode_iter=scratch_fg_mask_erode_iter,
                debug_dir=_scratch_debug_dir,
            )
            scratch_bboxes = scratch_result["scratches"]
            scratch_info = {
                "enabled": True,
                "bboxes": scratch_bboxes,
                "metrics": scratch_result["metrics"],
            }
            # 傷BBOXをvisA/visBに黄色で描画
            for (sx, sy, sw, sh) in scratch_bboxes:
                cv2.rectangle(visA, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                cv2.rectangle(visB, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
            if scratch_bboxes:
                _save_stage("13_scratch_bbox", visA, visB)
                print(f"傷検出BBOX数: {len(scratch_bboxes)}")
                for i, (sx, sy, sw, sh) in enumerate(scratch_bboxes):
                    print(f"  傷[{i}]: x={sx} y={sy} w={sw} h={sh}")

        # --- 輪郭差分検出 ---
        contour_info = None
        if contour_detection_enabled and not quality_info.get("blocked"):
            from src.core.contour_diff import detect_contour_diff
            _contour_debug_dir = None
            if _pipe_dir is not None:
                _contour_debug_dir = os.path.join(_pipe_dir, "contour")
                os.makedirs(_contour_debug_dir, exist_ok=True)
            contour_result = detect_contour_diff(
                imageA, imageB,
                blur_ksize=contour_blur_ksize,
                canny_low=contour_canny_low,
                canny_high=contour_canny_high,
                diff_thresh=contour_diff_thresh,
                min_area=contour_min_area,
                morph_kernel_size=contour_morph_kernel,
                morph_close_iter=contour_morph_close_iter,
                max_boxes=contour_max_count,
                fg_mask=fg_mask,
                fg_mask_dilate_iter=contour_fg_mask_dilate_iter,
                edge_band_width=contour_edge_band_width,
                min_length=contour_min_length,
                debug_dir=_contour_debug_dir,
            )
            contour_bboxes = contour_result["contours"]
            contour_info = {
                "enabled": True,
                "bboxes": contour_bboxes,
                "metrics": contour_result["metrics"],
            }
            # 輪郭BBOXをvisA/visBにマゼンタで描画
            for (cx, cy, cw, ch) in contour_bboxes:
                cv2.rectangle(visA, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 2)
                cv2.rectangle(visB, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 2)
            if contour_bboxes:
                _save_stage("14_contour_bbox", visA, visB)
                print(f"輪郭差分BBOX数: {len(contour_bboxes)}")
                for i, (cx, cy, cw, ch) in enumerate(contour_bboxes):
                    print(f"  輪郭[{i}]: x={cx} y={cy} w={cw} h={ch}")

        # 直近の強差分BBOXを保持（バッチ評価などで使用）
        self._last_strong_bboxes = strong_bboxes
        flip_info = {"enabled": False}
        return mse_value, ssim_score, diff, mask, bboxes, fig, align_info, crop_info, roi_info, flip_info, quality_info, structural_info, scratch_info, contour_info, pin_info

    def _structural_comparison(self, imageA, imageB,
                               ssim_normal_thresh=0.70,
                               ssim_flip_thresh=0.65,
                               ssim_flip_min_improvement=0.10,
                               iou_different_part=0.30,
                               iou_deficient=0.60,
                               overlap_deficiency=0.50,
                               block_bbox_on_fail=True,
                               _pipe_dir=None):
        """
        構造比較: 通常SSIM → flip → 前景マスクIoU の3段階判定。

        Returns:
            (imageA, imageB, structural_info)
            structural_info: {
                "enabled": True,
                "classification": "normal"|"symmetry"|"different_part"|"structural_deficiency"|"major_deformation"|"poor_match",
                "ssim_raw": float,
                "ssim_raw_flipped": float or None,
                "iou": float or None,
                "overlap_ratio": float or None,
                "block_bbox": bool,
                "message": str,
            }
        """
        def _save_structural(name, imgA=None, imgB=None, single=None):
            if _pipe_dir is None:
                return
            try:
                if single is not None:
                    safe_imwrite(os.path.join(_pipe_dir, f"{name}.png"), single)
                else:
                    if imgA is not None:
                        safe_imwrite(os.path.join(_pipe_dir, f"{name}_A.png"), imgA)
                    if imgB is not None:
                        safe_imwrite(os.path.join(_pipe_dir, f"{name}_B.png"), imgB)
            except Exception:
                pass

        # --- Step 1: 通常のSSIM_raw ---
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) if len(imageA.shape) == 3 else imageA
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) if len(imageB.shape) == 3 else imageB
        ssim_raw, _ = ssim(grayA, grayB, full=True)
        print(f"[StructuralComparison] Step1: SSIM_raw={ssim_raw:.4f} (thresh={ssim_normal_thresh})")

        if ssim_raw >= ssim_normal_thresh:
            print(f"[StructuralComparison] → classification=normal (SSIM_raw >= {ssim_normal_thresh})")
            return imageA, imageB, {
                "enabled": True,
                "classification": "normal",
                "ssim_raw": float(ssim_raw),
                "ssim_raw_flipped": None,
                "iou": None,
                "overlap_ratio": None,
                "block_bbox": False,
                "message": "通常比較（SSIM十分）",
                "supplementary": None,
            }

        # --- Step 2: 左右反転してSSIM_raw再計算 ---
        imageB_flip = np.ascontiguousarray(cv2.flip(imageB.copy(), 1))
        # サイズ揃え
        if imageB_flip.shape[:2] != imageA.shape[:2]:
            imageB_flip = cv2.resize(imageB_flip, (imageA.shape[1], imageA.shape[0]), interpolation=cv2.INTER_LINEAR)
        grayB_flip = cv2.cvtColor(imageB_flip, cv2.COLOR_BGR2GRAY) if len(imageB_flip.shape) == 3 else imageB_flip
        ssim_raw_flipped, _ = ssim(grayA, grayB_flip, full=True)
        improvement = ssim_raw_flipped - ssim_raw
        print(f"[StructuralComparison] Step2: SSIM_raw_flipped={ssim_raw_flipped:.4f} (thresh={ssim_flip_thresh}, improvement={improvement:.4f}, min_improvement={ssim_flip_min_improvement})")

        _save_structural("05b_structural_flip", imageA, imageB_flip)

        if ssim_raw_flipped >= ssim_flip_thresh and improvement >= ssim_flip_min_improvement:
            print(f"[StructuralComparison] → classification=symmetry (線対称部品 → 部品違いエラー)")
            return imageA, imageB, {
                "enabled": True,
                "classification": "symmetry",
                "ssim_raw": float(ssim_raw),
                "ssim_raw_flipped": float(ssim_raw_flipped),
                "iou": None,
                "overlap_ratio": None,
                "block_bbox": True,
                "message": "部品が裏表（左右逆）です。正しい向きの部品を使用してください。",
                "supplementary": None,
            }

        # --- Step 3: 前景マスクIoUで構造判定 ---
        from src.core.segmentation import get_foreground_mask
        fg_a = get_foreground_mask(imageA)
        fg_b = get_foreground_mask(imageB)

        if fg_a is None or fg_b is None:
            # マスク取得失敗 → poor_match扱い
            print(f"[StructuralComparison] Step3: 前景マスク取得失敗 → poor_match")
            return imageA, imageB, {
                "enabled": True,
                "classification": "poor_match",
                "ssim_raw": float(ssim_raw),
                "ssim_raw_flipped": float(ssim_raw_flipped),
                "iou": None,
                "overlap_ratio": None,
                "block_bbox": False,
                "message": "マッチング品質が低い（前景マスク取得失敗）",
            }

        # サイズ揃え（念のため）
        if fg_b.shape[:2] != fg_a.shape[:2]:
            fg_b = cv2.resize(fg_b, (fg_a.shape[1], fg_a.shape[0]), interpolation=cv2.INTER_NEAREST)

        intersection = cv2.bitwise_and(fg_a, fg_b)
        union = cv2.bitwise_or(fg_a, fg_b)
        n_intersection = cv2.countNonZero(intersection)
        n_union = cv2.countNonZero(union)
        n_fg_a = cv2.countNonZero(fg_a)

        iou = n_intersection / max(n_union, 1)
        overlap_ratio = n_intersection / max(n_fg_a, 1)
        print(f"[StructuralComparison] Step3: IoU={iou:.4f}, overlap_ratio={overlap_ratio:.4f}")

        # 前景マスクをデバッグ保存
        _save_structural("05b_structural_fg_master", single=fg_a)
        _save_structural("05b_structural_fg_compare", single=fg_b)
        _save_structural("05b_structural_fg_intersection", single=intersection)

        if iou < iou_different_part:
            classification = "different_part"
            message = "別部品です（形状が大きく異なる）"
            supplementary = "画角不足の可能性があります。撮影画像を確認してください。"
            block_bbox = block_bbox_on_fail
        elif iou < iou_deficient:
            if overlap_ratio < overlap_deficiency:
                classification = "structural_deficiency"
                message = "構造的に欠けています"
                supplementary = "画角不足の可能性があります。撮影画像を確認してください。"
                block_bbox = block_bbox_on_fail
            else:
                classification = "major_deformation"
                message = "大きな変形があります"
                supplementary = None
                block_bbox = block_bbox_on_fail
        else:
            classification = "poor_match"
            message = "マッチング品質が低い"
            supplementary = None
            block_bbox = False

        print(f"[StructuralComparison] → classification={classification} (IoU={iou:.4f}, overlap={overlap_ratio:.4f})")

        return imageA, imageB, {
            "enabled": True,
            "classification": classification,
            "ssim_raw": float(ssim_raw),
            "ssim_raw_flipped": float(ssim_raw_flipped),
            "iou": float(iou),
            "overlap_ratio": float(overlap_ratio),
            "block_bbox": bool(block_bbox),
            "message": message,
            "supplementary": supplementary,
        }

    def _handle_flip_compare(self, imageA, imageB, config, flip_compare_min_ssim=None, flip_compare_min_improvement=None):
        """左右反転比較の処理"""
        flip_B = np.ascontiguousarray(cv2.flip(imageB.copy(), 1))

        # 再帰的にcompare_imagesを呼び出す（use_flip_compare=False にして無限ループ防止）
        config_no_flip = config.copy()
        config_no_flip["use_flip_compare"] = False

        # パイプラインを2回実行
        r1 = self.run(imageA, imageB, **config_no_flip)
        r2 = self.run(imageA, flip_B, **config_no_flip)

        ssim1, ssim2 = r1[1], r2[1]
        improvement = ssim2 - ssim1
        min_ssim_ok = (flip_compare_min_ssim is None) or (ssim2 >= float(flip_compare_min_ssim))
        min_improve_ok = (flip_compare_min_improvement is None) or (improvement >= float(flip_compare_min_improvement))
        use_flip = (ssim2 > ssim1) and min_ssim_ok and min_improve_ok
        if use_flip:
            flip_info = {"enabled": True, "used_flip": True, "ssim_original": float(ssim1), "ssim_flipped": float(ssim2)}
            quality_info = r2[10] if len(r2) > 10 else None
            structural_info = r2[11] if len(r2) > 11 else None
            scratch_info = r2[12] if len(r2) > 12 else None
            contour_info = r2[13] if len(r2) > 13 else None
            pin_info = r2[14] if len(r2) > 14 else None
            return (r2[0], r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], r2[7], r2[8], flip_info, quality_info, structural_info, scratch_info, contour_info, pin_info)
        flip_info = {
            "enabled": True,
            "used_flip": False,
            "ssim_original": float(ssim1),
            "ssim_flipped": float(ssim2),
            "flip_improvement": float(improvement),
            "min_ssim_ok": bool(min_ssim_ok),
            "min_improve_ok": bool(min_improve_ok),
        }
        quality_info = r1[10] if len(r1) > 10 else None
        structural_info = r1[11] if len(r1) > 11 else None
        scratch_info = r1[12] if len(r1) > 12 else None
        contour_info = r1[13] if len(r1) > 13 else None
        pin_info = r1[14] if len(r1) > 14 else None
        return (r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8], flip_info, quality_info, structural_info, scratch_info, contour_info, pin_info)

    def _crop_to_fov(self, imageA, imageB, **kwargs):
        """画角切り出しの処理"""
        # この関数は非常に長いため、main.pyから直接インポートする
        # crop_to_master_fov関数を使用する
        from src.core.crop import crop_to_master_fov

        align_method = kwargs.get("align_method", "orb")
        align_max_features = kwargs.get("align_max_features", 500)
        align_match_ratio = kwargs.get("align_match_ratio", 0.75)
        crop_compare_mode = kwargs.get("crop_compare_mode", True)
        crop_two_pass = kwargs.get("crop_two_pass", False)
        crop_scale_min = kwargs.get("crop_scale_min", 0.3)
        crop_scale_step = kwargs.get("crop_scale_step", 0.05)
        crop_two_pass_min_matches = kwargs.get("crop_two_pass_min_matches", 10)
        crop_ssim_degradation_threshold = kwargs.get("crop_ssim_degradation_threshold", -0.02)
        crop_ssim_min = kwargs.get("crop_ssim_min", 0.3)
        crop_ssim_min_skip_matches = kwargs.get("crop_ssim_min_skip_matches", 20)
        crop_adopt_min_matches = kwargs.get("crop_adopt_min_matches", 8)
        crop_match_ratio_min = kwargs.get("crop_match_ratio_min", 1.0)
        crop_mse_degradation_threshold = kwargs.get("crop_mse_degradation_threshold", 0.05)
        crop_min_area_ratio = kwargs.get("crop_min_area_ratio", 1.5)
        crop_max_rotation_deg = kwargs.get("crop_max_rotation_deg", 15.0)
        crop_min_inlier_ratio = kwargs.get("crop_min_inlier_ratio", 0.3)
        crop_try_rotations = kwargs.get("crop_try_rotations", False)
        crop_rotation_search_step_deg = kwargs.get("crop_rotation_search_step_deg", 90)
        crop_master_padding_ratio = kwargs.get("crop_master_padding_ratio", 0)
        crop_template_match_fallback = kwargs.get("crop_template_match_fallback", True)
        crop_template_match_min_score = kwargs.get("crop_template_match_min_score", 0.3)
        crop_template_match_padding = kwargs.get("crop_template_match_padding", 0.05)
        debug_crop_log = kwargs.get("debug_crop_log", False)
        debug_crop_log_path = kwargs.get("debug_crop_log_path", None)
        debug_crop_save_images = kwargs.get("debug_crop_save_images", False)
        debug_crop_save_images_dir = kwargs.get("debug_crop_save_images_dir", None)
        _preprocess = kwargs.get("_preprocess")
        _try_flip = kwargs.get("try_flip", True)

        crop_info = None

        # 画角切り出しの結果をUI・JSON用に保持
        debug_crop_file = None
        if debug_crop_log and debug_crop_log_path:
            try:
                os.makedirs(os.path.dirname(debug_crop_log_path) or ".", exist_ok=True)
                debug_crop_file = open(debug_crop_log_path, "a", encoding="utf-8")
            except Exception:
                debug_crop_file = None

        def _dbg(msg):
            if debug_crop_file:
                try:
                    debug_crop_file.write(msg + "\n")
                    debug_crop_file.flush()
                except Exception:
                    pass

        _crop_save_run_id = None
        if debug_crop_save_images and debug_crop_save_images_dir and (not crop_compare_mode):
            print("  [画角デバッグ] 画像保存は「画角をマスターに合わせる」かつ「画角切り出しの比較モード」が ON のときのみ実行されます。")
        if debug_crop_save_images and debug_crop_save_images_dir:
            try:
                _crop_save_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                _crop_save_base = os.path.join(debug_crop_save_images_dir, _crop_save_run_id)
                os.makedirs(_crop_save_base, exist_ok=True)
                _crop_save_abspath = os.path.abspath(_crop_save_base)
                print(f"  [画角デバッグ] 処理中の画像を数値付きで保存: {_crop_save_abspath}")
            except Exception as _e:
                _crop_save_run_id = None
                print(f"  [画角デバッグ] 保存フォルダ作成失敗: {_e}")

        def _save_crop_candidate(img_master, img_compare, scale, num, ssim_val, mse_val, is_best, suffix=""):
            """スケール候補を「マスター|比較」の横並び＋数値オーバーレイで保存（目視確認用）"""
            if _crop_save_run_id is None or not debug_crop_save_images or not debug_crop_save_images_dir:
                return
            try:
                base = os.path.join(debug_crop_save_images_dir, _crop_save_run_id)
                h1, w1 = img_master.shape[:2]
                h2, w2 = img_compare.shape[:2]
                h = max(h1, h2)
                if h1 != h or h2 != h:
                    scale1 = h / h1 if h1 else 1
                    scale2 = h / h2 if h2 else 1
                    img_master = cv2.resize(img_master, (int(w1 * scale1), h), interpolation=cv2.INTER_LINEAR) if h1 != h else img_master
                    img_compare = cv2.resize(img_compare, (int(w2 * scale2), h), interpolation=cv2.INTER_LINEAR) if h2 != h else img_compare
                combined = np.hstack([img_master, img_compare])
                if combined.ndim == 2:
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                ssim_str = f"{ssim_val:.4e}" if ssim_val is not None else "—"
                mse_str = f"{mse_val:.1f}" if mse_val is not None else "—"
                pct = int(round(scale * 100))
                label = f"scale={scale:.2f} マスター{pct}% match={num} SSIM={ssim_str} MSE={mse_str}"
                if is_best:
                    label += " [BEST]"
                cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                safe = f"scale_{scale:.2f}_master{pct}pct_match_{num:03d}_ssim_{ssim_str}_mse_{mse_str}".replace(".", "_").replace("+", "").replace("—", "nan").replace(" ", "")[:100]
                if suffix:
                    safe = f"{safe}_{suffix}"
                fp = os.path.join(base, f"{safe}.png")
                if not safe_imwrite(fp, combined):
                    print(f"  [画角デバッグ] 画像保存失敗: {fp}")
            except Exception as _e:
                print(f"  [画角デバッグ] 画像保存失敗: {_e}")

        def _save_crop_skip(img_left, img_right, scale, reason, detail_str=""):
            """スキップしたスケールも「左|右」＋SKIP理由で保存（途中経過を全て残す）"""
            if _crop_save_run_id is None or not debug_crop_save_images or not debug_crop_save_images_dir:
                return
            try:
                base = os.path.join(debug_crop_save_images_dir, _crop_save_run_id)
                h1, w1 = img_left.shape[:2]
                h2, w2 = img_right.shape[:2]
                h = max(h1, h2)
                if h1 != h or h2 != h:
                    scale1 = h / h1 if h1 else 1
                    scale2 = h / h2 if h2 else 1
                    img_left = cv2.resize(img_left, (int(w1 * scale1), h), interpolation=cv2.INTER_LINEAR) if h1 != h else img_left
                    img_right = cv2.resize(img_right, (int(w2 * scale2), h), interpolation=cv2.INTER_LINEAR) if h2 != h else img_right
                combined = np.hstack([img_left, img_right])
                if combined.ndim == 2:
                    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                pct = int(round(scale * 100))
                label = f"scale={scale:.2f} マスター{pct}% SKIP: {reason}"
                if detail_str:
                    label += f" ({detail_str[:50]})"
                cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                safe_reason = reason.replace(" ", "_")[:30]
                fp = os.path.join(base, f"scale_{scale:.2f}_master{pct}pct_skip_{safe_reason}.png")
                if not safe_imwrite(fp, combined):
                    print(f"  [画角デバッグ] 画像保存失敗: {fp}")
            except Exception as _e:
                print(f"  [画角デバッグ] スキップ画像保存失敗: {_e}")

        try:
            _dbg("")
            _dbg("=" * 60)
            _dbg(f"=== crop debug {datetime.now().isoformat()} ===")
            _dbg(f"use_crop_to_master_fov=1 crop_compare_mode={crop_compare_mode}")
            if crop_compare_mode:
                # --- 面積比チェック: サイズ比が小さい場合はcropスキップ ---
                _area_a = imageA.shape[0] * imageA.shape[1]
                _area_b = imageB.shape[0] * imageB.shape[1]
                _area_ratio = _area_b / _area_a if _area_a > 0 else 1.0
                if _area_ratio < crop_min_area_ratio:
                    _dbg(f"[crop_skip] area_ratio={_area_ratio:.2f} < crop_min_area_ratio={crop_min_area_ratio} → cropスキップ、リサイズのみ")
                    print(f"  面積比 {_area_ratio:.2f} < {crop_min_area_ratio} → cropスキップ（リサイズのみ）")
                    if imageA.shape[:2] != imageB.shape[:2]:
                        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=cv2.INTER_LINEAR)
                    crop_info = {
                        "enabled": True,
                        "compare_mode": True,
                        "adopted": False,
                        "reason": f"area_ratio={_area_ratio:.2f} < {crop_min_area_ratio}",
                        "flipped": False,
                        "no_crop_resized": 1,
                        "resize_shape": imageA.shape[:2],
                    }
                    return imageA, imageB, crop_info

                # スケール一括: 最大(1.0)から最小まで全スケールを試し、最良の1つを採用
                imageA_no_crop = imageA.copy()
                imageB_no_crop = imageB.copy()
                w_orig, h_orig = imageA.shape[1], imageA.shape[0]
                if crop_master_padding_ratio and crop_master_padding_ratio > 0:
                    pad_x = int(w_orig * crop_master_padding_ratio)
                    pad_y = int(h_orig * crop_master_padding_ratio)
                    imageA_work = cv2.copyMakeBorder(imageA, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
                    h, w = imageA_work.shape[:2]
                    _dbg(f"[params] crop_master_padding_ratio={crop_master_padding_ratio} pad_x={pad_x} pad_y={pad_y}")
                    print(f"  マスターに余白を付けてスケール探索（余白率={crop_master_padding_ratio}）")
                else:
                    pad_x, pad_y = 0, 0
                    imageA_work = imageA
                    h, w = imageA.shape[:2]
                n_steps = max(1, int(round((1.0 - crop_scale_min) / crop_scale_step)))
                scales_all = [1.0 - i * crop_scale_step for i in range(n_steps + 1)]
                _dbg("[scale_search] mode=スケール一括（最大→最小までやり切る）")
                # 切り出し前（フル画像同士）のマッチ数を基準とする。固定値ではなく「これ未満は採用しない」
                baseline_matches = None
                baseline_fov_fail_detail = None  # 基準用画角合わせ失敗時の詳細（ログ用）
                ref_A_full, ref_B_full = imageA, imageB
                if imageA.shape[:2] != imageB.shape[:2]:
                    _res = crop_to_master_fov(
                        imageA, imageB,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_matches=4,
                        try_rotations=crop_try_rotations,
                        rotation_search_step_deg=crop_rotation_search_step_deg,
                        try_flip=_try_flip,
                        max_rotation_deg=crop_max_rotation_deg,
                        min_inlier_ratio=crop_min_inlier_ratio,
                    )
                    cropped_b, warped_a, ok_fov = _res[0], _res[1], _res[2]
                    if ok_fov and cropped_b is not None:
                        ref_A_full = warped_a if warped_a is not None else imageA
                        ref_B_full = cropped_b
                    else:
                        baseline_fov_fail_detail = _res[3] if len(_res) > 3 else None
                if ref_A_full.shape[:2] == ref_B_full.shape[:2]:
                    _, _, baseline_matches, _, _, _, _, _ = auto_align_images(
                        ref_A_full, ref_B_full,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,
                        min_translation=99999,
                    )
                # 画角合わせ失敗で基準が取れなかった場合: 比較画像をマスターにリサイズしてマッチ数を取得（フォールバック）
                baseline_from_resize = False
                if baseline_matches is None and imageA.shape[:2] != imageB.shape[:2]:
                    # 基準は「元マスターサイズ」で取得（余白付き時は w,h が余白込みなので w_orig,h_orig を使用）
                    size_baseline = (w_orig, h_orig)
                    imageB_resized = cv2.resize(imageB, size_baseline, interpolation=cv2.INTER_LINEAR)
                    _, _, baseline_matches, _, _, _, _, _ = auto_align_images(
                        imageA, imageB_resized,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,
                        min_translation=99999,
                    )
                    if baseline_matches is not None:
                        baseline_from_resize = True
                        _dbg(f"[params] baseline_from_resize={baseline_matches} (画角合わせ失敗のためリサイズで取得)")
                # 基準がリサイズ由来のときは「異サイズのまま画角合わせ」より有利なので、候補の最小をそのままにすると厳しすぎる。上限を設ける
                _baseline_resize_cap = 30
                if baseline_matches is not None:
                    min_matches_effective = max(4, baseline_matches)
                    if baseline_from_resize and min_matches_effective > _baseline_resize_cap:
                        min_matches_effective = max(4, _baseline_resize_cap)
                        _dbg(f"[params] baseline_from_resize のため min_matches_effective を {_baseline_resize_cap} にキャップ")
                else:
                    min_matches_effective = max(4, crop_two_pass_min_matches)
                _dbg(f"[params] master_size={w}x{h} scale_min={crop_scale_min} scale_step={crop_scale_step} baseline_matches={baseline_matches} min_matches_effective={min_matches_effective}")
                if baseline_fov_fail_detail is not None:
                    _baseline_detail_str = " ".join(f"{k}={v}" for k, v in baseline_fov_fail_detail.items())
                    _dbg(f"[params] baseline_fov_fail detail={_baseline_detail_str}")
                _dbg(f"[params] scales_all={[round(s, 2) for s in scales_all]}")
                print("\n--- 画角スケール探索（最大→最小まで一括）開始 ---")
                print(f"  マスターサイズ: {w}x{h}, スケール範囲: {crop_scale_min}〜1.0, 刻み: {crop_scale_step}")
                if baseline_matches is not None:
                    resize_note = " （リサイズで取得）" if baseline_from_resize else ""
                    cap_note = "（リサイズ基準のためキャップ）" if (baseline_from_resize and min_matches_effective == _baseline_resize_cap) else ""
                    print(f"  切り出し前マッチ数（参考）: {baseline_matches}（最小={min_matches_effective}）{resize_note} {cap_note}")
                else:
                    print(f"  切り出し前マッチ数（参考）: 取得失敗（最小={min_matches_effective}）")
                    if baseline_fov_fail_detail:
                        _msg = " ".join(f"{k}={v}" for k, v in baseline_fov_fail_detail.items())
                        print(f"    （基準用画角合わせ失敗: {_msg}）")
                print(f"  候補はマッチ数でスキップせず出し、採用は「切り出し前 vs 切り出し後」の全項目比較で判断")
                print(f"  スコア: MSE→SSIM→マッチ数→スケール大を優先（拡大しすぎで一致率が高く出るのを抑制）")
                print(f"  スケール列: {[round(s, 2) for s in scales_all]}")
                if _crop_save_run_id:
                    _save_crop_candidate(imageA_no_crop, imageB_no_crop, 1.0, 0, None, None, False, "no_crop")

                def _eval_at_scale(scale):
                    th, tw = int(h * scale), int(w * scale)
                    if tw < 10 or th < 10:
                        if _crop_save_run_id:
                            _save_crop_skip(imageA_work, imageB, scale, "size_too_small", f"tw={tw} th={th}")
                        return (None, "size_too_small", {"tw": tw, "th": th})
                    # マスターを比率保ったまま縮小（余白付きの場合は imageA_work を使用）
                    master_crop = cv2.resize(imageA_work, (tw, th), interpolation=cv2.INTER_LINEAR)
                    # min_matches は crop_to_master_fov 内部の「アフィン変換 vs ホモグラフィ」切替閾値。
                    # baseline_matches 由来の min_matches_effective（例: 138）を渡すと
                    # 49マッチでもアフィン（3点のみ）が選ばれ、ワープが崩壊する。
                    # ここでは固定の小さい値を渡し、候補の品質判定は後段のSSIM/MSE比較で行う。
                    _crop_min_matches = 10  # ホモグラフィに最低限必要なマッチ数
                    result = crop_to_master_fov(
                        master_crop, imageB,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_matches=_crop_min_matches,
                        try_rotations=crop_try_rotations,
                        rotation_search_step_deg=crop_rotation_search_step_deg,
                        try_flip=_try_flip,
                        max_rotation_deg=crop_max_rotation_deg,
                        min_inlier_ratio=crop_min_inlier_ratio,
                    )
                    ok = result[2]
                    cropped_b, warped_master = result[0], result[1]
                    crop_diag = result[3] if len(result) > 3 else None
                    if not ok:
                        if _crop_save_run_id:
                            _save_crop_skip(master_crop, imageB, scale, "crop_to_master_fov_fail", _fmt_detail(crop_diag) if crop_diag else "")
                        return (None, "crop_to_master_fov_fail", {"crop_to_master_fov": crop_diag})
                    # crop_to_master_fov 内部の good_matches 数（ホモグラフィ計算に使った数 = 信頼度指標）
                    crop_good_matches = crop_diag.get("good_matches", 0) if crop_diag and isinstance(crop_diag, dict) else 0
                    crop_flipped = crop_diag.get("flipped", False) if crop_diag and isinstance(crop_diag, dict) else False

                    # --- 面積比に対するマッチ数ゲート ---
                    # 面積比が極端に大きい場合、少ないマッチ数ではホモグラフィが不安定
                    _area_master = th * tw
                    _area_compare = imageB.shape[0] * imageB.shape[1]
                    _area_ratio_for_gate = max(_area_compare, _area_master) / max(min(_area_compare, _area_master), 1)
                    if _area_ratio_for_gate > 10.0 and crop_good_matches > 0:
                        _min_matches_for_ratio = max(12, int(math.sqrt(_area_ratio_for_gate) * 3))
                        if crop_good_matches < _min_matches_for_ratio:
                            if _crop_save_run_id:
                                _save_crop_skip(master_crop, imageB, scale, "insufficient_matches_for_area_ratio",
                                                f"matches={crop_good_matches} need={_min_matches_for_ratio} ratio={_area_ratio_for_gate:.1f}")
                            return (None, "insufficient_matches_for_area_ratio", {
                                "crop_good_matches": crop_good_matches,
                                "min_matches_required": _min_matches_for_ratio,
                                "area_ratio": round(_area_ratio_for_gate, 1),
                                "scale": scale,
                            })

                    ref_A = warped_master if warped_master is not None else master_crop
                    ref_B = cropped_b
                    if ref_A.shape[:2] != ref_B.shape[:2]:
                        if _crop_save_run_id:
                            _save_crop_skip(ref_A, ref_B, scale, "size_mismatch_after_crop", "")
                        return (None, "size_mismatch_after_crop", {"ref_A_shape": ref_A.shape[:2], "ref_B_shape": ref_B.shape[:2]})
                    _, success, num_align, _, _, _, _, _ = auto_align_images(
                        ref_A, ref_B,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,
                        min_translation=99999,
                    )
                    # スコアリング用マッチ数: crop内部マッチ数と後段アライメントマッチ数の大きい方を採用
                    # （後段は画像変換後の再検出なので不安定、crop内部の方が信頼度が高い場合が多い）
                    num = max(crop_good_matches if crop_good_matches else 0, num_align if num_align else 0)
                    # スキップせず候補として保持。採用可否は最終段階で「切り出し前 vs 切り出し後」の比較で判断する
                    procA = _preprocess(ref_A)
                    procB = _preprocess(ref_B)
                    ssim_val, _ = ssim(procA, procB, full=True)
                    mse_val = calculate_mse(procA, procB)
                    # preprocessを通さない生画像でのSSIM（スコアリング用: luminance前処理でSSIMが壊れる場合の対策）
                    grayA_raw = cv2.cvtColor(ref_A, cv2.COLOR_BGR2GRAY) if len(ref_A.shape) == 3 else ref_A
                    grayB_raw = cv2.cvtColor(ref_B, cv2.COLOR_BGR2GRAY) if len(ref_B.shape) == 3 else ref_B
                    ssim_raw, _ = ssim(grayA_raw, grayB_raw, full=True)
                    # ワープ結果が明らかに壊れている場合をスキップ（真っ黒画像、完全なズレ等）
                    _warp_quality_min = 0.35  # raw SSIM がこれ未満はホモグラフィ崩壊と判断
                    if ssim_raw < _warp_quality_min:
                        if _crop_save_run_id:
                            _save_crop_skip(ref_A, ref_B, scale, "warp_quality_too_low", f"ssim_raw={ssim_raw:.4f}")
                        return (None, "warp_quality_too_low", {"ssim_raw": ssim_raw, "scale": scale})
                    return ((ref_A, ref_B, num, ssim_val, mse_val, scale, ssim_raw, crop_flipped), None, None)

                def _score_tuple(r):
                    ssim_raw = r[6] if r[6] is not None else 0.0
                    num = r[2] if r[2] is not None else 0
                    mse = r[4] if r[4] is not None else 1e9
                    # raw SSIM高い → マッチ数多い → MSE低い → スケール大 を優先
                    # SSIM が最も信頼できる品質指標（ワープ失敗=SSIM≈0、成功=SSIM>0.8）
                    # r[7] = flipped (bool) — スコアには影響しない
                    return (-ssim_raw, -num, mse, -r[5])

                _skip_reason_ja = {
                    "size_too_small": "切り出しサイズが小さすぎ",
                    "center_crop_fail": "マスター中心クロップ失敗",
                    "crop_to_master_fov_fail": "画角合わせ失敗（特徴点マッチ不足）",
                    "size_mismatch_after_crop": "切り出し後サイズ不一致",
                    "warp_quality_too_low": "ワープ品質不良（SSIM_raw<0.1: ホモグラフィ崩壊）",
                    "insufficient_matches_for_area_ratio": "面積比に対してマッチ数不足",
                }

                def _fmt_detail(d):
                    if not d or not isinstance(d, dict):
                        return ""
                    return " ".join(f"{k}={v}" for k, v in d.items())

                def _log_scale_result(scale, r, is_best, skip_reason=None, skip_detail=None):
                    if r is None:
                        reason_str = skip_reason if skip_reason else "不明"
                        if skip_reason and "min_matches" in str(skip_reason):
                            reason_ja = f"切り出し前マッチ数未満（{skip_reason}）"
                        else:
                            reason_ja = _skip_reason_ja.get(skip_reason) if skip_reason in _skip_reason_ja else reason_str
                        detail_str = ""
                        if skip_detail:
                            if "crop_to_master_fov" in skip_detail and skip_detail["crop_to_master_fov"]:
                                d = skip_detail["crop_to_master_fov"]
                                detail_str = _fmt_detail(d)
                            else:
                                detail_str = _fmt_detail(skip_detail)
                        if detail_str:
                            _dbg(f"[scale] scale={scale:.2f} skip=1 reason={reason_str} detail={detail_str}")
                            print(f"    スケール {scale:.2f}: スキップ（{reason_ja}） 詳細: {detail_str}")
                        else:
                            _dbg(f"[scale] scale={scale:.2f} skip=1 reason={reason_str}")
                            print(f"    スケール {scale:.2f}: スキップ（{reason_ja}）")
                        return
                    _, _, num, ssim_val, mse_val, _, ssim_raw = r[:7]
                    _r_flipped = r[7] if len(r) > 7 else False
                    ssim_str = f"{ssim_val:.4f}" if ssim_val is not None else "—"
                    ssim_raw_str = f"{ssim_raw:.4f}" if ssim_raw is not None else "—"
                    mse_str = f"{mse_val:.2f}" if mse_val is not None else "—"
                    flip_str = " [左右反転]" if _r_flipped else ""
                    mark = " ← ベスト" if is_best else ""
                    _dbg(f"[scale] scale={scale:.2f} num={num} ssim={ssim_val} ssim_raw={ssim_raw} mse={mse_val} flipped={int(_r_flipped)} best={1 if is_best else 0}")
                    print(f"    スケール {scale:.2f}: マッチ={num}, SSIM_raw={ssim_raw_str}, MSE={mse_str}{flip_str}{mark}")

                # 最大→最小まで一括で試し、最良を1つ選ぶ
                print("  [スケール一括] 1.0 から最小まで試行")
                best, best_score = None, None
                all_candidates = []  # 全候補を保持（フォールバック用）
                for scale in scales_all:
                    data, skip_reason, skip_detail = _eval_at_scale(scale)
                    if data is None:
                        _log_scale_result(scale, None, False, skip_reason=skip_reason, skip_detail=skip_detail)
                        continue
                    r = data
                    all_candidates.append(r)
                    sc = _score_tuple(r)
                    is_best = best is None or sc < best_score
                    if is_best:
                        best, best_score = r, sc
                    _log_scale_result(scale, r, is_best)
                    ref_A, ref_B, num, ssim_val, mse_val, scale_val, ssim_raw_val = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
                    _save_crop_candidate(ref_A, ref_B, scale_val, num, ssim_val, mse_val, is_best)
                if best is not None:
                    _, _, num_best, ssim_best, mse_best, scale_best, ssim_raw_best = best[:7]
                    _best_flipped_flag = best[7] if len(best) > 7 else False
                    _flip_mark = " [左右反転]" if _best_flipped_flag else ""
                    _dbg(f"[best] scale={scale_best:.2f} num={num_best} ssim={ssim_best} ssim_raw={ssim_raw_best} mse={mse_best} flipped={int(_best_flipped_flag)}")
                    print(f"  ベスト: スケール={scale_best:.2f}, マッチ={num_best}, SSIM_raw={ssim_raw_best:.4f}, MSE={mse_best}{_flip_mark}")
                    # ベストが最小スケール付近なら「縮小が足りていない」可能性を案内
                    if scale_best <= crop_scale_min + 0.02:
                        print(f"  ※ ベストが最小スケール付近です。まだ合わない場合は config の crop_scale_min を下げてさらに縮小を試せます（例: 0.2）")
                else:
                    _dbg("[best] none")
                    print("  有効な候補なし")

                if best is None:
                    _dbg("[decision] no_candidates adopted=0 reason=有効な候補なし")
                    _area_master_fb = imageA_work.shape[0] * imageA_work.shape[1]
                    _area_compare_fb = imageB.shape[0] * imageB.shape[1]
                    _area_ratio_fb = _area_compare_fb / max(_area_master_fb, 1)
                    _fallback_done = False

                    # --- フォールバック1: テンプレートマッチング ---
                    if crop_template_match_fallback and _area_ratio_fb >= 1.5:
                        _dbg(f"[template_match_fallback] trying area_ratio={_area_ratio_fb:.1f}")
                        print(f"  テンプレートマッチングフォールバック試行（面積比={_area_ratio_fb:.1f}x）")
                        _tm_result, _tm_diag = template_match_crop(
                            imageA_work, imageB,
                            try_rotations=crop_try_rotations,
                            min_score=crop_template_match_min_score,
                            padding_ratio=crop_template_match_padding,
                        )
                        if _tm_result is not None:
                            _tm_score = _tm_diag.get("score", 0)
                            _tm_scale = _tm_diag.get("adopted_scale", 0)
                            _tm_rot = _tm_diag.get("adopted_rotation", 0)
                            print(f"  ✓ テンプレートマッチング成功: score={_tm_score:.4f}, scale={_tm_scale:.2f}, rotation={_tm_rot:.0f}°")
                            _dbg(f"[template_match_fallback] adopted score={_tm_score} scale={_tm_scale} rot={_tm_rot} diag={_tm_diag}")
                            print("--- 画角スケール探索 終了 ---\n")
                            crop_info = {
                                "enabled": True,
                                "compare_mode": True,
                                "two_pass": True,
                                "adopted": True,
                                "reason": "template_match",
                                "template_match_diag": _tm_diag,
                                "flipped": False,
                                "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                                "preview_pass1": None,
                                "preview_pass2": None,
                            }
                            imageA = imageA_work
                            imageB = _tm_result
                            _fallback_done = True
                        else:
                            _dbg(f"[template_match_fallback] failed diag={_tm_diag}")
                            print(f"  ✗ テンプレートマッチング失敗: {_tm_diag.get('reject', 'unknown')}")

                    # --- フォールバック2: 中央クロップ（テンプレートマッチ失敗時） ---
                    if not _fallback_done and 1.5 <= _area_ratio_fb <= 10.0:
                        _center_h, _center_w = imageA_work.shape[:2]
                        _bh, _bw = imageB.shape[:2]
                        if _bh >= _center_h and _bw >= _center_w:
                            _y0 = (_bh - _center_h) // 2
                            _x0 = (_bw - _center_w) // 2
                            _center_cropped = imageB[_y0:_y0 + _center_h, _x0:_x0 + _center_w]
                        else:
                            _center_cropped = None
                        if _center_cropped is not None and _center_cropped.shape[:2] == (_center_h, _center_w):
                            print(f"⚠ 画角スケール探索: テンプレートマッチ失敗 → 中央クロップフォールバック（面積比={_area_ratio_fb:.1f}x）")
                            _dbg(f"[center_crop_fallback] area_ratio={_area_ratio_fb:.1f} master=({_center_h},{_center_w})")
                            print("--- 画角スケール探索 終了 ---\n")
                            crop_info = {
                                "enabled": True,
                                "compare_mode": True,
                                "two_pass": True,
                                "adopted": True,
                                "reason": "center_crop_fallback",
                                "flipped": False,
                                "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                                "preview_pass1": None,
                                "preview_pass2": None,
                            }
                            imageA = imageA_work
                            imageB = _center_cropped
                            _fallback_done = True

                    # --- フォールバック3: 何もできない → 元の比較画像で続行 ---
                    if not _fallback_done:
                        print("⚠ 画角スケール探索: 有効な候補がありません。元の比較画像で続行")
                        print("--- 画角スケール探索 終了 ---\n")
                        crop_info = {
                            "enabled": True,
                            "compare_mode": True,
                            "two_pass": True,
                            "adopted": False,
                            "reason": "有効な候補なし",
                            "flipped": False,
                            "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                            "preview_pass1": None,
                            "preview_pass2": None,
                        }
                else:
                    ref_A, ref_B, num, ssim_val, mse_val, scale, ssim_raw_val, best_flipped = best

                    # 切り出しなし（元画像）の数値を取得し、全項目で良くなるか判定
                    h_nc, w_nc = imageA_no_crop.shape[:2]
                    if imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2]:
                        imageB_no_crop_resized = cv2.resize(imageB_no_crop, (w_nc, h_nc), interpolation=cv2.INTER_LINEAR)
                    else:
                        imageB_no_crop_resized = imageB_no_crop
                    _, success_orig, num_orig, _, _, _, _, _ = auto_align_images(
                        imageA_no_crop, imageB_no_crop_resized,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,
                        min_translation=99999,
                    )
                    procA_nc = _preprocess(imageA_no_crop)
                    procB_nc = _preprocess(imageB_no_crop_resized)
                    ssim_orig, _ = ssim(procA_nc, procB_nc, full=True)
                    mse_orig = calculate_mse(procA_nc, procB_nc)
                    # raw SSIM (no-crop baseline)
                    grayA_nc = cv2.cvtColor(imageA_no_crop, cv2.COLOR_BGR2GRAY) if len(imageA_no_crop.shape) == 3 else imageA_no_crop
                    grayB_nc = cv2.cvtColor(imageB_no_crop_resized, cv2.COLOR_BGR2GRAY) if len(imageB_no_crop_resized.shape) == 3 else imageB_no_crop_resized
                    ssim_raw_orig, _ = ssim(grayA_nc, grayB_nc, full=True)

                    if num_orig is None or num_orig <= 0:
                        match_ok = True
                        match_ratio = None
                    else:
                        match_ratio = num / float(num_orig)
                        match_ok = match_ratio >= float(crop_match_ratio_min)
                    # 切り出しなしではマッチがほぼ取れていない（num_orig<=1）が切り出しありでは十分マッチしている場合: 画角合わせが明らかに効いているので SSIM/MSE 悪化も許容して採用
                    _align_clearly_improved = num_orig <= 1 and num >= crop_adopt_min_matches and (ssim_raw_val is not None and ssim_raw_val >= 0.5)
                    # SSIM判定: preprocess版とraw版の両方を考慮（luminance前処理でSSIMが壊れる場合の対策）
                    ssim_ok_proc = (ssim_orig is None or ssim_val is None) or (ssim_val >= ssim_orig + crop_ssim_degradation_threshold) or _align_clearly_improved
                    ssim_ok_raw = (ssim_raw_orig is None or ssim_raw_val is None) or (ssim_raw_val >= ssim_raw_orig + crop_ssim_degradation_threshold) or _align_clearly_improved
                    ssim_ok = ssim_ok_proc or ssim_ok_raw  # どちらかがOKなら許容
                    mse_ok = (mse_orig is None or mse_val is None) or (mse_val <= mse_orig * (1.0 + crop_mse_degradation_threshold)) or _align_clearly_improved
                    # 切り出し採用時のみ: 絶対閾値 SSIM > crop_ssim_min（0/None なら無効）。マッチ数が十分ならスキップ（ナット違い等でSSIM低くても採用）。
                    # 画角合わせが明らかに効いている場合もスキップ。raw SSIMも考慮。
                    ssim_min_ok = (crop_ssim_min is None or crop_ssim_min <= 0) or (ssim_val is not None and ssim_val > crop_ssim_min) or (ssim_raw_val is not None and ssim_raw_val > crop_ssim_min) or (num >= crop_ssim_min_skip_matches) or _align_clearly_improved
                    # 切り出し採用時のみ: マッチ数が極端に少ない候補は不採用（失敗例: スケール=1.0, マッチ=1, SSIM≈0, MSE大）
                    adopt_match_ok = num >= crop_adopt_min_matches
                    all_improved = match_ok and ssim_ok and mse_ok and ssim_min_ok and adopt_match_ok

                    _dbg(f"[best] scale={scale:.2f} num={num} ssim={ssim_val} ssim_raw={ssim_raw_val} mse={mse_val} flipped={int(best_flipped)}")
                    _dbg(f"[no_crop] num_orig={num_orig} ssim_orig={ssim_orig} ssim_raw_orig={ssim_raw_orig} mse_orig={mse_orig}")
                    _dbg(f"[compare] match_ok={int(match_ok)} ssim_ok={int(ssim_ok)} mse_ok={int(mse_ok)} ssim_min_ok={int(ssim_min_ok)} adopt_match_ok={int(adopt_match_ok)} all_improved={int(all_improved)}")
                    _flip_note = " ※比較画像を左右反転して使用" if best_flipped else ""
                    print("  [全項目比較] 切り出しあり vs 切り出しなし（全て良くなる場合のみ採用）")
                    print(f"    切り出しなし: マッチ={num_orig}, SSIM_raw={ssim_raw_orig:.4f}, MSE={mse_orig}")
                    print(f"    切り出しあり: マッチ={num}, SSIM_raw={ssim_raw_val:.4f}, MSE={mse_val}{_flip_note}")
                    if all_improved:
                        _dbg(f"[decision] adopted=1 scale={scale:.2f}")
                        # 余白付きマスターで探索した場合: 2段階処理（余白ありで粗い切り出し→余白なしで綺麗に切り出し）
                        if pad_x > 0 or pad_y > 0:
                            tw_content = int(w_orig * scale)
                            th_content = int(h_orig * scale)
                            if tw_content >= 10 and th_content >= 10:
                                master_content = cv2.resize(imageA, (tw_content, th_content), interpolation=cv2.INTER_LINEAR)
                                res2 = crop_to_master_fov(
                                    master_content, imageB,
                                    method=align_method,
                                    max_features=align_max_features,
                                    match_ratio=align_match_ratio,
                                    min_matches=_crop_min_matches,
                                    try_rotations=crop_try_rotations,
                                    rotation_search_step_deg=crop_rotation_search_step_deg,
                                    try_flip=_try_flip,
                                    max_rotation_deg=crop_max_rotation_deg,
                                    min_inlier_ratio=crop_min_inlier_ratio,
                                )
                                if res2[2]:
                                    ref_A = res2[1] if res2[1] is not None else master_content
                                    ref_B = res2[0]
                                    _dbg(f"[padding_two_stage] stage2 ok: 余白なしマスターで crop_to_master_fov 再実行 → コンテンツのみで採用")
                                    print(f"  2段階: 余白なしマスターで再切り出し → 採用（コンテンツのみ）")
                                else:
                                    # 段階2失敗時: 従来どおりスライスで余白除去
                                    th_best = int(h * scale)
                                    tw_best = int(w * scale)
                                    content_left = int(pad_x * tw_best / w)
                                    content_top = int(pad_y * th_best / h)
                                    content_w = int((w - 2 * pad_x) * tw_best / w)
                                    content_h = int((h - 2 * pad_y) * th_best / h)
                                    ref_A = ref_A[content_top:content_top + content_h, content_left:content_left + content_w]
                                    ref_B = ref_B[content_top:content_top + content_h, content_left:content_left + content_w]
                                    _dbg(f"[padding_two_stage] stage2 fail → slice content_region left={content_left} top={content_top} w={content_w} h={content_h}")
                                    print(f"  2段階: 余白なしでの再切り出しは失敗 → コンテンツ領域 {content_w}x{content_h} でスライスして採用")
                            else:
                                th_best = int(h * scale)
                                tw_best = int(w * scale)
                                content_left = int(pad_x * tw_best / w)
                                content_top = int(pad_y * th_best / h)
                                content_w = int((w - 2 * pad_x) * tw_best / w)
                                content_h = int((h - 2 * pad_y) * th_best / h)
                                ref_A = ref_A[content_top:content_top + content_h, content_left:content_left + content_w]
                                ref_B = ref_B[content_top:content_top + content_h, content_left:content_left + content_w]
                                _dbg(f"[padding_remove] content_region left={content_left} top={content_top} w={content_w} h={content_h}")
                                print(f"  余白除去: コンテンツ領域 {content_w}x{content_h} で切り出し")
                        imageA, imageB = ref_A, ref_B
                        _flip_adopted_note = "、左右反転あり" if best_flipped else ""
                        print(f"✓ 画角スケール探索: 全項目OK → 採用（スケール={scale:.2f}{_flip_adopted_note}）")
                        print("--- 画角スケール探索 終了 ---\n")
                        # 余白除去した場合は最終画像でプレビュー（そうでなければベスト候補のまま）
                        preview_pass1 = (imageA.copy(), imageB.copy(), best[5], best[2], best[3], best[4])
                        crop_info = {
                            "enabled": True,
                            "compare_mode": True,
                            "two_pass": True,
                            "adopted": True,
                            "adopted_pass": 1,
                            "adopted_scale": float(scale),
                            "matches": num,
                            "ssim": float(ssim_val) if ssim_val is not None else None,
                            "ssim_raw": float(ssim_raw_val) if ssim_raw_val is not None else None,
                            "mse": float(mse_val) if mse_val is not None else None,
                            "matches_orig": num_orig,
                            "ssim_orig": float(ssim_orig) if ssim_orig is not None else None,
                            "ssim_raw_orig": float(ssim_raw_orig) if ssim_raw_orig is not None else None,
                            "mse_orig": float(mse_orig) if mse_orig is not None else None,
                            "match_ratio": float(match_ratio) if match_ratio is not None else None,
                            "match_ratio_min": float(crop_match_ratio_min),
                            "match_ok": bool(match_ok),
                            "ssim_ok": bool(ssim_ok),
                            "mse_ok": bool(mse_ok),
                            "ssim_min_ok": bool(ssim_min_ok),
                            "adopt_match_ok": bool(adopt_match_ok),
                            "pass1_scale": float(best[5]),
                            "pass2_scale": None,
                            "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                            "preview_pass1": preview_pass1,
                            "preview_pass2": None,
                            "flipped": bool(best_flipped),
                            "no_crop_resized": bool(imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2]),
                            "no_crop_resize_shape": imageB_no_crop_resized.shape[:2] if imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2] else None,
                        }
                    else:
                        # --- フォールバック: ベスト候補が不採用なら、マッチ数が最大の他の候補を試す ---
                        _fallback_adopted = False
                        if not match_ok and len(all_candidates) > 1:
                            # マッチ数降順でソートし、ベスト以外の候補を試す
                            _fb_candidates = sorted(all_candidates, key=lambda c: -(c[2] if c[2] else 0))
                            for _fb in _fb_candidates:
                                _fb_num = _fb[2] if _fb[2] else 0
                                _fb_scale = _fb[5]
                                if _fb_scale == scale and _fb_num == num:
                                    continue  # ベストと同じ候補はスキップ
                                if _fb_num <= num_orig:
                                    continue  # 切り出しなしよりマッチが多くないとダメ
                                # この候補で全項目チェック
                                _fb_ssim = _fb[3]
                                _fb_ssim_raw = _fb[6]
                                _fb_mse = _fb[4]
                                _fb_match_ratio = _fb_num / float(num_orig) if num_orig > 0 else None
                                _fb_match_ok = num_orig == 0 or (_fb_match_ratio is not None and _fb_match_ratio >= float(crop_match_ratio_min)) or (_fb_num > num_orig)
                                _fb_align_improved = num_orig <= 1 and _fb_num >= crop_adopt_min_matches and (_fb_ssim_raw is not None and _fb_ssim_raw >= 0.5)
                                _fb_ssim_ok_proc = (ssim_orig is None or _fb_ssim is None) or (_fb_ssim >= ssim_orig + crop_ssim_degradation_threshold) or _fb_align_improved
                                _fb_ssim_ok_raw = (ssim_raw_orig is None or _fb_ssim_raw is None) or (_fb_ssim_raw >= ssim_raw_orig + crop_ssim_degradation_threshold) or _fb_align_improved
                                _fb_ssim_ok = _fb_ssim_ok_proc or _fb_ssim_ok_raw
                                _fb_mse_ok = (mse_orig is None or _fb_mse is None) or (_fb_mse <= mse_orig * (1.0 + crop_mse_degradation_threshold)) or _fb_align_improved
                                _fb_ssim_min_ok = (crop_ssim_min is None or crop_ssim_min <= 0) or (_fb_ssim is not None and _fb_ssim > crop_ssim_min) or (_fb_ssim_raw is not None and _fb_ssim_raw > crop_ssim_min) or (_fb_num >= crop_ssim_min_skip_matches) or _fb_align_improved
                                _fb_adopt_ok = _fb_num >= crop_adopt_min_matches
                                _fb_all_ok = _fb_match_ok and _fb_ssim_ok and _fb_mse_ok and _fb_ssim_min_ok and _fb_adopt_ok
                                if _fb_all_ok:
                                    _dbg(f"[fallback] ベスト不採用→フォールバック候補 scale={_fb_scale:.2f} num={_fb_num} ssim_raw={_fb_ssim_raw} を採用")
                                    print(f"  [フォールバック] ベスト(scale={scale:.2f},マッチ={num})が不採用 → 候補(scale={_fb_scale:.2f},マッチ={_fb_num})で再判定OK → 採用")
                                    # フォールバック候補を採用
                                    best = _fb
                                    ref_A, ref_B, num, ssim_val, mse_val, scale, ssim_raw_val, best_flipped = best
                                    _fallback_adopted = True
                                    break
                        if _fallback_adopted:
                            # 採用パスへ合流（上のif all_improvedブロックと同じ処理）
                            if crop_master_padding_ratio and crop_master_padding_ratio > 0:
                                tw_best, th_best = ref_A.shape[1], ref_A.shape[0]
                                content_left = int(pad_x * tw_best / w)
                                content_top = int(pad_y * th_best / h)
                                content_w = int((w - 2 * pad_x) * tw_best / w)
                                content_h = int((h - 2 * pad_y) * th_best / h)
                                ref_A = ref_A[content_top:content_top + content_h, content_left:content_left + content_w]
                                ref_B = ref_B[content_top:content_top + content_h, content_left:content_left + content_w]
                            imageA, imageB = ref_A, ref_B
                            print(f"✓ 画角スケール探索: フォールバック採用（スケール={scale:.2f}）")
                            print("--- 画角スケール探索 終了 ---\n")
                            preview_pass1 = (imageA.copy(), imageB.copy(), best[5], best[2], best[3], best[4])
                            crop_info = {
                                "enabled": True, "compare_mode": True, "two_pass": True, "adopted": True,
                                "adopted_pass": 1, "adopted_scale": float(scale),
                                "matches": num, "ssim": float(ssim_val) if ssim_val is not None else None,
                                "ssim_raw": float(ssim_raw_val) if ssim_raw_val is not None else None,
                                "mse": float(mse_val) if mse_val is not None else None,
                                "matches_orig": num_orig, "ssim_orig": float(ssim_orig) if ssim_orig is not None else None,
                                "ssim_raw_orig": float(ssim_raw_orig) if ssim_raw_orig is not None else None,
                                "mse_orig": float(mse_orig) if mse_orig is not None else None,
                                "fallback": True, "flipped": bool(best_flipped),
                                "no_crop_resized": bool(imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2]),
                                "no_crop_resize_shape": imageB_no_crop_resized.shape[:2] if imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2] else None,
                                "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                                "preview_pass1": preview_pass1, "preview_pass2": None,
                            }
                        else:
                            reason_parts = []
                            if not match_ok:
                                reason_parts.append(f"マッチ数悪化 {num_orig}→{num}")
                            if not ssim_ok and ssim_orig is not None and ssim_val is not None:
                                reason_parts.append(f"SSIM悪化 {ssim_orig:.4f}→{ssim_val:.4f}")
                            if not mse_ok and mse_orig is not None and mse_val is not None:
                                reason_parts.append(f"MSE悪化 {mse_orig:.2f}→{mse_val:.2f}")
                            if not ssim_min_ok and ssim_val is not None and crop_ssim_min is not None and crop_ssim_min > 0:
                                reason_parts.append(f"SSIMが最小閾値未満 {ssim_val:.4f}≤{crop_ssim_min}")
                            if not adopt_match_ok:
                                reason_parts.append(f"マッチ数が採用最小未満 {num}<{crop_adopt_min_matches}")
                            reason_str = "、".join(reason_parts)
                            _dbg(f"[decision] adopted=0 reason={reason_str}")
                            print(f"✓ 画角スケール探索: 全項目で良くならず（{reason_str}）→ 切り出し不採用、元の比較画像で続行")
                            if not ssim_min_ok and num > num_orig and "SSIMが最小閾値未満" in reason_str:
                                print("  ※ マッチ数は改善しています。右側の角度ずれが原因の場合は crop_try_rotations: true と crop_rotation_search_step_deg: 10 で回転を探索すると改善することがあります。")
                            if not adopt_match_ok and num < crop_adopt_min_matches:
                                print("  ※ スケール刻みを細かく（crop_scale_step=0.02 や 0.01）すると、中間スケールでちょうどいい切り取りが見つかる場合があります。")
                            if scale <= crop_scale_min + 0.02:
                                print("  ※ ベストが最小スケール付近です。縮小が足りていない可能性があるため、config の crop_scale_min を下げて（例: 0.2）再実行してみてください。")
                            print("--- 画角スケール探索 終了 ---\n")
                            imageA = imageA_no_crop
                            imageB = imageB_no_crop_resized if imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2] else imageB_no_crop
                            preview_pass1 = (best[0].copy(), best[1].copy(), best[5], best[2], best[3], best[4])
                            crop_info = {
                                "enabled": True,
                                "compare_mode": True,
                                "two_pass": True,
                                "adopted": False,
                                "reason": reason_str,
                                "adopted_pass": 1,
                                "adopted_scale": float(scale),
                                "matches": num,
                                "ssim": float(ssim_val) if ssim_val is not None else None,
                                "ssim_raw": float(ssim_raw_val) if ssim_raw_val is not None else None,
                                "mse": float(mse_val) if mse_val is not None else None,
                                "matches_orig": num_orig,
                                "ssim_orig": float(ssim_orig) if ssim_orig is not None else None,
                                "ssim_raw_orig": float(ssim_raw_orig) if ssim_raw_orig is not None else None,
                                "mse_orig": float(mse_orig) if mse_orig is not None else None,
                                "match_ratio": float(match_ratio) if match_ratio is not None else None,
                                "match_ratio_min": float(crop_match_ratio_min),
                                "match_ok": bool(match_ok),
                                "ssim_ok": bool(ssim_ok),
                                "mse_ok": bool(mse_ok),
                                "ssim_min_ok": bool(ssim_min_ok),
                                "adopt_match_ok": bool(adopt_match_ok),
                                "pass1_scale": float(best[5]),
                                "pass2_scale": None,
                                "flipped": bool(best_flipped),
                                "preview_no_crop": (imageA_no_crop, imageB_no_crop),
                                "preview_pass1": preview_pass1,
                                "preview_pass2": None,
                                "no_crop_resized": bool(imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2]),
                                "no_crop_resize_shape": imageB_no_crop_resized.shape[:2] if imageA_no_crop.shape[:2] != imageB_no_crop.shape[:2] else None,
                            }
            else:
                # crop_compare_mode が False の場合の処理
                # （簡略化のため、ここでは main.py の該当部分を呼び出す）
                _res = crop_to_master_fov(
                    imageA,
                    imageB,
                    method=align_method,
                    max_features=align_max_features,
                    match_ratio=align_match_ratio,
                    min_matches=10,
                    try_rotations=crop_try_rotations,
                    rotation_search_step_deg=crop_rotation_search_step_deg,
                    try_flip=_try_flip,
                    max_rotation_deg=crop_max_rotation_deg,
                    min_inlier_ratio=crop_min_inlier_ratio,
                )
                cropped_b, warped_master, ok = _res[0], _res[1], _res[2]
                _single_crop_diag = _res[3] if len(_res) > 3 else {}
                _single_crop_flipped = _single_crop_diag.get("flipped", False) if isinstance(_single_crop_diag, dict) else False
                if ok and warped_master is not None:
                    _dbg(f"[single_crop] mode=マスターを比較サイズにワープ adopted=1 flipped={int(_single_crop_flipped)}")
                    # 比較画像がマスターより小さい場合：マスターを比較サイズにワープ済み（拡大しない）
                    imageA = warped_master
                    imageB = cropped_b
                    _flip_msg = "（※比較画像は左右反転して使用）" if _single_crop_flipped else ""
                    print(f"✓ 画角をマスターに合わせました（マスターを比較画像サイズにワープ・比較画像は拡大していません）{_flip_msg}")
                    crop_info = {
                        "enabled": True,
                        "compare_mode": False,
                        "adopted": True,
                        "warped_master_to_compare_size": True,
                        "flipped": bool(_single_crop_flipped),
                    }
                elif ok and crop_compare_mode:
                    # （この部分は非常に長いため省略し、main.pyの処理を利用）
                    pass
                elif ok:
                    _dbg(f"[single_crop] mode=比較モードなし adopted=1 flipped={int(_single_crop_flipped)}")
                    imageB = cropped_b
                    _flip_msg2 = "（※比較画像は左右反転して使用）" if _single_crop_flipped else ""
                    print(f"✓ 画角をマスターに合わせて切り出しました{_flip_msg2}")
                    crop_info = {"enabled": True, "compare_mode": False, "adopted": True, "flipped": bool(_single_crop_flipped)}
                else:
                    _dbg("[single_crop] mode=ワープ失敗")
                    # ワープ失敗時も検出マッチ数を取得してUIに表示する
                    _, _success, num_matches, _, _, _, _, _ = auto_align_images(
                        imageA, imageB,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,
                        min_translation=99999,
                    )
                    min_required = 10
                    reason = f"特徴点マッチが不足（検出: {num_matches}、必要: {min_required}以上）"
                    _dbg(f"[single_crop] mode=ワープ失敗 adopted=0 num_matches={num_matches} min_required={min_required}")
                    print(f"⚠ 画角切り出しはスキップ（{reason}）、元の比較画像で続行")
                    crop_info = {
                        "enabled": True,
                        "compare_mode": False,
                        "adopted": False,
                        "reason": reason,
                        "matches_orig": num_matches,
                        "min_matches_required": min_required,
                    }
        finally:
            if debug_crop_file:
                try:
                    debug_crop_file.close()
                except Exception:
                    pass

        return imageA, imageB, crop_info

    def _align_images(self, imageA, imageB, **kwargs):
        """位置合わせの処理"""
        align_mode = kwargs.get("align_mode", "feature_points")
        align_method = kwargs.get("align_method", "orb")
        align_max_features = kwargs.get("align_max_features", 500)
        align_edge_canny_low = kwargs.get("align_edge_canny_low", 50)
        align_edge_canny_high = kwargs.get("align_edge_canny_high", 150)
        align_match_ratio = kwargs.get("align_match_ratio", 0.75)
        align_min_rotation = kwargs.get("align_min_rotation", 3.0)
        align_min_translation = kwargs.get("align_min_translation", 30)
        align_compare_mode = kwargs.get("align_compare_mode", True)
        align_rematch_check = kwargs.get("align_rematch_check", True)
        align_fallback_mode = kwargs.get("align_fallback_mode", None)
        align_fallback_on_reject = kwargs.get("align_fallback_on_reject", True)
        align_match_improvement_threshold = kwargs.get("align_match_improvement_threshold", 0.20)
        align_inlier_improvement_threshold = kwargs.get("align_inlier_improvement_threshold", 0.10)
        align_ssim_improvement_threshold = kwargs.get("align_ssim_improvement_threshold", 0.03)
        align_ssim_degradation_threshold = kwargs.get("align_ssim_degradation_threshold", -0.02)
        align_mse_improvement_threshold = kwargs.get("align_mse_improvement_threshold", 0.10)
        align_mse_degradation_threshold = kwargs.get("align_mse_degradation_threshold", -0.10)
        _preprocess = kwargs.get("_preprocess")

        print("\n--- 自動位置合わせ実行中 ---")
        def _decide_alignment_for_edge_bbox(original_b, candidate_b, rotation_deg, translation_px):
            procA = _preprocess(imageA) if _preprocess is not None else cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            procB_orig = _preprocess(original_b) if _preprocess is not None else cv2.cvtColor(original_b, cv2.COLOR_BGR2GRAY)
            procB_aligned = _preprocess(candidate_b) if _preprocess is not None else cv2.cvtColor(candidate_b, cv2.COLOR_BGR2GRAY)
            mse_original = calculate_mse(procA, procB_orig)
            ssim_original, _ = ssim(procA, procB_orig, full=True)
            mse_aligned = calculate_mse(procA, procB_aligned)
            ssim_aligned, _ = ssim(procA, procB_aligned, full=True)
            ssim_improvement = (ssim_aligned - ssim_original) / (ssim_original + 1e-10)
            mse_improvement = (mse_original - mse_aligned) / (mse_original + 1e-10)
            decision = "補正なし"
            decision_step = "ステップ3: 小さなずれ"
            use_aligned = False
            mse_degradation_ok = True
            if align_mse_degradation_threshold is not None:
                mse_degradation_ok = mse_improvement >= float(align_mse_degradation_threshold)
            if not mse_degradation_ok and ssim_improvement <= align_ssim_improvement_threshold:
                # MSE悪化 かつ SSIM改善なし → 棄却
                decision = "補正なし"
                decision_step = "ステップ1: MSE悪化"
                use_aligned = False
            elif ssim_improvement > align_ssim_improvement_threshold:
                decision = "補正あり"
                decision_step = "ステップ1: SSIM改善"
                use_aligned = True
            elif ssim_improvement < align_ssim_degradation_threshold:
                decision = "補正なし"
                decision_step = "ステップ1: SSIM悪化"
            elif mse_improvement > align_mse_improvement_threshold:
                decision = "補正あり"
                decision_step = "ステップ2: MSE改善"
                use_aligned = True
            elif rotation_deg > align_min_rotation or translation_px > align_min_translation:
                decision = "補正あり"
                decision_step = "ステップ3: 大きなずれ"
                use_aligned = True
            return {
                "use_aligned": use_aligned,
                "decision": decision,
                "decision_step": decision_step,
                "ssim_original": float(ssim_original),
                "ssim_aligned": float(ssim_aligned),
                "mse_original": float(mse_original),
                "mse_aligned": float(mse_aligned),
                "ssim_improvement": float(ssim_improvement),
                "mse_improvement": float(mse_improvement),
            }

        if align_mode == "edge":
            aligned_imageB, success, num_matches_before, applied, rotation_deg, translation_px, inlier_ratio_before, failure_reason = align_edge_based(
                imageA, imageB, canny_low=align_edge_canny_low, canny_high=align_edge_canny_high
            )
            align_info = {
                "enabled": True,
                "success": success,
                "applied": applied,
                "matches_before": num_matches_before,
                "inlier_ratio_before": inlier_ratio_before,
                "rotation_deg": rotation_deg,
                "translation_px": translation_px,
                "decision": "補正あり(エッジ一致)" if success else "失敗",
                "decision_step": failure_reason if not success else "エッジ一致で補正",
                "failure_reason": failure_reason,
                "matches_after": 0,
                "inlier_ratio_after": 0.0,
                "match_improvement": 0.0,
                "inlier_improvement": 0.0,
            }
            if success:
                if align_compare_mode:
                    cmp = _decide_alignment_for_edge_bbox(imageB, aligned_imageB, rotation_deg, translation_px)
                    align_info["decision"] = cmp["decision"]
                    align_info["decision_step"] = cmp["decision_step"]
                    align_info["applied"] = cmp["use_aligned"]
                    align_info["compare_metrics"] = {
                        "ssim_original": cmp["ssim_original"],
                        "ssim_aligned": cmp["ssim_aligned"],
                        "mse_original": cmp["mse_original"],
                        "mse_aligned": cmp["mse_aligned"],
                        "ssim_improvement": cmp["ssim_improvement"],
                        "mse_improvement": cmp["mse_improvement"],
                    }
                    if cmp["use_aligned"]:
                        imageB = aligned_imageB
                else:
                    imageB = aligned_imageB
        elif align_mode == "fg_icp":
            aligned_imageB, success, num_matches_before, applied, rotation_deg, translation_px, inlier_ratio_before, failure_reason = align_fg_icp(
                imageA, imageB
            )
            align_info = {
                "enabled": True,
                "success": success,
                "applied": applied,
                "matches_before": num_matches_before,
                "inlier_ratio_before": inlier_ratio_before,
                "rotation_deg": rotation_deg,
                "translation_px": translation_px,
                "decision": "補正あり(fg_mask ICP)" if success else "失敗",
                "decision_step": failure_reason if not success else "fg_mask ICPで補正",
                "failure_reason": failure_reason,
                "matches_after": 0,
                "inlier_ratio_after": 0.0,
                "match_improvement": 0.0,
                "inlier_improvement": 0.0,
            }
            if success:
                if align_compare_mode:
                    cmp = _decide_alignment_for_edge_bbox(imageB, aligned_imageB, rotation_deg, translation_px)
                    align_info["decision"] = cmp["decision"]
                    align_info["decision_step"] = cmp["decision_step"]
                    align_info["applied"] = cmp["use_aligned"]
                    align_info["compare_metrics"] = {
                        "ssim_original": cmp["ssim_original"],
                        "ssim_aligned": cmp["ssim_aligned"],
                        "mse_original": cmp["mse_original"],
                        "mse_aligned": cmp["mse_aligned"],
                        "ssim_improvement": cmp["ssim_improvement"],
                        "mse_improvement": cmp["mse_improvement"],
                    }
                    if cmp["use_aligned"]:
                        imageB = aligned_imageB
                else:
                    imageB = aligned_imageB
        elif align_mode == "bbox":
            aligned_imageB, success, num_matches_before, applied, rotation_deg, translation_px, inlier_ratio_before, failure_reason = align_bbox_based(
                imageA, imageB
            )
            align_info = {
                "enabled": True,
                "success": success,
                "applied": applied,
                "matches_before": num_matches_before,
                "inlier_ratio_before": inlier_ratio_before,
                "rotation_deg": rotation_deg,
                "translation_px": translation_px,
                "decision": "補正あり(外接矩形一致)" if success else "失敗",
                "decision_step": failure_reason if not success else "外接矩形一致で補正",
                "failure_reason": failure_reason,
                "matches_after": 0,
                "inlier_ratio_after": 0.0,
                "match_improvement": 0.0,
                "inlier_improvement": 0.0,
            }
            if success:
                if align_compare_mode:
                    cmp = _decide_alignment_for_edge_bbox(imageB, aligned_imageB, rotation_deg, translation_px)
                    align_info["decision"] = cmp["decision"]
                    align_info["decision_step"] = cmp["decision_step"]
                    align_info["applied"] = cmp["use_aligned"]
                    align_info["compare_metrics"] = {
                        "ssim_original": cmp["ssim_original"],
                        "ssim_aligned": cmp["ssim_aligned"],
                        "mse_original": cmp["mse_original"],
                        "mse_aligned": cmp["mse_aligned"],
                        "ssim_improvement": cmp["ssim_improvement"],
                        "mse_improvement": cmp["mse_improvement"],
                    }
                    if cmp["use_aligned"]:
                        imageB = aligned_imageB
                else:
                    imageB = aligned_imageB
        else:
            # 特徴点マッチング（従来）
            aligned_imageB, success, num_matches_before, applied, rotation_deg, translation_px, inlier_ratio_before, failure_reason = auto_align_images(
                imageA,
                imageB,
                method=align_method,
                max_features=align_max_features,
                match_ratio=align_match_ratio,
                min_rotation=align_min_rotation,
                min_translation=align_min_translation,
            )
            align_info = {
                "enabled": True,
                "success": success,
                "applied": applied,
                "matches_before": num_matches_before,
                "inlier_ratio_before": inlier_ratio_before,
                "rotation_deg": rotation_deg,
                "translation_px": translation_px,
                "decision": None,
                "decision_step": None,
                "failure_reason": failure_reason,
            }
            if not success:
                align_info["decision"] = "失敗"
                align_info["decision_step"] = failure_reason if failure_reason else "特徴点検出またはマッチング失敗"
                align_info["matches_after"] = 0
                align_info["inlier_ratio_after"] = 0.0
                align_info["match_improvement"] = 0.0
                align_info["inlier_improvement"] = 0.0
            elif not applied:
                align_info["decision"] = "補正スキップ"
                align_info["decision_step"] = f"ずれが小さい（回転<{align_min_rotation}度, 移動<{align_min_translation}px）"
                align_info["matches_after"] = num_matches_before
                align_info["inlier_ratio_after"] = inlier_ratio_before
                align_info["match_improvement"] = 0.0
                align_info["inlier_improvement"] = 0.0
            if success and align_compare_mode:
                # 段階的判定モード: 複数指標を段階的にチェック
                print("--- 段階的判定実行中 ---")

                # 補正なしのMSE/SSIM
                procA = _preprocess(imageA)
                procB_original = _preprocess(imageB)
                mse_original = calculate_mse(procA, procB_original)
                ssim_original, _ = ssim(procA, procB_original, full=True)

                # 補正ありのMSE/SSIM
                procB_aligned = _preprocess(aligned_imageB)
                mse_aligned = calculate_mse(procA, procB_aligned)
                ssim_aligned, _ = ssim(procA, procB_aligned, full=True)

                # 改善率を計算
                ssim_improvement = (ssim_aligned - ssim_original) / (ssim_original + 1e-10)
                mse_improvement = (mse_original - mse_aligned) / (mse_original + 1e-10)

                print(f"補正なし: MSE={mse_original:.2f}, SSIM={ssim_original:.4f}")
                print(f"補正あり: MSE={mse_aligned:.2f}, SSIM={ssim_aligned:.4f}")
                print(f"改善率: SSIM={ssim_improvement*100:+.2f}%, MSE={mse_improvement*100:+.2f}%")

                # ステップ0: 補正後の再マッチングチェック（最も直接的な指標）
                if align_rematch_check:
                    print("--- 補正後の再マッチング検証 ---")
                    _, success_rematch, num_matches_after, _, _, _, inlier_ratio_after, rematch_failure_reason = auto_align_images(
                        imageA,
                        aligned_imageB,
                        method=align_method,
                        max_features=align_max_features,
                        match_ratio=align_match_ratio,
                        min_rotation=999,  # 閾値を極端に高くして補正をスキップ
                        min_translation=99999,
                    )

                    if success_rematch:
                        match_improvement = (num_matches_after - num_matches_before) / (num_matches_before + 1e-10)
                        inlier_improvement = inlier_ratio_after - inlier_ratio_before

                        print(f"補正前: マッチ数={num_matches_before}, インライア率={inlier_ratio_before:.2f}")
                        print(f"補正後: マッチ数={num_matches_after}, インライア率={inlier_ratio_after:.2f}")
                        print(f"改善: マッチ数={match_improvement*100:+.2f}%, インライア率={inlier_improvement:+.2f}")

                        # マッチング情報を更新（UI表示用）
                        align_info.update({
                            "matches_after": num_matches_after,
                            "inlier_ratio_after": inlier_ratio_after,
                            "match_improvement": match_improvement,
                            "inlier_improvement": inlier_improvement,
                        })

                        # 判定0-1: マッチング明確改善
                        if match_improvement > align_match_improvement_threshold and inlier_improvement > align_inlier_improvement_threshold:
                            print(f"[判定0] ✓ 補正あり を採用（マッチング明確改善: {match_improvement*100:.2f}% > {align_match_improvement_threshold*100:.1f}%）")
                            imageB = aligned_imageB
                            align_info["decision"] = "補正あり"
                            align_info["decision_step"] = "ステップ0: マッチング改善"
                            need_further_check = False  # 判定確定、ステップ1以降はスキップ
                        # 判定0-2: マッチング減少（微減でも補正なし）
                        elif match_improvement < 0:
                            print(f"[判定0] ✓ 補正なし を採用（マッチング減少: {match_improvement*100:.2f}%、補正が逆効果）")
                            align_info["decision"] = "補正なし"
                            align_info["decision_step"] = "ステップ0: マッチング減少"
                            need_further_check = False  # 判定確定、ステップ1以降はスキップ
                        # 判定0-3: マッチング微増（閾値未満だが増加）
                        else:
                            print(f"[判定0] 判定保留（マッチング微増: {match_improvement*100:.2f}%、閾値未満） → 次のステップへ")
                            # 判定を次のステップに委ねる
                            # ステップ1以降の処理を実行するためにフラグを設定
                            need_further_check = True
                    else:
                        # 補正後にパターンマッチが失敗 → 補正なしを採用
                        detail = rematch_failure_reason if rematch_failure_reason else "補正後の再マッチング失敗"
                        print(f"[判定0] ✓ 補正なし を採用（補正後にマッチング失敗: {detail}）")
                        align_info.update({
                            "matches_after": 0,
                            "inlier_ratio_after": 0.0,
                            "match_improvement": 0.0,
                            "inlier_improvement": 0.0,
                            "rematch_failure_reason": rematch_failure_reason,
                        })
                        align_info["decision"] = "補正なし"
                        align_info["decision_step"] = f"ステップ0: 補正後にマッチング失敗（{detail}）"
                        need_further_check = False  # 補正は適用しない、ステップ1以降はスキップ
                else:
                    need_further_check = True

                # ステップ0で判定できなかった場合のみ、ステップ1以降を実行
                if need_further_check:
                    # ステップ1: SSIMの明確な改善/悪化をチェック
                    if ssim_improvement > align_ssim_improvement_threshold:
                        print(f"[判定1] ✓ 補正あり を採用（明確な改善: SSIM {ssim_improvement*100:.2f}% > {align_ssim_improvement_threshold*100:.1f}%）")
                        imageB = aligned_imageB
                        if align_info:
                            align_info["decision"] = "補正あり"
                            align_info["decision_step"] = "ステップ1: SSIM改善"
                    elif ssim_improvement < align_ssim_degradation_threshold:
                        print(f"[判定1] ✓ 補正なし を採用（明確な悪化: SSIM {ssim_improvement*100:.2f}% < {align_ssim_degradation_threshold*100:.1f}%）")
                        if align_info:
                            align_info["decision"] = "補正なし"
                            align_info["decision_step"] = "ステップ1: SSIM悪化"
                    # ステップ2: 微妙な場合はMSEも確認
                    elif mse_improvement > align_mse_improvement_threshold:
                        print(f"[判定2] ✓ 補正あり を採用（MSE改善: {mse_improvement*100:.2f}% > {align_mse_improvement_threshold*100:.1f}%）")
                        imageB = aligned_imageB
                        if align_info:
                            align_info["decision"] = "補正あり"
                            align_info["decision_step"] = "ステップ2: MSE改善"
                    # ステップ3: それでも微妙なら変換量で判定
                    elif rotation_deg > align_min_rotation or translation_px > align_min_translation:
                        print(f"[判定3] ✓ 補正あり を採用（大きなずれ: 回転={rotation_deg:.2f}度, 平行移動={translation_px:.1f}px）")
                        imageB = aligned_imageB
                        if align_info:
                            align_info["decision"] = "補正あり"
                            align_info["decision_step"] = "ステップ3: 大きなずれ"
                    else:
                        print(f"[判定3] ✓ 補正なし を採用（小さなずれで改善なし）")
                        if align_info:
                            align_info["decision"] = "補正なし"
                            align_info["decision_step"] = "ステップ3: 小さなずれ"
            # フォールバック（edge/bbox）
            fallback_mode = (align_fallback_mode or "").lower() if align_fallback_mode is not None else ""
            if fallback_mode in ("edge", "bbox", "fg_icp"):
                decision = align_info.get("decision") if align_info else None
                should_fallback = (not success) or (align_fallback_on_reject and decision in ("補正なし", "失敗"))
                if should_fallback:
                    print(f"--- 位置合わせフォールバック: {fallback_mode} ---")
                    if fallback_mode == "edge":
                        fb_imageB, fb_success, _, fb_applied, fb_rot, fb_trans, _, fb_reason = align_edge_based(
                            imageA, imageB, canny_low=align_edge_canny_low, canny_high=align_edge_canny_high
                        )
                    elif fallback_mode == "fg_icp":
                        fb_imageB, fb_success, _, fb_applied, fb_rot, fb_trans, _, fb_reason = align_fg_icp(
                            imageA, imageB
                        )
                    else:
                        fb_imageB, fb_success, _, fb_applied, fb_rot, fb_trans, _, fb_reason = align_bbox_based(
                            imageA, imageB
                        )
                    if fb_success and fb_applied:
                        use_fallback = True
                        if _preprocess is not None:
                            procA = _preprocess(imageA)
                            procB_orig = _preprocess(imageB)
                            procB_fb = _preprocess(fb_imageB)
                            mse_orig = calculate_mse(procA, procB_orig)
                            mse_fb = calculate_mse(procA, procB_fb)
                            ssim_orig, _ = ssim(procA, procB_orig, full=True)
                            ssim_fb, _ = ssim(procA, procB_fb, full=True)
                            ssim_improve = (ssim_fb - ssim_orig) / (ssim_orig + 1e-10)
                            mse_improve = (mse_orig - mse_fb) / (mse_orig + 1e-10)
                            use_fallback = (ssim_improve > align_ssim_improvement_threshold) or (mse_improve > align_mse_improvement_threshold)
                            if align_info is not None:
                                align_info["fallback_metrics"] = {
                                    "ssim_orig": float(ssim_orig),
                                    "ssim_fb": float(ssim_fb),
                                    "ssim_improve": float(ssim_improve),
                                    "mse_orig": float(mse_orig),
                                    "mse_fb": float(mse_fb),
                                    "mse_improve": float(mse_improve),
                                }
                        if use_fallback:
                            imageB = fb_imageB
                            if align_info is None:
                                align_info = {"enabled": True}
                            align_info.update({
                                "success": True,
                                "applied": True,
                                "rotation_deg": fb_rot,
                                "translation_px": fb_trans,
                                "decision": f"補正あり(フォールバック:{fallback_mode})",
                                "decision_step": fb_reason or "フォールバックで補正",
                                "fallback_used": True,
                            })
                            print("✓ フォールバック補正を採用")
                        else:
                            if align_info is not None:
                                align_info["fallback_used"] = False
                            print("× フォールバック補正は採用せず（指標改善なし）")
                    else:
                        if align_info is not None:
                            align_info["fallback_used"] = False
                            align_info["fallback_failure_reason"] = fb_reason
                        print(f"× フォールバック失敗: {fb_reason}")
            elif success:
                # 閾値モード: 変換量に基づいて判定
                if applied:
                    print(f"✓ 位置合わせ成功・補正適用 (マッチ数: {num_matches_before})")
                    imageB = aligned_imageB
                else:
                    print(f"✓ 位置合わせ成功・補正不要 (マッチ数: {num_matches_before})")
            else:
                print(f"✗ 位置合わせ失敗、元の画像を使用 (マッチ数: {num_matches_before})")

        # --- ECC精密合わせ（既存alignの後にサブピクセル仕上げ） ---
        align_ecc_refine_enabled = kwargs.get("align_ecc_refine_enabled", False)
        if align_ecc_refine_enabled:
            ecc_blur = int(kwargs.get("align_ecc_blur_ksize", 5))
            ecc_max_iter = int(kwargs.get("align_ecc_max_iter", 1000))
            ecc_eps = float(kwargs.get("align_ecc_eps", 1e-6))
            ecc_warp_mode = kwargs.get("align_ecc_warp_mode", "affine")
            print("--- ECC精密合わせ実行中 ---")
            ecc_aligned, ecc_success, ecc_score, ecc_trans, ecc_rot, ecc_warp = align_ecc_refine(
                imageA, imageB,
                blur_ksize=ecc_blur,
                max_iter=ecc_max_iter,
                eps=ecc_eps,
                warp_mode=ecc_warp_mode,
            )
            if ecc_success:
                # SSIM比較して改善した場合のみ採用
                procA = _preprocess(imageA) if _preprocess is not None else cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                procB_before = _preprocess(imageB) if _preprocess is not None else cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                procB_ecc = _preprocess(ecc_aligned) if _preprocess is not None else cv2.cvtColor(ecc_aligned, cv2.COLOR_BGR2GRAY)
                ssim_before_ecc, _ = ssim(procA, procB_before, full=True)
                ssim_after_ecc, _ = ssim(procA, procB_ecc, full=True)
                ecc_ssim_improvement = ssim_after_ecc - ssim_before_ecc
                print(f"  ECC: cc={ecc_score:.4f}, tx={ecc_trans[0]:.2f}px, ty={ecc_trans[1]:.2f}px, rot={ecc_rot:.4f}deg")
                print(f"  SSIM: {ssim_before_ecc:.4f} → {ssim_after_ecc:.4f} ({ecc_ssim_improvement:+.4f})")
                if ecc_ssim_improvement > 0:
                    imageB = ecc_aligned
                    if align_info is None:
                        align_info = {"enabled": True}
                    align_info["ecc_refine"] = {
                        "applied": True,
                        "ecc_score": float(ecc_score),
                        "tx": float(ecc_trans[0]),
                        "ty": float(ecc_trans[1]),
                        "rotation_deg": float(ecc_rot),
                        "ssim_before": float(ssim_before_ecc),
                        "ssim_after": float(ssim_after_ecc),
                        "ssim_improvement": float(ecc_ssim_improvement),
                    }
                    print("  ✓ ECC精密合わせ採用（SSIM改善）")
                else:
                    if align_info is None:
                        align_info = {"enabled": True}
                    align_info["ecc_refine"] = {
                        "applied": False,
                        "ecc_score": float(ecc_score),
                        "ssim_before": float(ssim_before_ecc),
                        "ssim_after": float(ssim_after_ecc),
                        "ssim_improvement": float(ecc_ssim_improvement),
                        "reason": "SSIM悪化のため不採用",
                    }
                    print("  × ECC精密合わせ不採用（SSIM悪化）")
            else:
                if align_info is None:
                    align_info = {"enabled": True}
                align_info["ecc_refine"] = {"applied": False, "reason": "ECC収束失敗"}
                print("  × ECC精密合わせ失敗")

        return imageB, align_info

    def _apply_roi(self, imageA, imageB, roi_x, roi_y, roi_w, roi_h):
        """ROI指定検査の処理"""
        h, w = imageA.shape[:2]
        x1 = max(0, min(roi_x, w - 1))
        y1 = max(0, min(roi_y, h - 1))
        w1 = min(roi_w, w - x1)
        h1 = min(roi_h, h - y1)
        roi_info = None
        if w1 > 0 and h1 > 0:
            imageA = imageA[y1 : y1 + h1, x1 : x1 + w1]
            imageB = imageB[y1 : y1 + h1, x1 : x1 + w1]
            roi_info = {"enabled": True, "x": int(x1), "y": int(y1), "w": int(w1), "h": int(h1)}
            print(f"✓ ROIで検査領域を限定（x={x1}, y={y1}, w={w1}, h={h1}）")
        return imageA, imageB, roi_info

    def _detect_differences(self, diff, imageA, imageB, **kwargs):
        """前景マスク、背景重なりチェック、BBOX検出の処理"""
        diff_before_fg_mask = kwargs.get("diff_before_fg_mask")
        use_foreground_mask = kwargs.get("use_foreground_mask", False)
        fg_edge_exclude_enabled = kwargs.get("fg_edge_exclude_enabled", False)
        fg_edge_exclude_iter = kwargs.get("fg_edge_exclude_iter", 2)
        fg_edge_overlap_exclude_enabled = kwargs.get("fg_edge_overlap_exclude_enabled", False)
        fg_edge_raise_thresh_enabled = kwargs.get("fg_edge_raise_thresh_enabled", False)
        fg_edge_raise_thresh_offset = kwargs.get("fg_edge_raise_thresh_offset", 0)
        fg_contour_mask_enabled = kwargs.get("fg_contour_mask_enabled", False)
        fg_contour_mask_canny_low = kwargs.get("fg_contour_mask_canny_low", 50)
        fg_contour_mask_canny_high = kwargs.get("fg_contour_mask_canny_high", 150)
        fg_contour_mask_kernel = kwargs.get("fg_contour_mask_kernel", 5)
        fg_contour_mask_close_iter = kwargs.get("fg_contour_mask_close_iter", 3)
        fg_contour_mask_dilate_iter = kwargs.get("fg_contour_mask_dilate_iter", 1)
        fg_contour_mask_erode_iter = kwargs.get("fg_contour_mask_erode_iter", 5)
        fg_contour_mask_min_area_ratio = kwargs.get("fg_contour_mask_min_area_ratio", 0.05)
        fg_contour_mask_max_area_ratio = kwargs.get("fg_contour_mask_max_area_ratio", 0.98)
        fg_contour_mask_fallback_to_fg = kwargs.get("fg_contour_mask_fallback_to_fg", True)
        use_background_overlap_check = kwargs.get("use_background_overlap_check", False)
        background_overlap_max_ratio = kwargs.get("background_overlap_max_ratio", 0.5)
        background_overlap_use_diff = kwargs.get("background_overlap_use_diff", False)
        foreground_mask_dilate_iter = kwargs.get("foreground_mask_dilate_iter", 0)
        foreground_mask_kernel = kwargs.get("foreground_mask_kernel", 15)
        foreground_mask_keep_ratio = kwargs.get("foreground_mask_keep_ratio", 0.0)
        foreground_xor_add = kwargs.get("foreground_xor_add", False)
        detect_bbox = kwargs.get("detect_bbox", True)
        diff_thresh = kwargs.get("diff_thresh", 13)
        min_area = kwargs.get("min_area", 1500)
        morph_kernel = kwargs.get("morph_kernel", 20)
        max_boxes = kwargs.get("max_boxes", 6)
        bbox_edge_ignore_ratio = kwargs.get("bbox_edge_ignore_ratio", None)
        bbox_min_area_relax_ratio = kwargs.get("bbox_min_area_relax_ratio", None)
        bbox_min_width = kwargs.get("bbox_min_width", 0)
        bbox_drop_band_aspect = float(kwargs.get("bbox_drop_band_aspect", 0))
        bbox_drop_band_max_fill = float(kwargs.get("bbox_drop_band_max_fill", 0))
        bbox_drop_band_aspect_high = float(kwargs.get("bbox_drop_band_aspect_high", 0))
        bbox_drop_band_max_fill_high = float(kwargs.get("bbox_drop_band_max_fill_high", 0))
        bbox_drop_sparse_large_cover = float(kwargs.get("bbox_drop_sparse_large_cover", 0))
        bbox_drop_sparse_large_max_fill = float(kwargs.get("bbox_drop_sparse_large_max_fill", 0))
        bbox_drop_band_min_area_ratio = float(kwargs.get("bbox_drop_band_min_area_ratio", 0))
        bbox_fg_edge_suppress_enabled = kwargs.get("bbox_fg_edge_suppress_enabled", False)
        bbox_fg_edge_suppress_width = int(kwargs.get("bbox_fg_edge_suppress_width", 15))
        bbox_fg_edge_suppress_offset = int(kwargs.get("bbox_fg_edge_suppress_offset", 30))
        bbox_close_iter = kwargs.get("bbox_close_iter", 4)
        bbox_open_iter = kwargs.get("bbox_open_iter", 1)
        bbox_connectivity = kwargs.get("bbox_connectivity", 8)
        bbox_edge_min_fill_ratio = kwargs.get("bbox_edge_min_fill_ratio", None)
        bbox_edge_cut_enabled = kwargs.get("bbox_edge_cut_enabled", False)
        bbox_edge_cut_kernel = kwargs.get("bbox_edge_cut_kernel", 3)
        bbox_edge_cut_iter = kwargs.get("bbox_edge_cut_iter", 1)
        edge_sep_enabled = kwargs.get("edge_sep_enabled", True)
        edge_sep_min_dist = kwargs.get("edge_sep_min_dist", 3)
        edge_sep_max_dist = kwargs.get("edge_sep_max_dist", 30)
        edge_sep_step = kwargs.get("edge_sep_step", 1)
        bbox_drop_band_fill_ratio = kwargs.get("bbox_drop_band_fill_ratio", None)
        bbox_drop_band_width_ratio = kwargs.get("bbox_drop_band_width_ratio", None)
        bbox_drop_band_height_ratio = kwargs.get("bbox_drop_band_height_ratio", None)
        bbox_edge_roi_enabled = kwargs.get("bbox_edge_roi_enabled", False)
        bbox_edge_roi_canny_low = kwargs.get("bbox_edge_roi_canny_low", 50)
        bbox_edge_roi_canny_high = kwargs.get("bbox_edge_roi_canny_high", 150)
        bbox_edge_roi_kernel = kwargs.get("bbox_edge_roi_kernel", 5)
        bbox_edge_roi_close_iter = kwargs.get("bbox_edge_roi_close_iter", 2)
        bbox_edge_roi_dilate_iter = kwargs.get("bbox_edge_roi_dilate_iter", 1)
        bbox_edge_roi_min_area_ratio = kwargs.get("bbox_edge_roi_min_area_ratio", 0.1)
        bbox_edge_roi_max_area_ratio = kwargs.get("bbox_edge_roi_max_area_ratio", 0.95)
        bbox_edge_roi_require_closed = kwargs.get("bbox_edge_roi_require_closed", True)
        bbox_edge_roi_border_margin = kwargs.get("bbox_edge_roi_border_margin", 2)
        bbox_inner_erode_iter = kwargs.get("bbox_inner_erode_iter", 0)
        bbox_inner_overlap_ratio = kwargs.get("bbox_inner_overlap_ratio", None)
        bbox_rescue_drop_band_fill_ratio = kwargs.get("bbox_rescue_drop_band_fill_ratio", None)
        bbox_rescue_drop_band_width_ratio = kwargs.get("bbox_rescue_drop_band_width_ratio", None)
        bbox_rescue_drop_band_height_ratio = kwargs.get("bbox_rescue_drop_band_height_ratio", None)
        bbox_rescue_thresh_offset = kwargs.get("bbox_rescue_thresh_offset", 0)
        bbox_rescue_morph_kernel = kwargs.get("bbox_rescue_morph_kernel", None)
        bbox_rescue_close_iter = kwargs.get("bbox_rescue_close_iter", None)
        bbox_rescue_open_iter = kwargs.get("bbox_rescue_open_iter", None)
        bbox_rescue_edge_erode_iter = kwargs.get("bbox_rescue_edge_erode_iter", 1)
        bbox_rescue_use_fg_edge_band = kwargs.get("bbox_rescue_use_fg_edge_band", False)
        bbox_rescue_min_area_ratio = kwargs.get("bbox_rescue_min_area_ratio", None)
        hole_mask_enabled = kwargs.get("hole_mask_enabled", False)
        hole_mask_source = kwargs.get("hole_mask_source", "edge_roi_or_foreground")
        hole_mask_apply_mode = kwargs.get("hole_mask_apply_mode", "master")
        hole_mask_method = kwargs.get("hole_mask_method", "auto")
        hole_bg_border_ratio = kwargs.get("hole_bg_border_ratio", 0.05)
        hole_bg_dist_percentile = kwargs.get("hole_bg_dist_percentile", 95.0)
        hole_bg_dist_scale = kwargs.get("hole_bg_dist_scale", 1.5)
        hole_bg_dist_min = kwargs.get("hole_bg_dist_min", 6.0)
        hole_bg_dist_max = kwargs.get("hole_bg_dist_max", 60.0)
        hole_inner_erode_iter = kwargs.get("hole_inner_erode_iter", 1)
        hole_expand_iter = kwargs.get("hole_expand_iter", 0)
        hole_edge_exclude_enabled = kwargs.get("hole_edge_exclude_enabled", False)
        hole_edge_exclude_iter = kwargs.get("hole_edge_exclude_iter", 2)
        fg_hole_edge_exclude_enabled = kwargs.get("fg_hole_edge_exclude_enabled", False)
        fg_hole_edge_exclude_dilate = int(kwargs.get("fg_hole_edge_exclude_dilate", 25))
        fg_hole_edge_exclude_min_area_ratio = float(kwargs.get("fg_hole_edge_exclude_min_area_ratio", 0.05))
        hole_bbox_filter_enabled = kwargs.get("hole_bbox_filter_enabled", False)
        hole_bbox_filter_dilate_iter = kwargs.get("hole_bbox_filter_dilate_iter", 0)
        hole_bbox_filter_overlap_ratio = kwargs.get("hole_bbox_filter_overlap_ratio", 0.2)
        hole_bbox_filter_max_area_ratio = kwargs.get("hole_bbox_filter_max_area_ratio", 0.01)
        hole_bbox_filter_use_center = kwargs.get("hole_bbox_filter_use_center", True)
        hole_min_area_ratio = kwargs.get("hole_min_area_ratio", 0.0002)
        hole_max_area_ratio = kwargs.get("hole_max_area_ratio", 0.08)
        hole_min_fill_ratio = kwargs.get("hole_min_fill_ratio", 0.5)
        hole_aspect_min = kwargs.get("hole_aspect_min", 0.6)
        hole_aspect_max = kwargs.get("hole_aspect_max", 1.4)
        hole_shrink_iter = kwargs.get("hole_shrink_iter", 1)
        hole_max_total_ratio = kwargs.get("hole_max_total_ratio", 0.15)
        strong_bbox_enabled = kwargs.get("strong_bbox_enabled", False)
        strong_bbox_percentile = kwargs.get("strong_bbox_percentile", 99.5)
        strong_bbox_offset = kwargs.get("strong_bbox_offset", 10)
        strong_bbox_min_area_ratio = kwargs.get("strong_bbox_min_area_ratio", 0.3)
        strong_bbox_min_area = kwargs.get("strong_bbox_min_area", 0)
        strong_bbox_morph_kernel = kwargs.get("strong_bbox_morph_kernel", 5)
        strong_bbox_close_iter = kwargs.get("strong_bbox_close_iter", 1)
        strong_bbox_open_iter = kwargs.get("strong_bbox_open_iter", 0)
        strong_bbox_max_boxes = kwargs.get("strong_bbox_max_boxes", 5)
        strong_bbox_use_edge_filter = kwargs.get("strong_bbox_use_edge_filter", True)
        strong_bbox_edge_roi_enabled = kwargs.get("strong_bbox_edge_roi_enabled", False)
        strong_bbox_edge_roi_canny_low = kwargs.get("strong_bbox_edge_roi_canny_low", 50)
        strong_bbox_edge_roi_canny_high = kwargs.get("strong_bbox_edge_roi_canny_high", 150)
        strong_bbox_edge_roi_kernel = kwargs.get("strong_bbox_edge_roi_kernel", 5)
        strong_bbox_edge_roi_close_iter = kwargs.get("strong_bbox_edge_roi_close_iter", 2)
        strong_bbox_edge_roi_dilate_iter = kwargs.get("strong_bbox_edge_roi_dilate_iter", 1)
        strong_bbox_edge_roi_min_area_ratio = kwargs.get("strong_bbox_edge_roi_min_area_ratio", 0.1)
        strong_bbox_edge_roi_max_area_ratio = kwargs.get("strong_bbox_edge_roi_max_area_ratio", 0.95)
        strong_bbox_edge_roi_require_closed = kwargs.get("strong_bbox_edge_roi_require_closed", True)
        strong_bbox_edge_roi_border_margin = kwargs.get("strong_bbox_edge_roi_border_margin", 2)
        strong_bbox_drop_band_fill_ratio = kwargs.get("strong_bbox_drop_band_fill_ratio", None)
        strong_bbox_drop_band_width_ratio = kwargs.get("strong_bbox_drop_band_width_ratio", None)
        strong_bbox_drop_band_height_ratio = kwargs.get("strong_bbox_drop_band_height_ratio", None)
        strong_bbox_min_diff_max = kwargs.get("strong_bbox_min_diff_max", 0)
        strong_bbox_edge_margin = kwargs.get("strong_bbox_edge_margin", 0)
        strong_bbox_min_fill = kwargs.get("strong_bbox_min_fill", 0)
        strong_bbox_debug_log_path = kwargs.get("strong_bbox_debug_log_path", None)
        bbox_debug_log_path = kwargs.get("bbox_debug_log_path", None)
        diff_original = kwargs.get("diff_original", None)
        _save_stage = kwargs.get("_save_stage")
        ksize = max(3, int(foreground_mask_kernel) if int(foreground_mask_kernel) % 2 else int(foreground_mask_kernel) + 1)
        mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        master_fg = None
        compare_fg = None
        fg_edge_band = None
        fg_edge_overlap = None
        edge_band_for_thresh = None

        def _make_edge_roi_mask(
            src_image,
            canny_low,
            canny_high,
            kernel_size,
            close_iter,
            dilate_iter,
            min_area_ratio,
            max_area_ratio,
            require_closed,
            border_margin,
        ):
            try:
                if len(src_image.shape) == 3:
                    edge_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
                else:
                    edge_gray = src_image.copy()
                edges = cv2.Canny(edge_gray, int(canny_low), int(canny_high))
                ksize = max(3, int(kernel_size) if int(kernel_size) % 2 else int(kernel_size) + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                if close_iter and close_iter > 0:
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
                if dilate_iter and dilate_iter > 0:
                    edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return None, None
                cnt = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(edge_gray)
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else None
                if ratio is None:
                    return None, None
                if ratio < float(min_area_ratio) or ratio > float(max_area_ratio):
                    return None, ratio
                if require_closed:
                    x, y, w, h = cv2.boundingRect(cnt)
                    margin = int(border_margin)
                    if x <= margin or y <= margin or (x + w) >= (edge_gray.shape[1] - margin) or (y + h) >= (edge_gray.shape[0] - margin):
                        return None, ratio
                return mask, ratio
            except Exception:
                return None, None

        def _make_contour_interior_mask(
            src_image,
            fg_mask,
            canny_low,
            canny_high,
            kernel_size,
            close_iter,
            dilate_iter,
            erode_iter,
            min_area_ratio,
            max_area_ratio,
            fallback_to_fg,
        ):
            """前景マスクを収縮して内部保護マスクを作成する。

            Canny エッジ検出で板金輪郭を検出し、fg_mask との AND で
            精密な内部マスクを作成する。Canny 輪郭が fg_mask の 80%
            以上をカバーしていない場合は fg_mask をそのまま使用する。
            最終的に erode_iter 分だけ収縮して境界ずれの差分を除外する。

            Parameters
            ----------
            src_image : ndarray
                マスター画像（BGR or グレースケール）
            fg_mask : ndarray or None
                Otsu ベースの前景マスク
            canny_low, canny_high : int
                Canny エッジ検出の低/高閾値
            kernel_size : int
                モルフォロジーカーネルサイズ
            close_iter : int
                エッジ閉合のクローズ回数
            dilate_iter : int
                エッジ膨張回数
            erode_iter : int
                内部マスク収縮回数（境界ゾーンを除外）
            min_area_ratio, max_area_ratio : float
                面積比フィルタ
            fallback_to_fg : bool
                Canny 失敗時に fg_mask の収縮版へフォールバック

            Returns
            -------
            ndarray or None
                収縮済み内部マスク（255=内部, 0=外側/境界）
            """
            base_mask = None

            # Canny エッジ検出で輪郭を検出し、fg_mask より精密な輪郭を試みる
            try:
                if len(src_image.shape) == 3:
                    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = src_image.copy()
                edges = cv2.Canny(gray, int(canny_low), int(canny_high))
                ksize = max(3, int(kernel_size) if int(kernel_size) % 2 else int(kernel_size) + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                if close_iter and close_iter > 0:
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=int(close_iter))
                if dilate_iter and dilate_iter > 0:
                    edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    filled = np.zeros_like(gray)
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
                    ratio = float(np.count_nonzero(filled)) / float(filled.size) if filled.size else 0.0
                    if min_area_ratio <= ratio <= max_area_ratio:
                        # fg_mask との AND で安全ネット
                        if fg_mask is not None:
                            candidate = cv2.bitwise_and(filled, fg_mask)
                        else:
                            candidate = filled
                        # Canny 輪郭が fg_mask の 80% 以上をカバーしているか検証
                        # カバー率が低い = 複雑形状で最大輪郭が板金全体を囲めていない
                        if fg_mask is not None:
                            fg_nz = np.count_nonzero(fg_mask)
                            cand_nz = np.count_nonzero(candidate)
                            if fg_nz > 0 and cand_nz / float(fg_nz) >= 0.8:
                                base_mask = candidate
                        else:
                            base_mask = candidate
            except Exception:
                pass

            # フォールバック: Canny 失敗 or カバー不足時は fg_mask を使用
            if base_mask is None and fg_mask is not None:
                if fallback_to_fg:
                    base_mask = fg_mask.copy()

            if base_mask is None:
                return None

            # 収縮して境界ゾーンを除外
            if erode_iter and erode_iter > 0:
                k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                base_mask = cv2.erode(base_mask, k_erode, iterations=int(erode_iter))

            return base_mask

        def _erode_separate_interior(
            diff_gray,
            diff_thresh,
            max_erode=15,
            min_interior_area=50,
            connectivity=4,
        ):
            """差分マスクを段階的に erode し、辺から分離した内部成分を救済する。

            アルゴリズム:
              1. diff を閾値化してバイナリマスクを作成
              2. 1px ずつ erode しながら連結成分を監視
              3. 辺に接触する成分と接触しない成分に分離したら停止
              4. 分離した内部成分を erode 回数分 dilate して元サイズに復元
              5. 元マスクでクリップして返す

            Parameters
            ----------
            diff_gray : ndarray
                差分画像 (uint8)。fg_mask 適用済み。
            diff_thresh : int
                二値化の閾値
            max_erode : int
                最大 erode 回数
            min_interior_area : int
                内部成分として認める最小面積
            connectivity : int
                連結成分の接続性 (4 or 8)

            Returns
            -------
            interior_mask : ndarray or None
                内部領域のマスク (uint8, 0/255)。分離できなかった場合 None。
            info : dict
                診断情報
            """
            info = {
                "method": "erode_separate",
                "separated": False,
                "skipped": False,
                "skip_reason": None,
                "erode_count": 0,
                "interior_components": 0,
                "interior_total_area": 0,
                "edge_sides": [],
                "initial_edge_components": 0,
                "initial_interior_components": 0,
            }

            # 閾値化
            _, bin_mask = cv2.threshold(diff_gray, int(diff_thresh), 255, cv2.THRESH_BINARY)
            if np.count_nonzero(bin_mask) == 0:
                return None, info

            h_img, w_img = bin_mask.shape[:2]
            image_area = float(h_img * w_img)
            k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # 元マスクの辺接触状態を確認
            def _touching_sides(mask):
                """マスクが画像のどの辺に接触しているかを返す"""
                sides = set()
                if np.any(mask[0, :] > 0):
                    sides.add("top")
                if np.any(mask[h_img - 1, :] > 0):
                    sides.add("bottom")
                if np.any(mask[:, 0] > 0):
                    sides.add("left")
                if np.any(mask[:, w_img - 1] > 0):
                    sides.add("right")
                return sides

            original_sides = _touching_sides(bin_mask)
            if not original_sides:
                # 辺に接触していない → 全て内部なので処理不要
                info["skipped"] = True
                info["skip_reason"] = "no_edge_contact"
                return None, info

            info["edge_sides"] = list(original_sides)

            # === 発動条件チェック ===
            # 連結成分を分析して、erode分離が必要かどうか判定する
            num_init, labels_init, stats_init, _ = cv2.connectedComponentsWithStats(
                bin_mask, connectivity=int(connectivity)
            )
            edge_components = []   # 辺に接触する成分
            free_components = []   # 辺に接触しない成分
            for lbl in range(1, num_init):
                x, y, bw, bh, area = stats_init[lbl]
                lbl_mask = (labels_init == lbl).astype(np.uint8) * 255
                sides = _touching_sides(lbl_mask)
                if sides:
                    edge_components.append({"label": lbl, "area": area, "bbox": (x, y, bw, bh), "sides": sides})
                else:
                    if area >= min_interior_area:
                        free_components.append({"label": lbl, "area": area, "bbox": (x, y, bw, bh)})

            info["initial_edge_components"] = len(edge_components)
            info["initial_interior_components"] = len(free_components)

            if not edge_components:
                # 辺に接触する成分がない → 処理不要
                info["skipped"] = True
                info["skip_reason"] = "no_edge_components"
                return None, info

            # 辺に接触する成分のうち最大のものの cover 比を確認
            largest_edge = max(edge_components, key=lambda c: c["area"])
            le_x, le_y, le_w, le_h = largest_edge["bbox"]
            cover = (le_w * le_h) / image_area if image_area > 0 else 0
            fill = largest_edge["area"] / float(le_w * le_h) if le_w and le_h else 0

            # 発動条件: 辺接触成分が広範囲かつスカスカ（= 境界差分が内部を巻き込んでいる）
            # cover >= 0.3 かつ fill < 0.5 → 広い範囲をカバーするが中身がスカスカ
            if cover < 0.3 or fill >= 0.5:
                info["skipped"] = True
                info["skip_reason"] = f"edge_component_small_or_dense cover={cover:.3f} fill={fill:.3f}"
                return None, info

            # 既に辺に接触しない独立成分がある場合も分離不要
            # （元々分離しているなら erode で切る必要がない）
            if free_components and not edge_components:
                info["skipped"] = True
                info["skip_reason"] = "already_separated"
                return None, info

            # 段階的 erode → 分離検出
            eroded = bin_mask.copy()
            interior_labels_mask = None

            for step in range(1, max_erode + 1):
                eroded = cv2.erode(eroded, k_erode, iterations=1)
                if np.count_nonzero(eroded) == 0:
                    break

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    eroded, connectivity=int(connectivity)
                )

                # 辺に接触しない成分を収集
                interior_components = []
                for lbl in range(1, num_labels):
                    x, y, bw, bh, area = stats[lbl]
                    if area < min_interior_area:
                        continue
                    lbl_mask = (labels == lbl).astype(np.uint8) * 255
                    sides = _touching_sides(lbl_mask)
                    if not sides:
                        # 辺に接触しない = 内部成分
                        interior_components.append({
                            "label": lbl,
                            "bbox": (x, y, bw, bh),
                            "area": area,
                        })

                if interior_components:
                    # 分離成功
                    interior_labels_mask = np.zeros_like(eroded)
                    for comp in interior_components:
                        interior_labels_mask[labels == comp["label"]] = 255

                    info["separated"] = True
                    info["erode_count"] = step
                    info["interior_components"] = len(interior_components)
                    info["interior_total_area"] = sum(c["area"] for c in interior_components)
                    break

            if not info["separated"]:
                return None, info

            # === 内部成分を dilate で元サイズに復元 ===
            # erode で分離した内部成分は erode 回数分だけ縮んでいる。
            # 同じ回数 dilate して元のサイズに戻し、元マスクでクリップする。
            # これにより部品差分の形状が削れるのを防ぐ。
            step = info["erode_count"]
            restored = cv2.dilate(interior_labels_mask, k_erode, iterations=step)
            # 元マスクの範囲にクリップ（元の形を超えないように）
            result = cv2.bitwise_and(restored, bin_mask)

            if np.count_nonzero(result) == 0:
                return None, info

            return result, info

        def _build_hole_mask(
            src_bgr,
            base_mask,
            method,
            min_area_ratio,
            max_area_ratio,
            min_fill,
            aspect_min,
            aspect_max,
            shrink_iter,
            max_total_ratio,
            bg_border_ratio,
            bg_dist_percentile,
            bg_dist_scale,
            bg_dist_min,
            bg_dist_max,
            inner_erode_iter,
            expand_iter,
        ):
            if base_mask is None:
                return None, "base_mask_none"
            try:
                h, w = base_mask.shape[:2]
                if h == 0 or w == 0:
                    return None, "invalid_size"

                def _filter_components(holes):
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
                    image_area = float(h * w)
                    keep = np.zeros_like(holes)
                    total_area = 0.0
                    kept = 0
                    for label in range(1, num_labels):
                        x, y, bw, bh, area = stats[label]
                        if area <= 0:
                            continue
                        area_ratio = area / image_area
                        if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
                            continue
                        fill_ratio = float(area) / float(bw * bh) if bw > 0 and bh > 0 else 0.0
                        if fill_ratio < float(min_fill):
                            continue
                        aspect = (bw / float(bh)) if bh > 0 else 0.0
                        if aspect < float(aspect_min) or aspect > float(aspect_max):
                            continue
                        keep[labels == label] = 255
                        total_area += float(area)
                        kept += 1
                    if kept == 0:
                        return None, "no_holes_kept"
                    total_ratio = total_area / image_area if image_area > 0 else 0.0
                    if total_ratio > float(max_total_ratio):
                        return None, f"holes_too_large:{total_ratio:.3f}"
                    if shrink_iter and shrink_iter > 0:
                        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        keep = cv2.erode(keep, k2, iterations=int(shrink_iter))
                    if expand_iter and expand_iter > 0:
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        keep = cv2.dilate(keep, k3, iterations=int(expand_iter))
                    return keep, f"kept={kept} total_ratio={total_ratio:.4f}"

                def _holes_from_background():
                    if src_bgr is None:
                        return None, "bg_no_src"
                    if len(src_bgr.shape) == 2:
                        bgr = cv2.cvtColor(src_bgr, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = src_bgr
                    hh, ww = bgr.shape[:2]
                    border = max(2, int(min(hh, ww) * float(bg_border_ratio)))
                    if border * 2 >= min(hh, ww):
                        return None, "bg_border_too_large"
                    border_mask = np.zeros((hh, ww), dtype=np.uint8)
                    border_mask[:border, :] = 255
                    border_mask[-border:, :] = 255
                    border_mask[:, :border] = 255
                    border_mask[:, -border:] = 255
                    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                    border_pixels = lab[border_mask > 0]
                    if border_pixels.size == 0:
                        return None, "bg_no_samples"
                    bg_color = np.median(border_pixels, axis=0)
                    dist = np.sqrt(np.sum((lab - bg_color) ** 2, axis=2))
                    border_dist = dist[border_mask > 0]
                    if border_dist.size == 0:
                        return None, "bg_no_dists"
                    thr = np.percentile(border_dist, float(bg_dist_percentile)) * float(bg_dist_scale)
                    if bg_dist_min is not None:
                        thr = max(thr, float(bg_dist_min))
                    if bg_dist_max is not None:
                        thr = min(thr, float(bg_dist_max))
                    bg_like = (dist <= thr).astype(np.uint8) * 255
                    inner = base_mask
                    if inner_erode_iter and inner_erode_iter > 0:
                        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        inner = cv2.erode(base_mask, k2, iterations=int(inner_erode_iter))
                    if inner is None or np.count_nonzero(inner) == 0:
                        return None, "bg_no_inner"
                    holes = cv2.bitwise_and(bg_like, inner)
                    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k3, iterations=1)
                    if _save_stage is not None:
                        _save_stage("10_mask_holes_bg", single=holes)
                    return holes, f"bg_thr={thr:.2f}"

                method = (method or "").lower()
                if method in ("background", "auto"):
                    holes_bg, bg_info = _holes_from_background()
                    if holes_bg is not None:
                        keep, info = _filter_components(holes_bg)
                        if keep is not None:
                            return keep, f"method=background {bg_info} {info}"
                    if method == "background":
                        return None, f"background:{bg_info}"

                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                filled = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, k, iterations=2)
                holes = cv2.bitwise_and(filled, cv2.bitwise_not(base_mask))
                holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k, iterations=1)
                keep, info = _filter_components(holes)
                if keep is None:
                    return None, f"filled:{info}"
                return keep, f"method=filled {info}"
            except Exception:
                return None, "exception"

        def _dilate_mask(mask):
            if mask is None:
                return None
            if foreground_mask_dilate_iter and foreground_mask_dilate_iter > 0:
                return cv2.dilate(mask, mask_kernel, iterations=int(foreground_mask_dilate_iter))
            return mask

        # 前景マスク（ON時のみ）: 基準画像から前景領域を抽出し、差分・BBOXを前景内に限定
        fg_mask = None
        if use_foreground_mask:
            from src.core.segmentation import get_foreground_mask
            fg_mask = _dilate_mask(
                get_foreground_mask(
                    imageA,
                    blur_ksize=5,
                    keep_largest_ratio=foreground_mask_keep_ratio,
                    preclose_kernel=foreground_mask_kernel,
                )
            )
            master_fg = fg_mask
            if fg_mask is not None:
                diff = cv2.bitwise_and(diff, diff, mask=fg_mask)
                if _save_stage is not None:
                    _save_stage("09_fg_mask", single=fg_mask)

        bbox_inner_mask = None
        bbox_inner_mask_source = None
        if fg_mask is not None and bbox_inner_overlap_ratio is not None:
            inner = fg_mask
            if bbox_inner_erode_iter and bbox_inner_erode_iter > 0:
                k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                inner = cv2.erode(fg_mask, k_inner, iterations=int(bbox_inner_erode_iter))
            if inner is not None and np.count_nonzero(inner) > 0:
                bbox_inner_mask = inner
                bbox_inner_mask_source = "fg_mask"

        _save_stage("09_after_fg_mask", single=diff)

        # 背景の重なりチェック（ON時のみ）: 両方で前景抽出し、マスターの背景だった場所に比較の前景/閾値以上の差分が入れば検出に含める
        if use_background_overlap_check:
            from src.core.segmentation import get_foreground_mask
            if master_fg is None:
                master_fg = _dilate_mask(
                    get_foreground_mask(
                        imageA,
                        blur_ksize=5,
                        keep_largest_ratio=foreground_mask_keep_ratio,
                        preclose_kernel=foreground_mask_kernel,
                    )
                )
            if compare_fg is None:
                compare_fg = _dilate_mask(
                    get_foreground_mask(
                        imageB,
                        blur_ksize=5,
                        keep_largest_ratio=foreground_mask_keep_ratio,
                        preclose_kernel=foreground_mask_kernel,
                    )
                )
            if master_fg is not None and compare_fg is not None:
                master_bg = cv2.bitwise_not(master_fg)
                # 閾値以上の差分は前景マスク前の差分で判定（前景マスク後は背景領域が0のため）
                diff_for_intrusion = diff_before_fg_mask if diff_before_fg_mask is not None else diff
                intrusion = (master_bg > 0) & (compare_fg > 0)
                if background_overlap_use_diff and diff_for_intrusion is not None:
                    intrusion |= (master_bg > 0) & (diff_for_intrusion >= diff_thresh)
                intrusion_ratio = float(np.count_nonzero(intrusion)) / float(intrusion.size)
                if background_overlap_max_ratio is not None and intrusion_ratio > background_overlap_max_ratio:
                    print(f"⚠ 背景の重なりチェックをスキップ（侵入比率 {intrusion_ratio:.2f} > 上限 {background_overlap_max_ratio:.2f}）")
                else:
                    intrusion_uint8 = np.where(intrusion, 255, 0).astype(np.uint8)
                    # fg_maskエッジ帯を除外して注入（サンプル境目の差分混入を防ぐ）
                    if bbox_fg_edge_suppress_enabled and master_fg is not None:
                        _k_intr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        _fg_ero_intr = cv2.erode(master_fg, _k_intr, iterations=max(1, bbox_fg_edge_suppress_width))
                        _fg_edge_intr = cv2.bitwise_and(master_fg, cv2.bitwise_not(_fg_ero_intr))
                        intrusion_uint8 = cv2.bitwise_and(intrusion_uint8, cv2.bitwise_not(_fg_edge_intr))
                    diff = np.maximum(diff, intrusion_uint8)
                    print("✓ 背景の重なりチェックを適用（マスター背景に比較の前景/差分が入った領域を検出に含めました）")

        # 前景XORを追加（形状差分の強調）
        _xor_mask_for_type = None  # 性質別マスク用に保持
        if foreground_xor_add:
            from src.core.segmentation import get_foreground_mask
            if master_fg is None:
                master_fg = _dilate_mask(
                    get_foreground_mask(
                        imageA,
                        blur_ksize=5,
                        keep_largest_ratio=foreground_mask_keep_ratio,
                        preclose_kernel=foreground_mask_kernel,
                    )
                )
            if compare_fg is None:
                compare_fg = _dilate_mask(
                    get_foreground_mask(
                        imageB,
                        blur_ksize=5,
                        keep_largest_ratio=foreground_mask_keep_ratio,
                        preclose_kernel=foreground_mask_kernel,
                    )
                )
            if master_fg is not None and compare_fg is not None:
                xor_mask = cv2.bitwise_xor(master_fg, compare_fg)
                _xor_mask_for_type = xor_mask.copy()
                # fg_maskエッジ帯を除外して注入（サンプル境目の差分混入を防ぐ）
                if bbox_fg_edge_suppress_enabled:
                    _k_xor = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    _fg_ero_xor = cv2.erode(master_fg, _k_xor, iterations=max(1, bbox_fg_edge_suppress_width))
                    _fg_edge_xor = cv2.bitwise_and(master_fg, cv2.bitwise_not(_fg_ero_xor))
                    xor_mask = cv2.bitwise_and(xor_mask, cv2.bitwise_not(_fg_edge_xor))
                diff = np.maximum(diff, xor_mask)
                print("✓ 前景XORを追加（形状差分を検出に含めました）")

        # 前景境界の帯を除外 or 輪郭ベース内部保護（ズレ由来のリング差分対策）
        diff_before_edge = None
        fg_edge_band = None
        contour_interior_mask = None
        edge_band_for_thresh = None
        fg_edge_overlap = None

        if fg_contour_mask_enabled:
            # === 段階的 erode 分離方式 ===
            # 差分マスク自体を1pxずつ削って辺の差分と内部差分が分離したら
            # 分離座標より内側を全て内部と断定して元マスクから復元する
            diff_before_edge = diff.copy()

            erode_interior, erode_info = _erode_separate_interior(
                diff_gray=diff,
                diff_thresh=diff_thresh,
                max_erode=int(fg_contour_mask_erode_iter) if fg_contour_mask_erode_iter else 15,
                min_interior_area=max(10, int(min_area * 0.5)) if min_area else 50,
                connectivity=int(bbox_connectivity) if bbox_connectivity else 4,
            )

            if erode_interior is not None and np.count_nonzero(erode_interior) > 0:
                # 分離成功 → 内部領域のみの diff に置き換え
                diff = cv2.bitwise_and(diff, diff, mask=erode_interior)
            else:
                # 分離失敗 → フォールバック: fg_mask 収縮版で境界帯を除去
                if fg_contour_mask_fallback_to_fg and master_fg is not None:
                    contour_interior_mask = _make_contour_interior_mask(
                        src_image=imageA,
                        fg_mask=master_fg,
                        canny_low=fg_contour_mask_canny_low,
                        canny_high=fg_contour_mask_canny_high,
                        kernel_size=fg_contour_mask_kernel,
                        close_iter=fg_contour_mask_close_iter,
                        dilate_iter=fg_contour_mask_dilate_iter,
                        erode_iter=fg_contour_mask_erode_iter,
                        min_area_ratio=fg_contour_mask_min_area_ratio,
                        max_area_ratio=fg_contour_mask_max_area_ratio,
                        fallback_to_fg=fg_contour_mask_fallback_to_fg,
                    )
                    if contour_interior_mask is not None:
                        diff = cv2.bitwise_and(diff, diff, mask=contour_interior_mask)
                elif not fg_contour_mask_fallback_to_fg:
                    if master_fg is None:
                        from src.core.segmentation import get_foreground_mask
                        master_fg = _dilate_mask(
                            get_foreground_mask(
                                imageA,
                                blur_ksize=5,
                                keep_largest_ratio=foreground_mask_keep_ratio,
                                preclose_kernel=foreground_mask_kernel,
                            )
                        )

            if _save_stage is not None:
                _save_stage("09_erode_separate",
                            single=erode_interior if erode_interior is not None
                            else np.zeros_like(diff))
            if bbox_debug_log_path is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path),
                                             "09_erode_separate_info.txt")
                    before_nz = int(np.count_nonzero(diff_before_edge))
                    after_nz = int(np.count_nonzero(diff))
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"fg_contour_mask_enabled={fg_contour_mask_enabled}\n")
                        f.write(f"skipped={erode_info.get('skipped', False)}\n")
                        f.write(f"skip_reason={erode_info.get('skip_reason', None)}\n")
                        f.write(f"initial_edge_components={erode_info.get('initial_edge_components', 0)}\n")
                        f.write(f"initial_interior_components={erode_info.get('initial_interior_components', 0)}\n")
                        f.write(f"erode_separated={erode_info.get('separated', False)}\n")
                        f.write(f"erode_count={erode_info.get('erode_count', 0)}\n")
                        f.write(f"interior_components={erode_info.get('interior_components', 0)}\n")
                        f.write(f"interior_total_area={erode_info.get('interior_total_area', 0)}\n")
                        f.write(f"edge_sides={erode_info.get('edge_sides', [])}\n")
                        f.write(f"diff_nonzero_before={before_nz}\n")
                        f.write(f"diff_nonzero_after={after_nz}\n")
                        f.write(f"diff_removed={before_nz - after_nz}\n")
                        f.write(f"fallback_used={erode_interior is None}\n")
                except Exception:
                    pass

        elif fg_edge_exclude_enabled or fg_edge_raise_thresh_enabled or fg_edge_overlap_exclude_enabled:
            if master_fg is None:
                from src.core.segmentation import get_foreground_mask
                master_fg = _dilate_mask(
                    get_foreground_mask(
                        imageA,
                        blur_ksize=5,
                        keep_largest_ratio=foreground_mask_keep_ratio,
                        preclose_kernel=foreground_mask_kernel,
                    )
                    )
            if master_fg is not None:
                diff_before_edge = diff.copy()
                k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                it = max(1, int(fg_edge_exclude_iter))
                fg_dil = cv2.dilate(master_fg, k_edge, iterations=it)
                fg_ero = cv2.erode(master_fg, k_edge, iterations=it)
                fg_edge_band = cv2.bitwise_and(fg_dil, cv2.bitwise_not(fg_ero))
                edge_band_for_thresh = fg_edge_band
                if fg_edge_overlap_exclude_enabled:
                    if compare_fg is None:
                        from src.core.segmentation import get_foreground_mask
                        compare_fg = _dilate_mask(
                            get_foreground_mask(
                                imageB,
                                blur_ksize=5,
                                keep_largest_ratio=foreground_mask_keep_ratio,
                                preclose_kernel=foreground_mask_kernel,
                            )
                        )
                    if compare_fg is not None:
                        cmp_dil = cv2.dilate(compare_fg, k_edge, iterations=it)
                        cmp_ero = cv2.erode(compare_fg, k_edge, iterations=it)
                        cmp_edge_band = cv2.bitwise_and(cmp_dil, cv2.bitwise_not(cmp_ero))
                        fg_edge_overlap = cv2.bitwise_and(fg_edge_band, cmp_edge_band)
                        edge_band_for_thresh = cv2.bitwise_and(fg_edge_band, cv2.bitwise_not(fg_edge_overlap))
                        diff = cv2.bitwise_and(diff, cv2.bitwise_not(fg_edge_overlap))
                        if _save_stage is not None:
                            _save_stage("09_mask_fg_edge_overlap", single=fg_edge_overlap)
                if fg_edge_exclude_enabled:
                    diff = cv2.bitwise_and(diff, cv2.bitwise_not(fg_edge_band))
                if _save_stage is not None:
                    _save_stage("09_mask_fg_edge", single=fg_edge_band)
                if bbox_debug_log_path is not None:
                    try:
                        info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "09_fg_edge_exclude_info.txt")
                        before_nz = int(np.count_nonzero(diff_before_edge))
                        after_nz = int(np.count_nonzero(diff))
                        edge_nz = int(np.count_nonzero(fg_edge_band))
                        edge_overlap = int(np.count_nonzero((diff_before_edge > 0) & (fg_edge_band > 0)))
                        edge_ratio = edge_overlap / before_nz if before_nz else 0.0
                        with open(info_path, "w", encoding="utf-8") as f:
                            f.write(f"fg_edge_exclude_enabled={fg_edge_exclude_enabled}\n")
                            f.write(f"fg_edge_exclude_iter={it}\n")
                            f.write(f"fg_edge_overlap_exclude_enabled={fg_edge_overlap_exclude_enabled}\n")
                            f.write(f"fg_edge_raise_thresh_enabled={fg_edge_raise_thresh_enabled}\n")
                            f.write(f"fg_edge_raise_thresh_offset={int(fg_edge_raise_thresh_offset)}\n")
                            f.write(f"fg_edge_band_nonzero={edge_nz}\n")
                            f.write(f"diff_nonzero_before={before_nz}\n")
                            f.write(f"diff_nonzero_after={after_nz}\n")
                            f.write(f"edge_overlap={edge_overlap}\n")
                            f.write(f"edge_overlap_ratio={edge_ratio:.6f}\n")
                    except Exception:
                        pass
                if bbox_debug_log_path is not None:
                    try:
                        stats_path = os.path.join(os.path.dirname(bbox_debug_log_path), "09_fg_edge_value_stats.txt")
                        diff_stats = diff_before_edge
                        if diff_stats is not None and diff_stats.ndim == 3:
                            diff_stats = cv2.cvtColor(diff_stats, cv2.COLOR_BGR2GRAY)
                        if diff_stats is not None:
                            band_mask = edge_band_for_thresh if edge_band_for_thresh is not None else fg_edge_band
                            band_vals = diff_stats[band_mask > 0]
                            nonzero_vals = band_vals[band_vals > 0] if band_vals.size else band_vals
                            with open(stats_path, "w", encoding="utf-8") as f:
                                f.write(f"fg_edge_band_nonzero={int(np.count_nonzero(fg_edge_band))}\n")
                                if fg_edge_overlap is not None:
                                    f.write(f"fg_edge_overlap_nonzero={int(np.count_nonzero(fg_edge_overlap))}\n")
                                    f.write(f"fg_edge_band_for_thresh_nonzero={int(np.count_nonzero(band_mask))}\n")
                                f.write(f"band_pixels={int(band_vals.size)}\n")
                                f.write(f"band_nonzero={int(nonzero_vals.size)}\n")
                                if nonzero_vals.size:
                                    nz = nonzero_vals.astype(np.float32)
                                    f.write(f"band_nonzero_min={float(np.min(nz)):.3f}\n")
                                    f.write(f"band_nonzero_mean={float(np.mean(nz)):.3f}\n")
                                    f.write(f"band_nonzero_median={float(np.median(nz)):.3f}\n")
                                    f.write(f"band_nonzero_max={float(np.max(nz)):.3f}\n")
                                    for p in (50, 75, 90, 95, 99, 99.5):
                                        f.write(f"band_nonzero_p{p}={float(np.percentile(nz, p)):.3f}\n")
                                else:
                                    f.write("band_nonzero_min=none\n")
                                    f.write("band_nonzero_mean=none\n")
                                    f.write("band_nonzero_median=none\n")
                                    f.write("band_nonzero_max=none\n")
                    except Exception:
                        pass

        # 穴マスク（ランダムな穴を自動抽出して差分から除外）
        hole_mask = None
        hole_info = None
        hole_info_a = None
        hole_info_b = None
        hole_mask_a = None
        hole_mask_b = None
        if hole_mask_enabled:
            hole_source = (hole_mask_source or "").lower()

            def _get_hole_base(src_image, which):
                nonlocal master_fg, compare_fg
                base_mask = None
                if hole_source in ("edge_roi_or_foreground", "edge_roi"):
                    edge_mask, _ = _make_edge_roi_mask(
                        src_image,
                        bbox_edge_roi_canny_low,
                        bbox_edge_roi_canny_high,
                        bbox_edge_roi_kernel,
                        bbox_edge_roi_close_iter,
                        bbox_edge_roi_dilate_iter,
                        bbox_edge_roi_min_area_ratio,
                        bbox_edge_roi_max_area_ratio,
                        bbox_edge_roi_require_closed,
                        bbox_edge_roi_border_margin,
                    )
                    if edge_mask is not None:
                        base_mask = edge_mask
                if base_mask is None and hole_source in ("edge_roi_or_foreground", "foreground"):
                    from src.core.segmentation import get_foreground_mask
                    if which == "master":
                        if master_fg is None:
                            master_fg = _dilate_mask(
                                get_foreground_mask(
                                    imageA,
                                    blur_ksize=5,
                                    keep_largest_ratio=foreground_mask_keep_ratio,
                                    preclose_kernel=foreground_mask_kernel,
                                )
                            )
                        base_mask = master_fg
                    else:
                        if compare_fg is None:
                            compare_fg = _dilate_mask(
                                get_foreground_mask(
                                    imageB,
                                    blur_ksize=5,
                                    keep_largest_ratio=foreground_mask_keep_ratio,
                                    preclose_kernel=foreground_mask_kernel,
                                )
                            )
                        base_mask = compare_fg
                return base_mask

            apply_mode = (hole_mask_apply_mode or "master").lower()
            if apply_mode in ("both", "both_or", "both_and"):
                base_a = _get_hole_base(imageA, "master")
                base_b = _get_hole_base(imageB, "compare")
                if base_a is not None:
                    hole_mask_a, hole_info_a = _build_hole_mask(
                        imageA,
                        base_a,
                        hole_mask_method,
                        hole_min_area_ratio,
                        hole_max_area_ratio,
                        hole_min_fill_ratio,
                        hole_aspect_min,
                        hole_aspect_max,
                        hole_shrink_iter,
                        hole_max_total_ratio,
                        hole_bg_border_ratio,
                        hole_bg_dist_percentile,
                        hole_bg_dist_scale,
                        hole_bg_dist_min,
                        hole_bg_dist_max,
                        hole_inner_erode_iter,
                        hole_expand_iter,
                    )
                if base_b is not None:
                    hole_mask_b, hole_info_b = _build_hole_mask(
                        imageB,
                        base_b,
                        hole_mask_method,
                        hole_min_area_ratio,
                        hole_max_area_ratio,
                        hole_min_fill_ratio,
                        hole_aspect_min,
                        hole_aspect_max,
                        hole_shrink_iter,
                        hole_max_total_ratio,
                        hole_bg_border_ratio,
                        hole_bg_dist_percentile,
                        hole_bg_dist_scale,
                        hole_bg_dist_min,
                        hole_bg_dist_max,
                        hole_inner_erode_iter,
                        hole_expand_iter,
                    )
                if hole_mask_a is not None and hole_mask_b is not None:
                    if apply_mode == "both_and":
                        hole_mask = cv2.bitwise_and(hole_mask_a, hole_mask_b)
                    else:
                        hole_mask = cv2.bitwise_or(hole_mask_a, hole_mask_b)
                else:
                    hole_mask = hole_mask_a if hole_mask_a is not None else hole_mask_b
                hole_info = f"apply={apply_mode} a={hole_info_a} b={hole_info_b}"
                if _save_stage is not None:
                    if hole_mask_a is not None:
                        _save_stage("10_mask_holes_A", single=hole_mask_a)
                    if hole_mask_b is not None:
                        _save_stage("10_mask_holes_B", single=hole_mask_b)
            else:
                base = _get_hole_base(imageA, "master")
                if base is not None:
                    hole_mask, hole_info = _build_hole_mask(
                        imageA,
                        base,
                        hole_mask_method,
                        hole_min_area_ratio,
                        hole_max_area_ratio,
                        hole_min_fill_ratio,
                        hole_aspect_min,
                        hole_aspect_max,
                        hole_shrink_iter,
                        hole_max_total_ratio,
                        hole_bg_border_ratio,
                        hole_bg_dist_percentile,
                        hole_bg_dist_scale,
                        hole_bg_dist_min,
                        hole_bg_dist_max,
                        hole_inner_erode_iter,
                        hole_expand_iter,
                    )

        if hole_mask is not None:
            diff_before_hole = diff.copy()
            diff = cv2.bitwise_and(diff, cv2.bitwise_not(hole_mask))
            hole_edge_mask = None
            if hole_edge_exclude_enabled:
                try:
                    k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    hole_dil = cv2.dilate(hole_mask, k_edge, iterations=int(hole_edge_exclude_iter))
                    hole_edge_mask = cv2.bitwise_and(hole_dil, cv2.bitwise_not(hole_mask))
                    diff = cv2.bitwise_and(diff, cv2.bitwise_not(hole_edge_mask))
                    if _save_stage is not None:
                        _save_stage("10_mask_hole_edge", single=hole_edge_mask)
                except Exception:
                    hole_edge_mask = None
            if _save_stage is not None:
                _save_stage("10_mask_holes", single=hole_mask)
            if bbox_debug_log_path is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "10_mask_holes_info.txt")
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"hole_mask_enabled={hole_mask_enabled}\n")
                        f.write(f"hole_mask_source={hole_mask_source}\n")
                        f.write(f"hole_mask_apply_mode={hole_mask_apply_mode}\n")
                        f.write(f"hole_mask_method={hole_mask_method}\n")
                        f.write(f"hole_bg_border_ratio={hole_bg_border_ratio}\n")
                        f.write(f"hole_bg_dist_percentile={hole_bg_dist_percentile}\n")
                        f.write(f"hole_bg_dist_scale={hole_bg_dist_scale}\n")
                        f.write(f"hole_bg_dist_min={hole_bg_dist_min}\n")
                        f.write(f"hole_bg_dist_max={hole_bg_dist_max}\n")
                        f.write(f"hole_inner_erode_iter={hole_inner_erode_iter}\n")
                        f.write(f"hole_expand_iter={hole_expand_iter}\n")
                        f.write(f"hole_edge_exclude_enabled={hole_edge_exclude_enabled}\n")
                        f.write(f"hole_edge_exclude_iter={hole_edge_exclude_iter}\n")
                        if hole_info_a is not None or hole_info_b is not None:
                            f.write(f"hole_info_a={hole_info_a}\n")
                            f.write(f"hole_info_b={hole_info_b}\n")
                        f.write(f"hole_info={hole_info}\n")
                        try:
                            before_nz = int(np.count_nonzero(diff_before_hole))
                            after_nz = int(np.count_nonzero(diff))
                            overlap = int(np.count_nonzero((diff_before_hole > 0) & (hole_mask > 0)))
                            overlap_ratio = overlap / before_nz if before_nz else 0.0
                            f.write(f"diff_nonzero_before={before_nz}\n")
                            f.write(f"diff_nonzero_after={after_nz}\n")
                            f.write(f"hole_diff_overlap={overlap}\n")
                            f.write(f"hole_diff_overlap_ratio={overlap_ratio:.6f}\n")
                            if hole_edge_mask is not None:
                                edge_overlap = int(np.count_nonzero((diff_before_hole > 0) & (hole_edge_mask > 0)))
                                edge_overlap_ratio = edge_overlap / before_nz if before_nz else 0.0
                                f.write(f"hole_edge_overlap={edge_overlap}\n")
                                f.write(f"hole_edge_overlap_ratio={edge_overlap_ratio:.6f}\n")
                        except Exception:
                            pass
                except Exception:
                    pass
            if hole_mask is None and bbox_debug_log_path is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "10_mask_holes_info.txt")
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"hole_mask_enabled={hole_mask_enabled}\n")
                        f.write(f"hole_mask_source={hole_mask_source}\n")
                        f.write(f"hole_mask_apply_mode={hole_mask_apply_mode}\n")
                        f.write(f"hole_mask_method={hole_mask_method}\n")
                        f.write(f"hole_bg_border_ratio={hole_bg_border_ratio}\n")
                        f.write(f"hole_bg_dist_percentile={hole_bg_dist_percentile}\n")
                        f.write(f"hole_bg_dist_scale={hole_bg_dist_scale}\n")
                        f.write(f"hole_bg_dist_min={hole_bg_dist_min}\n")
                        f.write(f"hole_bg_dist_max={hole_bg_dist_max}\n")
                        f.write(f"hole_inner_erode_iter={hole_inner_erode_iter}\n")
                        f.write(f"hole_expand_iter={hole_expand_iter}\n")
                        f.write(f"hole_edge_exclude_enabled={hole_edge_exclude_enabled}\n")
                        f.write(f"hole_edge_exclude_iter={hole_edge_exclude_iter}\n")
                        if hole_info_a is not None or hole_info_b is not None:
                            f.write(f"hole_info_a={hole_info_a}\n")
                            f.write(f"hole_info_b={hole_info_b}\n")
                        f.write("hole_info=none\n")
                except Exception:
                    pass

        _save_stage("10_after_bg_overlap", single=diff)

        # ========== 前景マスク内部穴の境界差分除外 ==========
        # 前景マスクの内部にある大きな穴（中央の丸穴など）の境界付近の差分を除外する。
        # 表vs裏比較時に位置ずれで穴境界にリング状の差分が出るのを防ぐ。
        if fg_hole_edge_exclude_enabled and fg_mask is not None:
            try:
                not_fg = cv2.bitwise_not(fg_mask)
                h_fg, w_fg = fg_mask.shape[:2]
                fg_area = float(h_fg * w_fg)
                num_hole_labels, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
                    not_fg, connectivity=8
                )
                fg_hole_edge_mask = np.zeros_like(diff)
                fg_hole_count = 0
                for lbl in range(1, num_hole_labels):
                    hx, hy, hw, hh, harea = hole_stats[lbl]
                    area_ratio = harea / fg_area if fg_area > 0 else 0.0
                    if area_ratio < fg_hole_edge_exclude_min_area_ratio:
                        continue
                    # 画像端に接する領域は背景なので除外（内部穴のみ対象）
                    touches_border = (hx == 0) or (hy == 0) or (hx + hw >= w_fg) or (hy + hh >= h_fg)
                    if touches_border:
                        continue
                    # この穴を膨張させて境界リングを作成
                    single_hole = (hole_labels == lbl).astype(np.uint8) * 255
                    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    dilated = cv2.dilate(single_hole, k_dil, iterations=fg_hole_edge_exclude_dilate)
                    ring = cv2.bitwise_and(dilated, fg_mask)  # 前景内の穴近傍リング
                    fg_hole_edge_mask = cv2.bitwise_or(fg_hole_edge_mask, ring)
                    fg_hole_count += 1
                if fg_hole_count > 0 and np.count_nonzero(fg_hole_edge_mask) > 0:
                    diff_before_fg_hole = np.count_nonzero(diff)
                    diff = cv2.bitwise_and(diff, cv2.bitwise_not(fg_hole_edge_mask))
                    diff_after_fg_hole = np.count_nonzero(diff)
                    if _save_stage is not None:
                        _save_stage("10_fg_hole_edge_mask", single=fg_hole_edge_mask)
                    print(f"[FgHoleEdgeExclude] holes={fg_hole_count} "
                          f"dilate={fg_hole_edge_exclude_dilate} "
                          f"diff: {diff_before_fg_hole} -> {diff_after_fg_hole} "
                          f"removed={diff_before_fg_hole - diff_after_fg_hole}")
            except Exception as e:
                print(f"[FgHoleEdgeExclude] error: {e}")

        # ========== 性質別マスク（debug用） ==========
        # 最終 diff が確定したタイミングで、差分の性質を3分類してdebug保存する
        # A: 形状差分（前景XOR由来）= 輪郭の違い
        # B: 表面差分（前景内部のabsdiff）= 傷、汚れ、変色
        # C: アーティファクト（erode_separateで除去された領域）= 位置合わせ残差
        _bbox_type_coloring_legacy = kwargs.get("bbox_type_coloring", True)
        if _save_stage is not None and _bbox_type_coloring_legacy:
            try:
                _thresh_bin = diff_thresh if diff_thresh else 13
                # A: 形状マスク = XOR mask をthresholdして二値化
                if _xor_mask_for_type is not None:
                    _, type_shape = cv2.threshold(_xor_mask_for_type, 127, 255, cv2.THRESH_BINARY)
                    # 最終diffでも残っている部分のみ（erode等で除去された部分は除外）
                    _, diff_bin = cv2.threshold(diff, _thresh_bin, 255, cv2.THRESH_BINARY)
                    type_shape = cv2.bitwise_and(type_shape, diff_bin)
                else:
                    type_shape = np.zeros_like(diff)

                # B: 表面マスク = 最終diff のうち形状マスク以外の部分
                type_surface = cv2.bitwise_and(diff_bin if '_xor_mask_for_type' in dir() and _xor_mask_for_type is not None else cv2.threshold(diff, _thresh_bin, 255, cv2.THRESH_BINARY)[1],
                                                cv2.bitwise_not(type_shape))

                # C: アーティファクトマスク = erode_separateで除去された領域
                if diff_before_edge is not None:
                    _, before_bin = cv2.threshold(diff_before_edge, _thresh_bin, 255, cv2.THRESH_BINARY)
                    _, after_bin = cv2.threshold(diff, _thresh_bin, 255, cv2.THRESH_BINARY)
                    type_artifact = cv2.subtract(before_bin, after_bin)
                else:
                    type_artifact = np.zeros_like(diff)

                _save_stage("10b_type_shape", single=type_shape)
                _save_stage("10b_type_surface", single=type_surface)
                _save_stage("10b_type_artifact", single=type_artifact)
                # BBOX色分け用にインスタンスに保持
                self._type_shape_mask = type_shape
                self._type_surface_mask = type_surface

                # 情報ファイル
                shape_nz = int(np.count_nonzero(type_shape))
                surface_nz = int(np.count_nonzero(type_surface))
                artifact_nz = int(np.count_nonzero(type_artifact))
                total_nz = int(np.count_nonzero(diff_bin if '_xor_mask_for_type' in dir() and _xor_mask_for_type is not None else cv2.threshold(diff, _thresh_bin, 255, cv2.THRESH_BINARY)[1]))
                print(f"  [性質別マスク] 形状={shape_nz}px, 表面={surface_nz}px, アーティファクト={artifact_nz}px, 合計diff={total_nz}px")
                if bbox_debug_log_path is not None:
                    try:
                        info_path = os.path.join(os.path.dirname(bbox_debug_log_path),
                                                 "10b_type_classification.txt")
                        with open(info_path, "w", encoding="utf-8") as f:
                            f.write(f"shape_pixels={shape_nz}\n")
                            f.write(f"surface_pixels={surface_nz}\n")
                            f.write(f"artifact_pixels={artifact_nz}\n")
                            f.write(f"total_diff_pixels={total_nz}\n")
                            f.write(f"xor_available={'yes' if _xor_mask_for_type is not None else 'no'}\n")
                            f.write(f"erode_available={'yes' if diff_before_edge is not None else 'no'}\n")
                    except Exception:
                        pass
            except Exception as e:
                print(f"  [性質別マスク] 生成エラー: {e}")
        # ========== 性質別マスク ここまで ==========

        diff_for_bbox = diff

        # ========== fg_maskエッジ帯のdiff抑制（サンプル境目の差分除外） ==========
        _fg_for_suppress = fg_mask if fg_mask is not None else master_fg
        if bbox_fg_edge_suppress_enabled and _fg_for_suppress is not None:
            try:
                _sup_width = max(1, bbox_fg_edge_suppress_width)
                _sup_offset = max(1, bbox_fg_edge_suppress_offset)
                _k_sup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                _fg_eroded = cv2.erode(_fg_for_suppress, _k_sup, iterations=_sup_width)
                _fg_edge_sup = cv2.bitwise_and(_fg_for_suppress, cv2.bitwise_not(_fg_eroded))
                _sup_idx = _fg_edge_sup > 0
                if np.any(_sup_idx):
                    if diff_for_bbox is diff:
                        diff_for_bbox = diff.copy()
                    _before_nz = int(np.count_nonzero(diff_for_bbox))
                    if _sup_offset >= 255:
                        # 完全除外モード
                        diff_for_bbox[_sup_idx] = 0
                    else:
                        _lowered = diff_for_bbox[_sup_idx].astype(np.int16) - _sup_offset
                        diff_for_bbox[_sup_idx] = np.clip(_lowered, 0, 255).astype(np.uint8)
                    _after_nz = int(np.count_nonzero(diff_for_bbox))
                    print(f"[BboxFgEdgeSuppress] width={_sup_width} offset={_sup_offset} "
                          f"edge_band={int(np.count_nonzero(_fg_edge_sup))}px "
                          f"diff: {_before_nz} -> {_after_nz} (removed {_before_nz - _after_nz})")
                    if _save_stage is not None:
                        _save_stage("10c_bbox_fg_edge_suppress", single=_fg_edge_sup)
            except Exception as e:
                print(f"[BboxFgEdgeSuppress] error: {e}")

        if fg_edge_raise_thresh_enabled and edge_band_for_thresh is not None:
            try:
                offset = int(fg_edge_raise_thresh_offset)
            except Exception:
                offset = 0
            if offset > 0:
                diff_for_bbox = diff.copy()
                band_idx = edge_band_for_thresh > 0
                if np.any(band_idx):
                    lowered = diff_for_bbox[band_idx].astype(np.int16) - offset
                    diff_for_bbox[band_idx] = np.clip(lowered, 0, 255).astype(np.uint8)
                if bbox_debug_log_path is not None:
                    try:
                        info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "09_fg_edge_thresh_info.txt")
                        before_nz = int(np.count_nonzero(diff))
                        after_nz = int(np.count_nonzero(diff_for_bbox))
                        edge_nz = int(np.count_nonzero(fg_edge_band))
                        edge_overlap = int(np.count_nonzero((diff > 0) & (fg_edge_band > 0)))
                        edge_ratio = edge_overlap / before_nz if before_nz else 0.0
                        with open(info_path, "w", encoding="utf-8") as f:
                            f.write(f"fg_edge_raise_thresh_enabled={fg_edge_raise_thresh_enabled}\n")
                            f.write(f"fg_edge_raise_thresh_offset={offset}\n")
                            f.write(f"fg_edge_band_nonzero={edge_nz}\n")
                            f.write(f"diff_nonzero_before={before_nz}\n")
                            f.write(f"diff_nonzero_after={after_nz}\n")
                            f.write(f"edge_overlap={edge_overlap}\n")
                            f.write(f"edge_overlap_ratio={edge_ratio:.6f}\n")
                    except Exception:
                        pass

        # マスク→BBOX（塊だけ）※オプション
        strong_bboxes = []
        if detect_bbox:
            bbox_debug_dir = os.path.dirname(bbox_debug_log_path) if bbox_debug_log_path else None
            bbox_debug_image_prefix = os.path.join(bbox_debug_dir, "11_mask") if bbox_debug_dir else None
            bbox_debug_image_prefix_roi = os.path.join(bbox_debug_dir, "11_mask_roi") if bbox_debug_dir else None
            edge_roi_mask = None
            edge_roi_ratio = None
            edge_roi_used = False
            edge_roi_reason = None
            if bbox_edge_roi_enabled:
                edge_roi_mask, edge_roi_ratio = _make_edge_roi_mask(
                    imageA,
                    bbox_edge_roi_canny_low,
                    bbox_edge_roi_canny_high,
                    bbox_edge_roi_kernel,
                    bbox_edge_roi_close_iter,
                    bbox_edge_roi_dilate_iter,
                    bbox_edge_roi_min_area_ratio,
                    bbox_edge_roi_max_area_ratio,
                    bbox_edge_roi_require_closed,
                    bbox_edge_roi_border_margin,
                )
                # edge ROI が前景マスクの穴の内側を指している場合（= 穴のエッジを
                # ROI として検出してしまった場合）、前景マスクにフォールバックする
                if edge_roi_mask is not None and fg_mask is not None:
                    fg_nz = np.count_nonzero(fg_mask)
                    overlap = np.count_nonzero(cv2.bitwise_and(edge_roi_mask, fg_mask))
                    roi_nz = np.count_nonzero(edge_roi_mask)
                    if roi_nz > 0 and fg_nz > 0:
                        overlap_ratio = overlap / float(roi_nz)
                        # edge ROI の大半が前景の外（穴の中）にある場合
                        if overlap_ratio < 0.5:
                            edge_roi_mask = fg_mask.copy()
                            edge_roi_ratio = float(fg_nz) / float(fg_mask.size) if fg_mask.size else None
            if bbox_inner_mask is None and edge_roi_mask is not None and bbox_inner_overlap_ratio is not None:
                inner = edge_roi_mask
                if bbox_inner_erode_iter and bbox_inner_erode_iter > 0:
                    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    inner = cv2.erode(edge_roi_mask, k_inner, iterations=int(bbox_inner_erode_iter))
                if inner is not None and np.count_nonzero(inner) > 0:
                    bbox_inner_mask = inner
                    bbox_inner_mask_source = "edge_roi"
            if _save_stage is not None and bbox_inner_mask is not None:
                _save_stage("09_bbox_inner_mask", single=bbox_inner_mask)
            if bbox_debug_log_path is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "09_bbox_inner_info.txt")
                    fg_nz = int(np.count_nonzero(fg_mask)) if fg_mask is not None else 0
                    fg_total = int(fg_mask.size) if fg_mask is not None else 0
                    inner_nz = int(np.count_nonzero(bbox_inner_mask)) if bbox_inner_mask is not None else 0
                    total = int(bbox_inner_mask.size) if bbox_inner_mask is not None else 0
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"bbox_inner_erode_iter={bbox_inner_erode_iter}\n")
                        f.write(f"bbox_inner_overlap_ratio={bbox_inner_overlap_ratio}\n")
                        f.write(f"bbox_inner_source={bbox_inner_mask_source}\n")
                        f.write(f"fg_mask_nonzero={fg_nz}\n")
                        f.write(f"fg_mask_ratio={fg_nz / fg_total if fg_total else 0:.6f}\n")
                        f.write(f"bbox_inner_mask_nonzero={inner_nz}\n")
                        f.write(f"bbox_inner_mask_ratio={inner_nz / total if total else 0:.6f}\n")
                except Exception:
                    pass
            # まずROIなしでBBOXを取る（巻き込み回避のため）
            mask_full, bboxes_full = diff_to_bboxes(
                diff_gray=diff_for_bbox,
                thresh=diff_thresh,
                min_area=min_area,
                min_width=bbox_min_width,
                morph_kernel=morph_kernel,
                close_iter=bbox_close_iter,
                open_iter=bbox_open_iter,
                max_boxes=max_boxes,
                edge_ignore_ratio=bbox_edge_ignore_ratio,
                min_area_relax_ratio=bbox_min_area_relax_ratio,
                connectivity=bbox_connectivity,
                edge_min_fill_ratio=bbox_edge_min_fill_ratio,
                drop_band_fill_ratio=bbox_drop_band_fill_ratio,
                drop_band_width_ratio=bbox_drop_band_width_ratio,
                drop_band_height_ratio=bbox_drop_band_height_ratio,
                drop_band_aspect=bbox_drop_band_aspect,
                drop_band_max_fill=bbox_drop_band_max_fill,
                drop_band_aspect_high=bbox_drop_band_aspect_high,
                drop_band_max_fill_high=bbox_drop_band_max_fill_high,
                drop_sparse_large_cover=bbox_drop_sparse_large_cover,
                drop_sparse_large_max_fill=bbox_drop_sparse_large_max_fill,
                drop_band_min_area_ratio=bbox_drop_band_min_area_ratio,
                inner_mask=bbox_inner_mask,
                inner_overlap_ratio=bbox_inner_overlap_ratio,
                debug_path=bbox_debug_log_path,
                debug_image_prefix=bbox_debug_image_prefix,
            )
            mask, bboxes = mask_full, bboxes_full
            if edge_roi_mask is not None:
                if _save_stage is not None:
                    _save_stage("10_mask_edge_roi", single=edge_roi_mask)
                # 全体囲い/帯が出た場合のみROIで絞る
                need_edge_roi = False
                if bboxes_full:
                    h_img, w_img = diff.shape[:2]
                    image_area = float(h_img * w_img) if h_img and w_img else 0.0
                    for (x, y, bw, bh) in bboxes_full:
                        cover_ratio = (bw * bh) / image_area if image_area > 0 else 0.0
                        width_ratio = bw / float(w_img) if w_img else 0.0
                        height_ratio = bh / float(h_img) if h_img else 0.0
                        touches_edge = x == 0 or y == 0 or (x + bw) >= w_img or (y + bh) >= h_img
                        if bbox_edge_ignore_ratio is not None and cover_ratio >= float(bbox_edge_ignore_ratio):
                            need_edge_roi = True
                            edge_roi_reason = f"cover>={bbox_edge_ignore_ratio}"
                            break
                        if touches_edge and width_ratio >= 0.98 and height_ratio >= 0.12:
                            need_edge_roi = True
                            edge_roi_reason = "full_width_band"
                            break
                if need_edge_roi:
                    diff_roi = cv2.bitwise_and(diff_for_bbox, diff_for_bbox, mask=edge_roi_mask)
                    mask_roi, bboxes_roi = diff_to_bboxes(
                        diff_gray=diff_roi,
                        thresh=diff_thresh,
                        min_area=min_area,
                        min_width=bbox_min_width,
                        morph_kernel=morph_kernel,
                        close_iter=bbox_close_iter,
                        open_iter=bbox_open_iter,
                        max_boxes=max_boxes,
                        edge_ignore_ratio=bbox_edge_ignore_ratio,
                        min_area_relax_ratio=bbox_min_area_relax_ratio,
                        connectivity=bbox_connectivity,
                        edge_min_fill_ratio=bbox_edge_min_fill_ratio,
                        drop_band_fill_ratio=bbox_drop_band_fill_ratio,
                        drop_band_width_ratio=bbox_drop_band_width_ratio,
                        drop_band_height_ratio=bbox_drop_band_height_ratio,
                        drop_band_aspect=bbox_drop_band_aspect,
                        drop_band_max_fill=bbox_drop_band_max_fill,
                        drop_band_aspect_high=bbox_drop_band_aspect_high,
                        drop_band_max_fill_high=bbox_drop_band_max_fill_high,
                        drop_sparse_large_cover=bbox_drop_sparse_large_cover,
                        drop_sparse_large_max_fill=bbox_drop_sparse_large_max_fill,
                        drop_band_min_area_ratio=bbox_drop_band_min_area_ratio,
                        inner_mask=bbox_inner_mask,
                        inner_overlap_ratio=bbox_inner_overlap_ratio,
                        debug_path=None,
                        debug_image_prefix=bbox_debug_image_prefix_roi,
                    )
                    if bboxes_roi:
                        mask, bboxes = mask_roi, bboxes_roi
                        edge_roi_used = True
                    else:
                        # ROIでBBOXが消える場合はフル差分を維持
                        edge_roi_used = False
            # 外周と繋がった成分を内側マスクで救済（全体BBOXを出さずにナット等を拾う）
            if bbox_inner_mask is not None and bbox_edge_ignore_ratio is not None:
                try:
                    rescue_min_area = min_area
                    if bbox_rescue_min_area_ratio is not None and bbox_rescue_min_area_ratio > 0 and min_area > 0:
                        rescue_min_area = max(10, int(min_area * float(bbox_rescue_min_area_ratio)))
                    elif bbox_min_area_relax_ratio is not None and bbox_min_area_relax_ratio > 0 and min_area > 0:
                        rescue_min_area = max(10, int(min_area * float(bbox_min_area_relax_ratio)))
                    rescue_thresh = diff_thresh + int(bbox_rescue_thresh_offset or 0)
                    if rescue_thresh < 1:
                        rescue_thresh = 1
                    rescue_kernel = morph_kernel if bbox_rescue_morph_kernel is None else int(bbox_rescue_morph_kernel)
                    if rescue_kernel % 2 == 0:
                        rescue_kernel += 1
                    rescue_close = bbox_close_iter if bbox_rescue_close_iter is None else int(bbox_rescue_close_iter)
                    rescue_open = bbox_open_iter if bbox_rescue_open_iter is None else int(bbox_rescue_open_iter)

                    source_for_drop = diff_before_edge if diff_before_edge is not None else diff_for_bbox
                    _, seed = cv2.threshold(source_for_drop, rescue_thresh, 255, cv2.THRESH_BINARY)
                    k_drop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rescue_kernel, rescue_kernel))
                    if rescue_open > 0:
                        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k_drop, iterations=rescue_open)
                    if rescue_close > 0:
                        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k_drop, iterations=rescue_close)
                    edge_drop_mask = np.zeros_like(seed)
                    h_img, w_img = seed.shape[:2]
                    image_area = float(h_img * w_img) if h_img and w_img else 0.0
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=int(bbox_connectivity))
                    for label in range(1, num_labels):
                        x, y, bw, bh, area = stats[label]
                        touches_edge = x == 0 or y == 0 or (x + bw) >= w_img or (y + bh) >= h_img
                        if not touches_edge or image_area <= 0:
                            continue
                        cover_ratio = (bw * bh) / image_area
                        if cover_ratio < float(bbox_edge_ignore_ratio):
                            continue
                        fill_ratio = float(area) / float(bw * bh) if bw and bh else 0.0
                        if bbox_edge_min_fill_ratio is not None and fill_ratio >= float(bbox_edge_min_fill_ratio):
                            continue
                        edge_drop_mask[labels == label] = 255
                    if np.count_nonzero(edge_drop_mask) > 0:
                        edge_cut_mask = edge_drop_mask
                        edge_cut_applied = False
                        edge_cut_nonzero = None
                        if bbox_edge_cut_enabled:
                            try:
                                edge_cut_applied = True
                                k_size = int(bbox_edge_cut_kernel or 3)
                                if k_size % 2 == 0:
                                    k_size += 1
                                k_cut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                                cut = cv2.morphologyEx(edge_drop_mask, cv2.MORPH_OPEN, k_cut, iterations=int(bbox_edge_cut_iter or 1))
                                num_cut, labels_cut, stats_cut, _ = cv2.connectedComponentsWithStats(cut, connectivity=int(bbox_connectivity))
                                edge_free = np.zeros_like(cut)
                                for lbl in range(1, num_cut):
                                    x, y, bw, bh, area = stats_cut[lbl]
                                    touches = x == 0 or y == 0 or (x + bw) >= w_img or (y + bh) >= h_img
                                    if touches:
                                        continue
                                    aspect = max(bw, bh) / max(1, min(bw, bh))
                                    if aspect > 1.5:
                                        # 四分音符型（低fill）: 距離変換でコアのみ抽出
                                        comp_mask = np.zeros_like(cut)
                                        comp_mask[labels_cut == lbl] = 255
                                        _dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
                                        _dt = max(5, k_size)
                                        _, _core = cv2.threshold(_dist, _dt, 255, cv2.THRESH_BINARY)
                                        _core = _core.astype(np.uint8)
                                        if np.count_nonzero(_core) > 0:
                                            edge_free = cv2.bitwise_or(edge_free, _core)
                                        else:
                                            edge_free[labels_cut == lbl] = 255
                                    else:
                                        edge_free[labels_cut == lbl] = 255
                                if np.count_nonzero(edge_free) > 0:
                                    edge_cut_mask = edge_free
                                else:
                                    # edge_freeが空=MORPH_OPENで分離できなかった場合、
                                    # edge_cut_maskをリセットしてborder exclusion zoneの結果のみ使う
                                    edge_cut_mask = np.zeros_like(edge_drop_mask)
                                # MORPH_OPEN後もエッジに接する成分: 高fillならナット等の実体として保持
                                remaining_edge = cv2.bitwise_and(cut, cv2.bitwise_not(edge_free))
                                if np.count_nonzero(remaining_edge) > 0:
                                    num_re, labels_re, stats_re, _ = cv2.connectedComponentsWithStats(
                                        remaining_edge, connectivity=int(bbox_connectivity)
                                    )
                                    for lbl_re in range(1, num_re):
                                        x_re, y_re, bw_re, bh_re, a_re = stats_re[lbl_re]
                                        if a_re < rescue_min_area:
                                            continue
                                        fill_re = a_re / max(1, bw_re * bh_re)
                                        if fill_re >= 0.4:
                                            edge_cut_mask[labels_re == lbl_re] = 255
                                    if _save_stage is not None:
                                        _save_stage("11_mask_edge_cut_interior", single=remaining_edge)
                                edge_cut_nonzero = int(np.count_nonzero(edge_cut_mask))
                                if _save_stage is not None:
                                    _save_stage("11_mask_edge_cut", single=edge_cut_mask)
                            except Exception as _exc:
                                edge_cut_applied = False
                        # edge_cut で分離できなかった場合 → 距離ベース分離を試行
                        if edge_sep_enabled and fg_mask is not None and np.count_nonzero(edge_cut_mask) == 0:
                            try:
                                from src.core.edge_separation import separate_edge_from_interior
                                sep_interior, sep_info = separate_edge_from_interior(
                                    edge_drop_mask=edge_drop_mask,
                                    fg_mask=fg_mask,
                                    diff_gray=source_for_drop,
                                    min_dist=edge_sep_min_dist,
                                    max_search_dist=edge_sep_max_dist,
                                    search_step=edge_sep_step,
                                    min_interior_area=rescue_min_area,
                                    connectivity=int(bbox_connectivity),
                                )
                                if sep_interior is not None and np.count_nonzero(sep_interior) > 0:
                                    edge_cut_mask = sep_interior
                                if _save_stage is not None:
                                    _save_stage("11_mask_edge_sep_interior", single=sep_interior if sep_interior is not None else np.zeros_like(edge_drop_mask))
                            except Exception:
                                pass
                        # 全画面の疎成分が落ちた場合は、縁を落として内側を救済
                        source_diff = diff_before_edge if diff_before_edge is not None else diff_for_bbox
                        rescue_base = edge_cut_mask.copy()
                        rescue_edge_band = None
                        if bbox_rescue_edge_erode_iter and int(bbox_rescue_edge_erode_iter) > 0:
                            k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            eroded = cv2.erode(rescue_base, k_inner, iterations=int(bbox_rescue_edge_erode_iter))
                            rescue_edge_band = cv2.bitwise_and(rescue_base, cv2.bitwise_not(eroded))
                            rescue_base = eroded
                        if bbox_rescue_use_fg_edge_band and fg_edge_band is not None:
                            rescue_edge_band = rescue_edge_band if rescue_edge_band is not None else cv2.bitwise_and(rescue_base, fg_edge_band)
                            rescue_base = cv2.bitwise_and(rescue_base, cv2.bitwise_not(fg_edge_band))
                        if bbox_inner_mask is not None:
                            rescue_base = cv2.bitwise_and(rescue_base, bbox_inner_mask)
                        if np.count_nonzero(rescue_base) == 0 and bbox_inner_mask is not None:
                            # MORPH_OPEN後のcutを使う（元のedge_drop_maskより外周が除去されている）
                            _fallback = cut if edge_cut_applied else edge_drop_mask
                            rescue_base = cv2.bitwise_and(_fallback, bbox_inner_mask)
                        rescue_diff = cv2.bitwise_and(source_diff, source_diff, mask=rescue_base)
                        if _save_stage is not None:
                            _save_stage("11_mask_edge_rescue", single=rescue_base)
                            if rescue_edge_band is not None:
                                _save_stage("11_mask_edge_rescue_band", single=rescue_edge_band)
                        rescue_debug_path = None
                        rescue_debug_prefix = None
                        if bbox_debug_dir:
                            rescue_debug_path = os.path.join(bbox_debug_dir, "11_mask_rescue_components.txt")
                            rescue_debug_prefix = os.path.join(bbox_debug_dir, "11_mask_rescue")
                        # rescue_* は edge_drop の算出に合わせて上で計算済み
                        rescue_mask, rescue_bboxes = diff_to_bboxes(
                            diff_gray=rescue_diff,
                            thresh=rescue_thresh,
                            min_area=rescue_min_area,
                            min_width=bbox_min_width,
                            morph_kernel=rescue_kernel,
                            close_iter=rescue_close,
                            open_iter=rescue_open,
                            max_boxes=max_boxes,
                            edge_ignore_ratio=None,
                            min_area_relax_ratio=None,
                            connectivity=bbox_connectivity,
                            edge_min_fill_ratio=None,
                            drop_band_fill_ratio=bbox_rescue_drop_band_fill_ratio,
                            drop_band_width_ratio=bbox_rescue_drop_band_width_ratio,
                            drop_band_height_ratio=bbox_rescue_drop_band_height_ratio,
                            drop_band_aspect=bbox_drop_band_aspect,
                            drop_band_max_fill=bbox_drop_band_max_fill,
                            drop_band_aspect_high=bbox_drop_band_aspect_high,
                            drop_band_max_fill_high=bbox_drop_band_max_fill_high,
                            drop_sparse_large_cover=bbox_drop_sparse_large_cover,
                            drop_sparse_large_max_fill=bbox_drop_sparse_large_max_fill,
                            debug_path=rescue_debug_path,
                            debug_image_prefix=rescue_debug_prefix,
                        )
                        if _save_stage is not None:
                            _save_stage("11_mask_edge_rescue_diff", single=rescue_diff)
                            if rescue_mask is not None:
                                _save_stage("11_mask_edge_rescue_mask", single=rescue_mask)
                            if rescue_bboxes:
                                try:
                                    overlay = np.zeros((source_diff.shape[0], source_diff.shape[1], 3), dtype=np.uint8)
                                    for (x, y, w, h) in rescue_bboxes:
                                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                    _save_stage("11_mask_edge_rescue_boxes", single=overlay)
                                except Exception:
                                    pass
                        if rescue_mask is not None and np.count_nonzero(rescue_mask) > 0:
                            mask = cv2.bitwise_or(mask, rescue_mask)
                        if rescue_bboxes:
                            def _iou(a, b):
                                ax, ay, aw, ah = a
                                bx, by, bw, bh = b
                                ax2, ay2 = ax + aw, ay + ah
                                bx2, by2 = bx + bw, by + bh
                                ix1, iy1 = max(ax, bx), max(ay, by)
                                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                                inter = iw * ih
                                if inter <= 0:
                                    return 0.0
                                union = aw * ah + bw * bh - inter
                                return inter / union if union > 0 else 0.0
                            merged = list(bboxes)
                            for rb in rescue_bboxes:
                                if all(_iou(rb, kb) < 0.5 for kb in merged):
                                    merged.append(rb)
                            # 面積順で上限に丸める
                            merged.sort(key=lambda b: b[2] * b[3], reverse=True)
                            bboxes = merged[:max_boxes]
                        if bbox_debug_log_path is not None:
                            try:
                                info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_edge_rescue_info.txt")
                                with open(info_path, "w", encoding="utf-8") as f:
                                    f.write(f"edge_drop_nonzero={int(np.count_nonzero(edge_drop_mask))}\n")
                                    f.write(f"edge_cut_enabled={bbox_edge_cut_enabled}\n")
                                    f.write(f"edge_cut_kernel={bbox_edge_cut_kernel}\n")
                                    f.write(f"edge_cut_iter={bbox_edge_cut_iter}\n")
                                    if edge_cut_applied:
                                        f.write(f"edge_cut_nonzero={edge_cut_nonzero}\n")
                                    f.write(f"rescue_base_nonzero={int(np.count_nonzero(rescue_base))}\n")
                                    if rescue_edge_band is not None:
                                        f.write(f"rescue_edge_band_nonzero={int(np.count_nonzero(rescue_edge_band))}\n")
                                    if rescue_mask is not None:
                                        f.write(f"rescue_mask_nonzero={int(np.count_nonzero(rescue_mask))}\n")
                                    if rescue_diff is not None:
                                        f.write(f"rescue_diff_nonzero={int(np.count_nonzero(rescue_diff))}\n")
                                    f.write(f"rescue_edge_erode_iter={bbox_rescue_edge_erode_iter}\n")
                                    f.write(f"rescue_use_fg_edge_band={bbox_rescue_use_fg_edge_band}\n")
                                    f.write(f"edge_drop_thresh={rescue_thresh}\n")
                                    f.write(f"edge_drop_kernel={rescue_kernel}\n")
                                    f.write(f"edge_drop_close_iter={rescue_close}\n")
                                    f.write(f"edge_drop_open_iter={rescue_open}\n")
                                    f.write(f"rescue_min_area={rescue_min_area}\n")
                                    f.write(f"rescue_min_area_ratio={bbox_rescue_min_area_ratio}\n")
                                    f.write(f"rescue_thresh={rescue_thresh}\n")
                                    f.write(f"rescue_morph_kernel={rescue_kernel}\n")
                                    f.write(f"rescue_close_iter={rescue_close}\n")
                                    f.write(f"rescue_open_iter={rescue_open}\n")
                                    f.write(f"rescue_drop_band_fill_ratio={bbox_rescue_drop_band_fill_ratio}\n")
                                    f.write(f"rescue_drop_band_width_ratio={bbox_rescue_drop_band_width_ratio}\n")
                                    f.write(f"rescue_drop_band_height_ratio={bbox_rescue_drop_band_height_ratio}\n")
                                    f.write(f"rescue_source={'diff_before_edge' if diff_before_edge is not None else 'diff_for_bbox'}\n")
                                    f.write(f"rescue_bbox_count={len(rescue_bboxes) if rescue_bboxes else 0}\n")
                            except Exception:
                                pass
                except Exception:
                    pass
            if fg_mask is not None:
                mask = cv2.bitwise_and(mask, fg_mask)
            if bbox_debug_log_path is not None and edge_roi_mask is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_edge_roi_info.txt")
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"bbox_edge_roi_enabled={bbox_edge_roi_enabled}\n")
                        f.write(f"bbox_edge_roi_ratio={edge_roi_ratio if edge_roi_ratio is not None else 'none'}\n")
                        f.write(f"bbox_edge_roi_used={edge_roi_used}\n")
                        if edge_roi_reason:
                            f.write(f"bbox_edge_roi_reason={edge_roi_reason}\n")
                        if bboxes_full is not None:
                            f.write(f"bbox_count_full={len(bboxes_full)}\n")
                        if bboxes is not None:
                            f.write(f"bbox_count_final={len(bboxes)}\n")
                        if not edge_roi_used:
                            f.write("bbox_edge_roi_fallback=1\n")
                except Exception:
                    pass
            if hole_bbox_filter_enabled and hole_mask is not None and bboxes:
                try:
                    hole_filter = hole_mask.copy()
                    if hole_bbox_filter_dilate_iter and hole_bbox_filter_dilate_iter > 0:
                        kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        hole_filter = cv2.dilate(hole_filter, kf, iterations=int(hole_bbox_filter_dilate_iter))
                    h_img, w_img = hole_filter.shape[:2]
                    kept = []
                    dropped = []
                    for (x, y, w, h) in bboxes:
                        if w <= 0 or h <= 0:
                            dropped.append((x, y, w, h, "invalid_size"))
                            continue
                        x2 = min(x + w, w_img)
                        y2 = min(y + h, h_img)
                        x1 = max(0, x)
                        y1 = max(0, y)
                        if x2 <= x1 or y2 <= y1:
                            dropped.append((x, y, w, h, "out_of_bounds"))
                            continue
                        roi = hole_filter[y1:y2, x1:x2]
                        overlap = int(np.count_nonzero(roi))
                        area = int(w * h)
                        overlap_ratio = overlap / float(area) if area > 0 else 0.0
                        area_ratio = area / float(w_img * h_img) if w_img and h_img else 0.0
                        center_hit = False
                        if hole_bbox_filter_use_center:
                            cx = int(x + w // 2)
                            cy = int(y + h // 2)
                            if 0 <= cx < w_img and 0 <= cy < h_img:
                                center_hit = hole_filter[cy, cx] > 0
                        drop = False
                        if area_ratio <= float(hole_bbox_filter_max_area_ratio) and overlap_ratio >= float(hole_bbox_filter_overlap_ratio):
                            drop = True
                        if hole_bbox_filter_use_center and center_hit and area_ratio <= float(hole_bbox_filter_max_area_ratio):
                            drop = True
                        if drop:
                            dropped.append((x, y, w, h, f"overlap={overlap_ratio:.2f} area_ratio={area_ratio:.3f} center={int(center_hit)}"))
                        else:
                            kept.append((x, y, w, h))
                    bboxes = kept
                    if bbox_debug_log_path is not None:
                        try:
                            info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_hole_filter.txt")
                            with open(info_path, "w", encoding="utf-8") as f:
                                f.write(f"hole_bbox_filter_enabled={hole_bbox_filter_enabled}\n")
                                f.write(f"hole_bbox_filter_dilate_iter={hole_bbox_filter_dilate_iter}\n")
                                f.write(f"hole_bbox_filter_overlap_ratio={hole_bbox_filter_overlap_ratio}\n")
                                f.write(f"hole_bbox_filter_max_area_ratio={hole_bbox_filter_max_area_ratio}\n")
                                f.write(f"hole_bbox_filter_use_center={hole_bbox_filter_use_center}\n")
                                f.write(f"dropped_count={len(dropped)}\n")
                                for x, y, w, h, reason in dropped:
                                    f.write(f"drop bbox=({x},{y},{w},{h}) {reason}\n")
                        except Exception:
                            pass
                except Exception:
                    pass
        else:
            # BBOX検出をスキップ
            _, mask = cv2.threshold(diff_for_bbox, diff_thresh, 255, cv2.THRESH_BINARY)
            if fg_mask is not None:
                mask = cv2.bitwise_and(mask, fg_mask)
            bboxes = []

        # 強差分BBOXを追加表示（ON時のみ）
        if strong_bbox_enabled and detect_bbox:
            strong_thresh = diff_thresh + int(strong_bbox_offset)
            nonzero = diff_for_bbox[diff_for_bbox > 0]
            if nonzero.size:
                try:
                    perc = float(np.percentile(nonzero, strong_bbox_percentile))
                    strong_thresh = max(strong_thresh, int(perc))
                except Exception:
                    pass
            if strong_thresh >= 255:
                # cv2.threshold は「> thresh」なので 255 だと常に空になる
                strong_thresh = 254
            def _write_components(mask, path, title):
                if path is None or mask is None:
                    return
                try:
                    if mask.ndim == 3:
                        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_gray = mask
                    _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=int(bbox_connectivity))
                    comps = []
                    for label in range(1, num_labels):
                        x, y, bw, bh, area = stats[label]
                        fill = float(area) / float(bw * bh) if bw > 0 and bh > 0 else 0.0
                        comps.append((area, x, y, bw, bh, fill))
                    comps.sort(reverse=True)
                    lines = [f"{title} components: {len(comps)}"]
                    h, w = mask_bin.shape[:2]
                    for area, x, y, bw, bh, fill in comps:
                        width_ratio = bw / float(w) if w else 0.0
                        height_ratio = bh / float(h) if h else 0.0
                        lines.append(
                            f"  area={area} bbox=({x},{y},{bw},{bh}) width={width_ratio:.2f} height={height_ratio:.2f} fill={fill:.2f}"
                        )
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                except Exception:
                    pass
            # エッジ外周が取れるときだけ内側に絞る
            edge_roi_mask = None
            edge_roi_ratio = None
            if strong_bbox_edge_roi_enabled:
                edge_roi_mask, edge_roi_ratio = _make_edge_roi_mask(
                    imageA,
                    strong_bbox_edge_roi_canny_low,
                    strong_bbox_edge_roi_canny_high,
                    strong_bbox_edge_roi_kernel,
                    strong_bbox_edge_roi_close_iter,
                    strong_bbox_edge_roi_dilate_iter,
                    strong_bbox_edge_roi_min_area_ratio,
                    strong_bbox_edge_roi_max_area_ratio,
                    strong_bbox_edge_roi_require_closed,
                    strong_bbox_edge_roi_border_margin,
                )
            # strong BBOXはfg_mask/erode/suppress前の元diffを使用
            # （各フィルタで端の耳差分が消されすぎる。strong_threshとmin_diff_maxで十分フィルタされる）
            diff_for_strong_base = diff_original.copy() if diff_original is not None else diff.copy()
            if fg_mask is not None:
                diff_for_strong_base = cv2.bitwise_and(diff_for_strong_base, diff_for_strong_base, mask=fg_mask)
            if edge_roi_mask is not None:
                if _save_stage is not None:
                    _save_stage("11_mask_strong_roi", single=edge_roi_mask)
                diff_for_strong = cv2.bitwise_and(diff_for_strong_base, diff_for_strong_base, mask=edge_roi_mask)
            else:
                diff_for_strong = diff_for_strong_base
            # 強差分の生マスクを保存（デバッグ用）
            if _save_stage is not None:
                _, strong_raw = cv2.threshold(diff_for_strong, strong_thresh, 255, cv2.THRESH_BINARY)
                if fg_mask is not None:
                    strong_raw = cv2.bitwise_and(strong_raw, fg_mask)
                _save_stage("11_mask_strong_raw", single=strong_raw)
                if bbox_debug_log_path is not None:
                    raw_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_raw_components.txt")
                    _write_components(strong_raw, raw_path, "strong_raw")
            strong_min_area = int(strong_bbox_min_area) if strong_bbox_min_area else 0
            if strong_min_area <= 0:
                strong_min_area = max(10, int(min_area * float(strong_bbox_min_area_ratio)))
            strong_kernel = int(strong_bbox_morph_kernel)
            if strong_kernel % 2 == 0:
                strong_kernel += 1
            if strong_bbox_use_edge_filter:
                edge_ratio = bbox_edge_ignore_ratio
                edge_fill = bbox_edge_min_fill_ratio
            else:
                edge_ratio = None
                edge_fill = None
            strong_mask, strong_bboxes = diff_to_bboxes(
                diff_gray=diff_for_strong,
                thresh=strong_thresh,
                min_area=strong_min_area,
                min_width=bbox_min_width,
                morph_kernel=max(3, strong_kernel),
                close_iter=strong_bbox_close_iter,
                open_iter=strong_bbox_open_iter,
                max_boxes=strong_bbox_max_boxes,
                edge_ignore_ratio=edge_ratio,
                min_area_relax_ratio=None,
                connectivity=bbox_connectivity,
                edge_min_fill_ratio=edge_fill,
                drop_band_fill_ratio=strong_bbox_drop_band_fill_ratio,
                drop_band_width_ratio=strong_bbox_drop_band_width_ratio,
                drop_band_height_ratio=strong_bbox_drop_band_height_ratio,
                drop_band_aspect=bbox_drop_band_aspect,
                drop_band_max_fill=bbox_drop_band_max_fill,
                drop_band_aspect_high=bbox_drop_band_aspect_high,
                drop_band_max_fill_high=bbox_drop_band_max_fill_high,
                drop_sparse_large_cover=bbox_drop_sparse_large_cover,
                drop_sparse_large_max_fill=bbox_drop_sparse_large_max_fill,
                drop_band_min_area_ratio=bbox_drop_band_min_area_ratio,
                inner_mask=bbox_inner_mask,
                inner_overlap_ratio=bbox_inner_overlap_ratio,
                debug_path=strong_bbox_debug_log_path,
            )
            # diff最大値フィルタ（位置ずれノイズ排除）
            if strong_bbox_min_diff_max and strong_bbox_min_diff_max > 0 and strong_bboxes:
                _min_dm = int(strong_bbox_min_diff_max)
                filtered_strong = []
                for (bx, by, bw, bh) in strong_bboxes:
                    roi = diff_for_strong[by:by+bh, bx:bx+bw]
                    roi_max = int(roi.max()) if roi.size > 0 else 0
                    if roi_max >= _min_dm:
                        filtered_strong.append((bx, by, bw, bh))
                    else:
                        if bbox_debug_log_path is not None:
                            try:
                                log_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_components.txt")
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(f"  drop_min_diff_max bbox=({bx},{by},{bw},{bh}) max_diff={roi_max} < {_min_dm}\n")
                            except Exception:
                                pass
                strong_bboxes = filtered_strong
            # 画像端マージンフィルタ（端ノイズ排除）
            if strong_bbox_edge_margin and strong_bbox_edge_margin > 0 and strong_bboxes:
                _margin = int(strong_bbox_edge_margin)
                h_img, w_img = diff_for_strong.shape[:2]
                filtered_edge = []
                for (bx, by, bw, bh) in strong_bboxes:
                    min_dist = min(bx, by, w_img - (bx + bw), h_img - (by + bh))
                    if min_dist >= _margin:
                        filtered_edge.append((bx, by, bw, bh))
                    else:
                        if bbox_debug_log_path is not None:
                            try:
                                log_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_components.txt")
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(f"  drop_edge_margin bbox=({bx},{by},{bw},{bh}) min_dist={min_dist} < {_margin}\n")
                            except Exception:
                                pass
                strong_bboxes = filtered_edge
            # fill率フィルタ（スカスカBBOX排除）
            if strong_bbox_min_fill and strong_bbox_min_fill > 0 and strong_bboxes:
                _min_fill = float(strong_bbox_min_fill)
                filtered_fill = []
                for (bx, by, bw, bh) in strong_bboxes:
                    bbox_area = bw * bh
                    if bbox_area > 0:
                        roi = diff_for_strong[by:by+bh, bx:bx+bw]
                        roi_nonzero = int(np.count_nonzero(roi > strong_thresh)) if roi.size > 0 else 0
                        fill = roi_nonzero / float(bbox_area)
                    else:
                        fill = 0.0
                    if fill >= _min_fill:
                        filtered_fill.append((bx, by, bw, bh))
                    else:
                        if bbox_debug_log_path is not None:
                            try:
                                log_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_components.txt")
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(f"  drop_min_fill bbox=({bx},{by},{bw},{bh}) fill={fill:.4f} < {_min_fill}\n")
                            except Exception:
                                pass
                strong_bboxes = filtered_fill
            if fg_mask is not None:
                strong_mask = cv2.bitwise_and(strong_mask, fg_mask)
            _save_stage("11_mask_strong", single=strong_mask)
            if bbox_debug_log_path is not None:
                strong_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_components_summary.txt")
                _write_components(strong_mask, strong_path, "strong_mask")
            if bbox_debug_log_path is not None:
                try:
                    info_path = os.path.join(os.path.dirname(bbox_debug_log_path), "11_mask_strong_info.txt")
                    nz = int(np.count_nonzero(strong_mask)) if strong_mask is not None else 0
                    total = int(strong_mask.size) if strong_mask is not None else 0
                    with open(info_path, "w", encoding="utf-8") as f:
                        f.write(f"strong_thresh={strong_thresh}\n")
                        f.write(f"strong_percentile={strong_bbox_percentile}\n")
                        f.write(f"strong_offset={strong_bbox_offset}\n")
                        f.write(f"strong_min_area={strong_min_area}\n")
                        f.write(f"strong_kernel={strong_kernel}\n")
                        f.write(f"strong_close_iter={strong_bbox_close_iter}\n")
                        f.write(f"strong_open_iter={strong_bbox_open_iter}\n")
                        f.write(f"strong_max_boxes={strong_bbox_max_boxes}\n")
                        f.write(f"strong_use_edge_filter={strong_bbox_use_edge_filter}\n")
                        f.write(f"strong_edge_roi_enabled={strong_bbox_edge_roi_enabled}\n")
                        f.write(f"strong_edge_roi_ratio={edge_roi_ratio if edge_roi_ratio is not None else 'none'}\n")
                        f.write(f"strong_mask_nonzero={nz}\n")
                        f.write(f"strong_mask_ratio={nz / total if total else 0:.6f}\n")
                except Exception:
                    pass

        # 性質別マスクを返り値に含める（BBOX色分け用）
        _type_masks = {
            "shape": getattr(self, '_type_shape_mask', None),
            "surface": getattr(self, '_type_surface_mask', None),
        }
        return diff, mask, bboxes, fg_mask, strong_bboxes, _type_masks

    def _generate_result_figure(self, visA, visB, mask, diff, ssim_map, bboxes, title, ssim_score, mse_value, max_boxes, show_plot):
        """結果の可視化図を生成"""
        # ヒートマップとSSIMも残す
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(visA, cv2.COLOR_BGR2RGB))
        plt.title("基準画像 + BBOX", fontweight="bold")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(visB, cv2.COLOR_BGR2RGB))
        plt.title("比較画像 + BBOX", fontweight="bold")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap="gray")
        plt.title("Diff Mask（塊化後）", fontweight="bold")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        im = plt.imshow(diff, cmap="jet")
        plt.title("差分ヒートマップ（絶対差分）", fontweight="bold")
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.subplot(2, 3, 5)
        # 【修正】SSIMマップを「どこがダメか」の異常検知表示にする（1.0=一致→0.0=青、似てない→赤）
        ssim_defect_view = 1.0 - np.nan_to_num(np.asarray(ssim_map, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        ssim_defect_view = np.clip(ssim_defect_view, 0.0, 1.0)
        im2 = plt.imshow(ssim_defect_view, cmap="jet", vmin=0.0, vmax=0.5)
        plt.title("SSIM 異常検知マップ", fontweight="bold")
        plt.axis("off")
        plt.colorbar(im2, fraction=0.046, pad=0.04)

        plt.subplot(2, 3, 6)
        plt.axis("off")
        bbox_text = "\n".join(
            [f"{i+1}. x={x}, y={y}, w={w}, h={h}" for i, (x, y, w, h) in enumerate(bboxes)]
        )
        plt.text(0, 1, bbox_text if bbox_text else "検出なし", va="top", fontsize=10)
        plt.title("BBOX一覧", fontweight="bold")
        plt.suptitle(
            f"{title}\nBBOX={len(bboxes)} | SSIM={ssim_score:.4f} | MSE={mse_value:.2f}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.subplots_adjust(top=0.9, bottom=0.06, wspace=0.2, hspace=0.2)
        if show_plot:
            plt.show()

        return fig


def compare_images(imageA, imageB, **params):
    """
    既存のcompare_images関数と互換性のあるラッパー

    Args:
        imageA: 基準画像
        imageB: 比較画像
        **params: compare_imagesの全パラメータ

    Returns:
        タプル形式: (mse, ssim, diff, mask, bboxes, fig, align_info, crop_info, roi_info, flip_info, quality_info, structural_info, scratch_info, contour_info, pin_info)
        または辞書形式（return_dict=Trueの場合）
    """
    # 不要なパラメータを除外（テスト用の設定に含まれている可能性がある）
    params_filtered = {k: v for k, v in params.items() if k not in [
        'auto_diff_thresh', 'crop_to_master_fov', 'flip_compare', 'return_dict'
    ]}

    # crop_to_master_fov と use_crop_to_master_fov のマッピング
    if 'crop_to_master_fov' in params:
        params_filtered['use_crop_to_master_fov'] = params['crop_to_master_fov']

    # flip_compare と use_flip_compare のマッピング
    if 'flip_compare' in params:
        params_filtered['use_flip_compare'] = params['flip_compare']

    pipeline = SymmetryPipeline(params_filtered)
    result_tuple = pipeline.run(imageA, imageB, **params_filtered)
    strong_bboxes = getattr(pipeline, "_last_strong_bboxes", None)

    # 辞書形式で返す場合（テスト用）
    if params.get('return_dict', False):
        mse_value, ssim_score, diff, mask, bboxes, fig, align_info, crop_info, roi_info, flip_info = result_tuple[:10]
        quality_info = result_tuple[10] if len(result_tuple) > 10 else None
        structural_info = result_tuple[11] if len(result_tuple) > 11 else None
        scratch_info = result_tuple[12] if len(result_tuple) > 12 else None
        contour_info = result_tuple[13] if len(result_tuple) > 13 else None
        pin_info = result_tuple[14] if len(result_tuple) > 14 else None
        aligned = bool(align_info and align_info.get("success"))
        return {
            'mse': mse_value,
            'ssim': ssim_score,
            'diff': diff,
            'mask': mask,
            'bboxes': bboxes,
            'strong_bboxes': strong_bboxes if strong_bboxes is not None else [],
            'fig': fig,
            'figure': fig,
            'align_info': align_info,
            'align_diagnostics': align_info,
            'aligned': aligned,
            'crop_info': crop_info,
            'roi_info': roi_info,
            'flip_info': flip_info,
            'quality_info': quality_info,
            'structural_info': structural_info,
            'scratch_info': scratch_info,
            'contour_info': contour_info,
            'pin_info': pin_info,
        }

    return result_tuple
