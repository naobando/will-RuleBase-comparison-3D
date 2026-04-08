"""
コア機能モジュール
"""
from .preprocessing import preprocess_image
from .metrics import calculate_mse
from .quality_check import check_image_quality
from .bbox_detection import diff_to_bboxes, merge_nearby_bboxes
from .alignment import (
    auto_align_images,
    align_edge_based,
    align_bbox_based,
    align_ecc_refine,
    align_fg_icp,
    estimate_transform_magnitude,
    _rotate_image_by_angle,
    _get_main_rect_from_edges,
    _get_main_rect_from_bbox,
    _align_by_rect,
)
from .auto_thresh import calculate_auto_diff_thresh
from .segmentation import get_foreground_mask
from .crop import crop_to_master_fov, template_match_crop
from .edge_separation import separate_edge_from_interior
from .scratch import detect_scratches
from .diff_classifier import classify_diff, masks_to_bboxes
from .calibration import calibrate_master
from .pin_profile import (
    extract_pin_profile,
    compare_pin_profiles,
    visualize_pin_compare,
    image_to_binary_mask,
    PinProfile,
    PinCompareResult,
)

__all__ = [
    "preprocess_image",
    "calculate_mse",
    "check_image_quality",
    "diff_to_bboxes",
    "merge_nearby_bboxes",
    "auto_align_images",
    "align_edge_based",
    "align_bbox_based",
    "align_ecc_refine",
    "align_fg_icp",
    "estimate_transform_magnitude",
    "_rotate_image_by_angle",
    "_get_main_rect_from_edges",
    "_get_main_rect_from_bbox",
    "_align_by_rect",
    "calculate_auto_diff_thresh",
    "get_foreground_mask",
    "crop_to_master_fov",
    "template_match_crop",
    "separate_edge_from_interior",
    "detect_scratches",
    "classify_diff",
    "masks_to_bboxes",
    "calibrate_master",
    "extract_pin_profile",
    "compare_pin_profiles",
    "visualize_pin_compare",
    "image_to_binary_mask",
    "PinProfile",
    "PinCompareResult",
]
