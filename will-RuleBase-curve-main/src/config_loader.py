from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import yaml

def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing key: {key}")
    return d[key]

@dataclass(frozen=True)
class CoarseCropConfig:
    enabled: bool
    pad_ratio: float
    top_exclude_ratio: float
    out_size: Tuple[int, int]

@dataclass(frozen=True)
class ScaleNormalizeConfig:
    enabled: bool
    method: str
    min_scale: float
    max_scale: float
    pad_value: int

@dataclass(frozen=True)
class ECCConfig:
    enabled: bool
    motion: str
    scale: float
    num_iter: int
    eps: float

@dataclass(frozen=True)
class AlignmentConfig:
    coarse_crop: CoarseCropConfig
    scale_normalize: ScaleNormalizeConfig
    ecc: ECCConfig

@dataclass(frozen=True)
class PartMaskConfig:
    enabled: bool
    method: str
    erode_ksize: int
    erode_iter: int

@dataclass(frozen=True)
class ReferenceMaskCutoutConfig:
    enabled: bool

@dataclass(frozen=True)
class ContourDiffConfig:
    enabled: bool
    method: str
    band_width: int
    miss_ratio_th: float
    extra_ratio_th: float
    xor_threshold_pixels: int
    xor_threshold_ratio: float

@dataclass(frozen=True)
class DetectionConfig:
    part_mask: PartMaskConfig
    reference_mask_cutout: ReferenceMaskCutoutConfig
    contour_diff: ContourDiffConfig

@dataclass(frozen=True)
class OverlayRedHSVConfig:
    h_ranges: List[List[int]]
    s_min: int
    v_min: int
    morph_close_ksize: int
    morph_close_iter: int

@dataclass(frozen=True)
class TemplateConfig:
    from_overlay: bool
    overlay_red_hsv: OverlayRedHSVConfig

@dataclass(frozen=True)
class VisualizationConfig:
    save_intermediate: bool
    overlay_alpha: float

@dataclass(frozen=True)
class IOConfig:
    outputs_dir: str
    templates_dir: str

@dataclass(frozen=True)
class AppConfig:
    io: IOConfig
    alignment: AlignmentConfig
    detection: DetectionConfig
    template: TemplateConfig
    visualization: VisualizationConfig

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    io = _req(raw, "io")
    alignment = _req(raw, "alignment")
    detection = _req(raw, "detection")
    template = _req(raw, "template")
    vis = _req(raw, "visualization")

    cc = _req(alignment, "coarse_crop")
    sn = _req(alignment, "scale_normalize")
    ecc = _req(alignment, "ecc")

    pm = _req(detection, "part_mask")
    rmc = _req(detection, "reference_mask_cutout")
    cd = _req(detection, "contour_diff")

    hsv = _req(template, "overlay_red_hsv")

    cfg = AppConfig(
        io=IOConfig(
            outputs_dir=str(_req(io, "outputs_dir")),
            templates_dir=str(_req(io, "templates_dir")),
        ),
        alignment=AlignmentConfig(
            coarse_crop=CoarseCropConfig(
                enabled=bool(_req(cc, "enabled")),
                pad_ratio=float(_req(cc, "pad_ratio")),
                top_exclude_ratio=float(_req(cc, "top_exclude_ratio")),
                out_size=(int(_req(cc, "out_size")[0]), int(_req(cc, "out_size")[1])),
            ),
            scale_normalize=ScaleNormalizeConfig(
                enabled=bool(_req(sn, "enabled")),
                method=str(_req(sn, "method")).lower(),
                min_scale=float(_req(sn, "min_scale")),
                max_scale=float(_req(sn, "max_scale")),
                pad_value=int(_req(sn, "pad_value")),
            ),
            ecc=ECCConfig(
                enabled=bool(_req(ecc, "enabled")),
                motion=str(_req(ecc, "motion")).lower(),
                scale=float(_req(ecc, "scale")),
                num_iter=int(_req(ecc, "num_iter")),
                eps=float(_req(ecc, "eps")),
            )
        ),
        detection=DetectionConfig(
            part_mask=PartMaskConfig(
                enabled=bool(_req(pm, "enabled")),
                method=str(_req(pm, "method")).lower(),
                erode_ksize=int(_req(pm, "erode_ksize")),
                erode_iter=int(_req(pm, "erode_iter")),
            ),
            reference_mask_cutout=ReferenceMaskCutoutConfig(
                enabled=bool(_req(rmc, "enabled")),
            ),
            contour_diff=ContourDiffConfig(
                enabled=bool(_req(cd, "enabled")),
                method=str(_req(cd, "method")).lower(),
                band_width=int(_req(cd, "band_width")),
                miss_ratio_th=float(_req(cd, "miss_ratio_th")),
                extra_ratio_th=float(_req(cd, "extra_ratio_th")),
                xor_threshold_pixels=int(_req(cd, "xor_threshold_pixels")),
                xor_threshold_ratio=float(_req(cd, "xor_threshold_ratio")),
            )
        ),
        template=TemplateConfig(
            from_overlay=bool(_req(template, "from_overlay")),
            overlay_red_hsv=OverlayRedHSVConfig(
                h_ranges=[list(x) for x in _req(hsv, "h_ranges")],
                s_min=int(_req(hsv, "s_min")),
                v_min=int(_req(hsv, "v_min")),
                morph_close_ksize=int(_req(hsv, "morph_close_ksize")),
                morph_close_iter=int(_req(hsv, "morph_close_iter")),
            )
        ),
        visualization=VisualizationConfig(
            save_intermediate=bool(_req(vis, "save_intermediate")),
            overlay_alpha=float(_req(vis, "overlay_alpha")),
        )
    )

    if cfg.alignment.ecc.motion not in ("translation", "euclidean", "affine", "homography"):
        raise ValueError("alignment.ecc.motion must be translation|euclidean|affine|homography")
    if cfg.alignment.scale_normalize.method not in ("area", "bbox_height"):
        raise ValueError("alignment.scale_normalize.method must be area|bbox_height")
    if cfg.detection.part_mask.erode_ksize % 2 == 0:
        raise ValueError("detection.part_mask.erode_ksize must be odd")
    return cfg
