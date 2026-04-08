"""中央パネル: ライブビュー + 解析結果タブ"""
from __future__ import annotations

import io

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QScrollArea, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget,
)

from ui.widgets.image_label import ImageLabel
from ui.widgets.collapsible import CollapsibleSection


def _make_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("color: #444;")
    return sep


def _badge(text: str, color: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(
        f"background: {color}; color: white; font-weight: bold; "
        f"border-radius: 4px; padding: 6px 12px; font-size: 14px;"
    )
    lbl.setWordWrap(True)
    return lbl


class CenterPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- ライブビュー / 解析結果図 ---
        self._live_label = QLabel("ライブビュー")
        self._live_label.setStyleSheet("font-weight: bold; color: #aaa;")
        self._view = ImageLabel("カメラ未接続 / 画像未選択")
        self._view.setMinimumHeight(240)

        layout.addWidget(self._live_label)
        layout.addWidget(self._view, stretch=2)
        layout.addWidget(_make_separator())

        # --- 結果タブ ---
        self._tabs = QTabWidget()
        self._tabs.setMinimumHeight(200)

        self._tab_score = _ScoreTab()
        self._tab_align = _TextTab("位置合わせ情報なし")
        self._tab_crop = _CropTab()
        self._tab_detail = _TextTab("詳細情報なし")
        self._tab_calib = _TextTab("キャリブレーション情報なし")

        self._tabs.addTab(self._tab_score, "スコア・判定")
        self._tabs.addTab(self._tab_align, "位置合わせ")
        self._tabs.addTab(self._tab_crop, "画角切り出し")
        self._tabs.addTab(self._tab_detail, "構造・傷・品質")
        self._tabs.addTab(self._tab_calib, "キャリブレーション")

        layout.addWidget(self._tabs, stretch=1)

    # ── ライブビュー ──────────────────────────────

    def set_live_frame(self, img: np.ndarray | None) -> None:
        self._live_label.setText("ライブビュー")
        self._view.set_cv2(img)

    def set_result_figure(self, fig_bytes: bytes | None) -> None:
        """解析結果の Matplotlib PNG bytes を表示"""
        self._live_label.setText("解析結果")
        if fig_bytes is None:
            self._view.clear_image()
            return
        pixmap = QPixmap()
        pixmap.loadFromData(fig_bytes)
        self._view.set_pixmap_source(pixmap)

    def clear_view(self) -> None:
        self._live_label.setText("ライブビュー")
        self._view.clear_image()

    # ── 結果タブ更新 ──────────────────────────────

    def show_analysis_error(self, msg: str) -> None:
        self._tab_score.show_error(msg)

    def update_result(self, result: tuple) -> None:
        (mse, ssim_score, diff, mask, bboxes, fig_bytes,
         align_info, crop_info, roi_info,
         flip_info, quality_info, structural_info, scratch_info,
         diff_thresh, min_area, morph_kernel, max_boxes) = result

        bbox_count = len(bboxes) if bboxes else 0
        self._tab_score.update(mse, ssim_score, bbox_count, max_boxes, bboxes)
        self._tab_align.set_text(_format_align(align_info, flip_info))
        self._tab_crop.update(crop_info)
        self._tab_detail.set_text(_format_detail(structural_info, scratch_info, quality_info, roi_info))

    def update_calibration(self, result: dict | None) -> None:
        if result is None:
            self._tab_calib.set_text("キャリブレーション情報なし")
        else:
            self._tab_calib.set_text(_format_calibration(result))


# ── サブタブ ──────────────────────────────────────────────────────────────

class _ScoreTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._verdict = QLabel("")
        self._verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict.setWordWrap(True)
        self._verdict.setMinimumHeight(40)

        self._grid_w = QWidget()
        grid = QGridLayout(self._grid_w)
        grid.setContentsMargins(0, 0, 0, 0)

        def _row_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #aaa;")
            return lbl

        self._lbl_mse = QLabel("—")
        self._lbl_ssim = QLabel("—")
        self._lbl_bbox = QLabel("—")

        grid.addWidget(_row_label("MSE"), 0, 0)
        grid.addWidget(self._lbl_mse, 0, 1)
        grid.addWidget(_row_label("SSIM"), 1, 0)
        grid.addWidget(self._lbl_ssim, 1, 1)
        grid.addWidget(_row_label("BBOX数"), 2, 0)
        grid.addWidget(self._lbl_bbox, 2, 1)

        layout.addWidget(self._verdict)
        layout.addWidget(self._grid_w)
        layout.addStretch()

    def update(self, mse: float, ssim: float, bbox_count: int, max_boxes: int, bboxes) -> None:
        is_ok = bbox_count == 0
        if is_ok:
            self._verdict.setText("✅  判定: OK（差異なし）")
            self._verdict.setStyleSheet(
                "background: #1a4a1a; color: #6f6; font-weight: bold; "
                "border-radius: 6px; padding: 8px; font-size: 14px;"
            )
        else:
            self._verdict.setText(f"❌  判定: NG（差異 {bbox_count} 箇所検出）")
            self._verdict.setStyleSheet(
                "background: #4a1a1a; color: #f66; font-weight: bold; "
                "border-radius: 6px; padding: 8px; font-size: 14px;"
            )
        self._lbl_mse.setText(f"{mse:.2f}（0に近いほど類似）")
        self._lbl_ssim.setText(f"{ssim:.4f}（1に近いほど類似）")
        self._lbl_bbox.setText(f"{bbox_count}（上限 {max_boxes}）")

    def show_error(self, msg: str) -> None:
        self._verdict.setText(f"⚠ エラー: {msg}")
        self._verdict.setStyleSheet(
            "background: #4a3a1a; color: #fa0; font-weight: bold; "
            "border-radius: 6px; padding: 8px;"
        )
        self._lbl_mse.setText("—")
        self._lbl_ssim.setText("—")
        self._lbl_bbox.setText("—")


class _TextTab(QScrollArea):
    """スクロール可能なテキスト表示タブ"""

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self.setWidgetResizable(True)
        self._lbl = QLabel(placeholder)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet("color: #ccc; padding: 8px;")
        self._lbl.setTextFormat(Qt.TextFormat.PlainText)
        self.setWidget(self._lbl)

    def set_text(self, text: str) -> None:
        self._lbl.setText(text)


class _CropTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._info_lbl = QLabel("画角切り出し情報なし")
        self._info_lbl.setStyleSheet("color: #ccc;")
        self._info_lbl.setWordWrap(True)
        layout.addWidget(self._info_lbl)

        # 比較プレビュー用タブ（切り出しなし vs ベスト）
        self._preview_tabs = QTabWidget()
        self._orig_view = ImageLabel("切り出しなし")
        self._best_view = ImageLabel("ベスト切り出し")
        self._preview_tabs.addTab(self._orig_view, "切り出しなし")
        self._preview_tabs.addTab(self._best_view, "ベスト")
        self._preview_tabs.setVisible(False)

        layout.addWidget(self._preview_tabs)
        layout.addStretch()

    def update(self, crop_info: dict | None) -> None:
        if not crop_info or not crop_info.get("enabled"):
            self._info_lbl.setText("画角切り出し: 無効")
            self._preview_tabs.setVisible(False)
            return

        lines = []
        adopted = crop_info.get("adopted", False)
        lines.append(f"採用: {'✅ あり' if adopted else '❌ なし'}")
        if crop_info.get("compare_mode"):
            lines.append("比較モード: ON")
        if adopted:
            lines.append(f"スケール: {crop_info.get('adopted_scale', '?')}")
            lines.append(f"マッチ数(元): {crop_info.get('matches_orig', '?')} → (切り出し): {crop_info.get('matches_crop', '?')}")
            ssim_o = crop_info.get("ssim_orig")
            ssim_c = crop_info.get("ssim_crop")
            if ssim_o is not None and ssim_c is not None:
                lines.append(f"SSIM: {ssim_o:.4f} → {ssim_c:.4f}")
        else:
            reason = crop_info.get("reason", "")
            if reason:
                lines.append(f"理由: {reason}")
        self._info_lbl.setText("\n".join(lines))
        self._preview_tabs.setVisible(True)


# ── フォーマット関数 ──────────────────────────────────────────────────────

def _format_align(align_info: dict | None, flip_info: dict | None) -> str:
    lines = []
    if flip_info and flip_info.get("enabled"):
        lines.append("=== 左右反転比較 ===")
        lines.append(f"  元SSIM: {flip_info.get('ssim_original', 0):.4f}")
        lines.append(f"  反転SSIM: {flip_info.get('ssim_flipped', 0):.4f}")
        lines.append(f"  採用: {'反転' if flip_info.get('used_flip') else '元のまま'}")
        lines.append("")

    if not align_info or not align_info.get("enabled"):
        lines.append("自動位置合わせ: 無効")
        return "\n".join(lines)

    decision = align_info.get("decision", "unknown")
    lines.append(f"=== 自動位置合わせ ===")
    lines.append(f"  決定: {decision}")
    lines.append(f"  成功: {'✅' if align_info.get('success') else '❌'}")
    lines.append(f"  回転: {align_info.get('rotation_deg', 0):.2f}°")
    lines.append(f"  平行移動: {align_info.get('translation_px', 0):.1f}px")
    lines.append(f"  マッチ数(補正前): {align_info.get('matches_before', 0)}")
    if align_info.get("decision_step"):
        lines.append(f"  判定ステップ: {align_info['decision_step']}")
    return "\n".join(lines)


def _format_detail(
    structural_info: dict | None,
    scratch_info: dict | None,
    quality_info: dict | None,
    roi_info: dict | None,
) -> str:
    lines = []

    if structural_info and structural_info.get("enabled"):
        lines.append("=== 構造比較 ===")
        cls = structural_info.get("classification", "unknown")
        msg = structural_info.get("message", "")
        lines.append(f"  分類: {cls}")
        if msg:
            lines.append(f"  {msg}")
        lines.append("")

    if scratch_info and scratch_info.get("enabled"):
        lines.append("=== 傷検出 ===")
        count = scratch_info.get("count", 0)
        lines.append(f"  検出数: {count}")
        if count > 0:
            lines.append(f"  最大長: {scratch_info.get('max_length', 0):.1f}px")
        lines.append("")

    if quality_info and quality_info.get("enabled"):
        lines.append("=== 画質ゲーティング ===")
        passed = quality_info.get("passed", True)
        lines.append(f"  結果: {'✅ OK' if passed else '❌ NG'}")
        reasons = quality_info.get("reasons", [])
        if reasons:
            for r in reasons:
                lines.append(f"    - {r}")
        m = quality_info.get("master", {})
        t = quality_info.get("test", {})
        if m or t:
            lines.append("  [マスター]")
            for k, v in m.items():
                lines.append(f"    {k}: {v}")
            lines.append("  [比較]")
            for k, v in t.items():
                lines.append(f"    {k}: {v}")
        lines.append("")

    if roi_info and roi_info.get("enabled"):
        lines.append("=== ROI ===")
        lines.append(
            f"  x={roi_info.get('x',0)}, y={roi_info.get('y',0)}, "
            f"w={roi_info.get('w',0)}, h={roi_info.get('h',0)}"
        )

    return "\n".join(lines) if lines else "詳細情報なし"


def _format_calibration(result: dict) -> str:
    lines = ["=== マスター自動キャリブレーション ==="]
    if result.get("calibrated"):
        lines.append(f"  ensemble_thresh: {result.get('ensemble_thresh')}")
        lines.append(f"  ensemble_bbox_min_mean: {result.get('ensemble_bbox_min_mean')}")
    else:
        lines.append("  キャリブレーション未実施（無効化またはスキップ）")
    stats = result.get("noise_stats", {})
    if stats:
        lines.append("  ノイズ統計:")
        for k, v in stats.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.2f}")
            else:
                lines.append(f"    {k}: {v}")
    elapsed = result.get("elapsed_sec")
    if elapsed is not None:
        lines.append(f"  処理時間: {elapsed:.2f}s")
    return "\n".join(lines)
