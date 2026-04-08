"""左パネル: 操作・画像入力・検出パラメータ (Debugモード全パラメータ露出)"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from src.utils.image_utils import load_image_from_bytes, safe_imread


def _make_h_separator():
    from PySide6.QtWidgets import QFrame
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("color: #444;")
    return sep


def _group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox { font-weight: bold; color: #bbb; margin-top: 4px; } "
                      "QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(6, 12, 6, 6)
    layout.setSpacing(4)
    return box, layout


def _linked_slider(
    min_val: int, max_val: int, default: int, step: int = 1
) -> tuple[QHBoxLayout, QSlider, QSpinBox]:
    """スライダー + スピンボックス連動ペア"""
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(min_val, max_val)
    slider.setSingleStep(step)
    slider.setPageStep(step)
    slider.setValue(default)

    spinbox = QSpinBox()
    spinbox.setRange(min_val, max_val)
    spinbox.setSingleStep(step)
    spinbox.setValue(default)
    spinbox.setFixedWidth(70)

    slider.valueChanged.connect(spinbox.setValue)
    spinbox.valueChanged.connect(slider.setValue)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(slider, stretch=1)
    row.addWidget(spinbox)
    return row, slider, spinbox


class LeftPanel(QScrollArea):
    """操作・パラメータの左サイドバー"""

    # シグナル
    reset_requested = Signal()
    camera_connect_requested = Signal(int, int, int)   # device, width, height
    camera_disconnect_requested = Signal()
    capture_requested = Signal(int, int, int)          # device, width, height
    analysis_requested = Signal(dict)                  # params dict
    master_image_changed = Signal(object, tuple)       # (np.ndarray, key)
    test_image_changed = Signal(object, tuple)         # (np.ndarray, key)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(260)
        self.setMaximumWidth(360)

        container = QWidget()
        self.setWidget(container)
        root = QVBoxLayout(container)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        self._build_operation(root, config)
        self._build_image_input(root, config)
        self._build_camera_settings(root, config)
        self._build_detect_settings(root, config)
        self._build_sliders(root, config)
        self._build_pin_profile(root, config)
        root.addStretch()

    # ── 操作ボタン ────────────────────────────────────────────────────────

    def _build_operation(self, root: QVBoxLayout, config: dict) -> None:
        box, lay = _group("操作")

        self._btn_reset = QPushButton("リセット")
        self._btn_reset.setStyleSheet("background: #555;")
        self._btn_reset.clicked.connect(self.reset_requested)

        self._cb_camera_enabled = QCheckBox("カメラ接続を有効化")
        self._cb_camera_enabled.setChecked(True)
        self._cb_quality_check = QCheckBox("品質チェック")
        self._cb_quality_check.setChecked(config["camera"].get("quality_check", True))

        self._btn_connect = QPushButton("カメラ接続")
        self._btn_connect.clicked.connect(self._on_connect)
        self._btn_disconnect = QPushButton("カメラ接続解除")
        self._btn_disconnect.clicked.connect(self.camera_disconnect_requested)
        self._btn_capture = QPushButton("比較画像を撮影")
        self._btn_capture.clicked.connect(self._on_capture)

        self._btn_analyze = QPushButton("解析実行")
        self._btn_analyze.setStyleSheet(
            "background: #2a6a2a; color: white; font-weight: bold; padding: 6px;"
        )
        self._btn_analyze.clicked.connect(self._on_analyze)

        lay.addWidget(self._btn_reset)
        lay.addWidget(self._cb_camera_enabled)
        lay.addWidget(self._cb_quality_check)
        lay.addWidget(self._btn_connect)
        lay.addWidget(self._btn_disconnect)
        lay.addWidget(self._btn_capture)
        lay.addWidget(self._btn_analyze)

        root.addWidget(box)

    # ── 画像入力 ──────────────────────────────────────────────────────────

    def _build_image_input(self, root: QVBoxLayout, config: dict) -> None:
        box, lay = _group("画像入力 (clip/ / ファイル)")

        # clip/ クイック選択
        self._clip_master_cb = QComboBox()
        self._clip_test_cb = QComboBox()
        self._populate_clip(config)

        self._clip_master_cb.currentTextChanged.connect(self._on_clip_master_changed)
        self._clip_test_cb.currentTextChanged.connect(self._on_clip_test_changed)

        lay.addWidget(QLabel("基準 (clip/)"))
        lay.addWidget(self._clip_master_cb)
        lay.addWidget(QLabel("比較 (clip/)"))
        lay.addWidget(self._clip_test_cb)

        lay.addWidget(_make_h_separator())

        # ファイル選択
        self._btn_master_file = QPushButton("基準画像をファイルから選択...")
        self._btn_test_file = QPushButton("比較画像をファイルから選択...")
        self._lbl_master_file = QLabel("未選択")
        self._lbl_test_file = QLabel("未選択")
        for lbl in (self._lbl_master_file, self._lbl_test_file):
            lbl.setStyleSheet("color: #888; font-size: 10px;")
            lbl.setWordWrap(True)

        self._btn_master_file.clicked.connect(self._on_master_file)
        self._btn_test_file.clicked.connect(self._on_test_file)

        lay.addWidget(self._btn_master_file)
        lay.addWidget(self._lbl_master_file)
        lay.addWidget(self._btn_test_file)
        lay.addWidget(self._lbl_test_file)

        # 品質警告ラベル
        self._lbl_master_warn = QLabel("")
        self._lbl_test_warn = QLabel("")
        for lbl in (self._lbl_master_warn, self._lbl_test_warn):
            lbl.setStyleSheet("color: #fa0; font-size: 10px;")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
        lay.addWidget(self._lbl_master_warn)
        lay.addWidget(self._lbl_test_warn)

        root.addWidget(box)

    def _populate_clip(self, config: dict) -> None:
        self._clip_master_cb.addItem("-- 選択してください --")
        self._clip_test_cb.addItem("-- 選択してください --")

        # マスター候補
        master_dir = config["guide"].get("master_dir", "master")
        if os.path.isdir(master_dir):
            for f in sorted(os.listdir(master_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self._clip_master_cb.addItem(f, userData=os.path.join(master_dir, f))

        # clip/ ディレクトリから比較候補
        clip_dir = "clip"
        if os.path.isdir(clip_dir):
            for name in sorted(os.listdir(clip_dir)):
                path = os.path.join(clip_dir, name)
                if os.path.isfile(path) and name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self._clip_test_cb.addItem(name, userData=path)
                elif os.path.isdir(path):
                    for sub in sorted(os.listdir(path)):
                        if sub.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            sub_path = os.path.join(path, sub)
                            self._clip_test_cb.addItem(f"{name}/{sub}", userData=sub_path)

    # ── カメラ設定 ────────────────────────────────────────────────────────

    def _build_camera_settings(self, root: QVBoxLayout, config: dict) -> None:
        box, lay = _group("カメラ設定")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self._sb_device = QSpinBox()
        self._sb_device.setRange(0, 10)
        self._sb_device.setValue(config["camera"].get("device_index", 0))

        self._sb_width = QSpinBox()
        self._sb_width.setRange(0, 4096)
        self._sb_width.setSingleStep(16)
        self._sb_width.setValue(config["camera"].get("capture_width", 0))

        self._sb_height = QSpinBox()
        self._sb_height.setRange(0, 4096)
        self._sb_height.setSingleStep(16)
        self._sb_height.setValue(config["camera"].get("capture_height", 0))

        form.addRow("デバイス番号", self._sb_device)
        form.addRow("取得幅(px)", self._sb_width)
        form.addRow("取得高さ(px)", self._sb_height)
        lay.addLayout(form)
        root.addWidget(box)

    # ── 検出設定 ──────────────────────────────────────────────────────────

    def _build_detect_settings(self, root: QVBoxLayout, config: dict) -> None:
        acfg = config["analysis"]
        box, lay = _group("検出設定")

        # 前処理モード
        self._cb_preprocess = QComboBox()
        _modes = ["gray_blur", "luminance", "normalize", "edge", "blackhat", "contrast"]
        _labels = {
            "gray_blur": "グレースケール＋ブラー",
            "luminance": "輝度＋ブラー（標準）",
            "normalize": "明るさ正規化",
            "edge": "エッジ（Canny等）",
            "blackhat": "BlackHat",
            "contrast": "コントラスト強調",
        }
        for m in _modes:
            self._cb_preprocess.addItem(_labels[m], userData=m)
        cur = acfg.get("preprocess_mode", "luminance")
        self._cb_preprocess.setCurrentIndex(_modes.index(cur) if cur in _modes else 1)

        # 位置合わせ
        self._chk_auto_align = QCheckBox("自動位置合わせ")
        self._chk_auto_align.setChecked(acfg.get("auto_align", True))
        self._cb_align_mode = QComboBox()
        for m, lbl in [("feature_points", "特徴点"), ("edge", "エッジ一致"), ("bbox", "外接矩形")]:
            self._cb_align_mode.addItem(lbl, userData=m)
        cur_align = acfg.get("align_mode", "edge")
        _align_modes = ["feature_points", "edge", "bbox"]
        self._cb_align_mode.setCurrentIndex(_align_modes.index(cur_align) if cur_align in _align_modes else 1)
        self._chk_auto_align.stateChanged.connect(
            lambda s: self._cb_align_mode.setEnabled(bool(s))
        )
        self._cb_align_mode.setEnabled(acfg.get("auto_align", True))

        # 画角
        self._chk_crop = QCheckBox("画角をマスターに合わせる")
        self._chk_crop.setChecked(acfg.get("crop_to_master_fov", True))
        self._chk_crop_compare = QCheckBox("画角切り出し比較モード")
        self._chk_crop_compare.setChecked(acfg.get("crop_compare_mode", True))
        self._chk_crop.stateChanged.connect(
            lambda s: self._chk_crop_compare.setEnabled(bool(s))
        )
        self._chk_crop_compare.setEnabled(acfg.get("crop_to_master_fov", True))

        # BBOX
        self._chk_detect_bbox = QCheckBox("BBOX検出")
        self._chk_detect_bbox.setChecked(acfg.get("detect_bbox", True))
        self._chk_bbox_coloring = QCheckBox("BBOX色分類")
        self._chk_bbox_coloring.setChecked(acfg.get("bbox_type_coloring", True))
        self._chk_detect_bbox.stateChanged.connect(
            lambda s: self._chk_bbox_coloring.setEnabled(bool(s))
        )
        self._chk_bbox_coloring.setEnabled(acfg.get("detect_bbox", True))

        # 前景・その他
        self._chk_fg_mask = QCheckBox("前景マスク")
        self._chk_fg_mask.setChecked(acfg.get("foreground_mask", True))
        self._chk_bg_overlap = QCheckBox("背景の重なりチェック")
        self._chk_bg_overlap.setChecked(acfg.get("background_overlap_check", False))
        self._chk_fg_xor = QCheckBox("前景XOR追加")
        self._chk_fg_xor.setChecked(acfg.get("foreground_xor_add", False))
        self._chk_flip = QCheckBox("左右反転比較")
        self._chk_flip.setChecked(acfg.get("flip_compare", False))

        # ROI
        self._chk_roi = QCheckBox("ROIで検査領域を限定")
        self._chk_roi.setChecked(acfg.get("roi_enabled", False))
        roi_form = QFormLayout()
        roi_form.setContentsMargins(16, 0, 0, 0)
        self._sb_roi_x = QSpinBox(); self._sb_roi_x.setRange(0, 99999)
        self._sb_roi_y = QSpinBox(); self._sb_roi_y.setRange(0, 99999)
        self._sb_roi_w = QSpinBox(); self._sb_roi_w.setRange(0, 99999); self._sb_roi_w.setValue(640)
        self._sb_roi_h = QSpinBox(); self._sb_roi_h.setRange(0, 99999); self._sb_roi_h.setValue(480)
        for sb in (self._sb_roi_x, self._sb_roi_y, self._sb_roi_w, self._sb_roi_h):
            sb.setEnabled(acfg.get("roi_enabled", False))
        roi_form.addRow("x", self._sb_roi_x)
        roi_form.addRow("y", self._sb_roi_y)
        roi_form.addRow("幅", self._sb_roi_w)
        roi_form.addRow("高さ", self._sb_roi_h)
        self._chk_roi.stateChanged.connect(lambda s: [
            sb.setEnabled(bool(s))
            for sb in (self._sb_roi_x, self._sb_roi_y, self._sb_roi_w, self._sb_roi_h)
        ])

        lay.addWidget(QLabel("前処理モード"))
        lay.addWidget(self._cb_preprocess)
        lay.addWidget(self._chk_auto_align)
        lay.addWidget(self._cb_align_mode)
        lay.addWidget(self._chk_crop)
        lay.addWidget(self._chk_crop_compare)

        # 切り出し方式
        self._cb_crop_method = QComboBox()
        self._cb_crop_method.addItem("従来方式 (minAreaRect)", userData="conventional")
        self._cb_crop_method.addItem("ピンアート型取り", userData="pin_art")
        cur_crop_method = acfg.get("crop_method", "conventional")
        self._cb_crop_method.setCurrentIndex(
            1 if cur_crop_method == "pin_art" else 0
        )
        lay.addWidget(QLabel("切り出し方式"))
        lay.addWidget(self._cb_crop_method)

        lay.addWidget(self._chk_detect_bbox)
        lay.addWidget(self._chk_bbox_coloring)
        lay.addWidget(self._chk_fg_mask)
        lay.addWidget(self._chk_bg_overlap)
        lay.addWidget(self._chk_fg_xor)
        lay.addWidget(self._chk_flip)
        lay.addWidget(self._chk_roi)
        lay.addLayout(roi_form)

        root.addWidget(box)

    # ── スライダー ────────────────────────────────────────────────────────

    def _build_sliders(self, root: QVBoxLayout, config: dict) -> None:
        acfg = config["analysis"]
        box, lay = _group("スライダーパラメータ")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        row_dt, self._sl_diff_thresh, self._sb_diff_thresh = _linked_slider(
            1, 50, acfg.get("diff_thresh", 38))
        row_ma, self._sl_min_area, self._sb_min_area = _linked_slider(
            10, 5000, acfg.get("min_area", 850))
        row_mk, self._sl_morph_kernel, self._sb_morph_kernel = _linked_slider(
            3, 51, acfg.get("morph_kernel", 9), step=2)
        row_mb, self._sl_max_boxes, self._sb_max_boxes = _linked_slider(
            1, 10, acfg.get("max_boxes", 10))
        row_et, self._sl_ensemble_thresh, self._sb_ensemble_thresh = _linked_slider(
            10, 255, acfg.get("ensemble_thresh", 70))

        self._lbl_dt_auto = QLabel("")
        self._lbl_ma_auto = QLabel("")
        self._lbl_mk_auto = QLabel("")
        for lbl in (self._lbl_dt_auto, self._lbl_ma_auto, self._lbl_mk_auto):
            lbl.setStyleSheet("color: #6a6; font-size: 10px;")

        form.addRow("diff_thresh", row_dt)
        form.addRow("", self._lbl_dt_auto)
        form.addRow("min_area", row_ma)
        form.addRow("", self._lbl_ma_auto)
        form.addRow("morph_kernel", row_mk)
        form.addRow("", self._lbl_mk_auto)
        form.addRow("max_boxes", row_mb)
        form.addRow("ensemble_thresh", row_et)

        lay.addLayout(form)
        root.addWidget(box)

    # ── ピンアート外形照合 ─────────────────────────────────────────────────

    def _build_pin_profile(self, root: QVBoxLayout, config: dict) -> None:
        acfg = config["analysis"]
        box, lay = _group("ピンアート外形照合")

        # ON/OFF チェックボックス
        self._chk_pin = QCheckBox("外形照合を有効にする")
        self._chk_pin.setChecked(acfg.get("pin_profile_enabled", False))

        # サブコントロール（有効時のみ操作可能）
        sub = QWidget()
        sub_form = QFormLayout(sub)
        sub_form.setContentsMargins(16, 0, 0, 0)
        sub_form.setSpacing(4)

        row_burr, self._sl_pin_burr, self._sb_pin_burr = _linked_slider(
            1, 50, acfg.get("pin_burr_threshold", 5))
        row_chip, self._sl_pin_chip, self._sb_pin_chip = _linked_slider(
            1, 50, acfg.get("pin_chip_threshold", 5))

        self._sb_pin_erode = QSpinBox()
        self._sb_pin_erode.setRange(0, 10)
        self._sb_pin_erode.setValue(acfg.get("pin_noise_erode", 1))
        self._sb_pin_erode.setFixedWidth(70)

        self._sb_pin_anchor = QSpinBox()
        self._sb_pin_anchor.setRange(1, 30)
        self._sb_pin_anchor.setValue(acfg.get("pin_anchor_band_min_length", 5))
        self._sb_pin_anchor.setFixedWidth(70)

        sub_form.addRow("バリ閾値 (px)", row_burr)
        sub_form.addRow("欠け閾値 (px)", row_chip)
        sub_form.addRow("ノイズ除去", self._sb_pin_erode)
        sub_form.addRow("アンカー最小長", self._sb_pin_anchor)

        # ON/OFF 連動
        def _update_pin_sub(state: int) -> None:
            enabled = bool(state)
            sub.setEnabled(enabled)

        self._chk_pin.stateChanged.connect(_update_pin_sub)
        sub.setEnabled(acfg.get("pin_profile_enabled", False))

        lay.addWidget(self._chk_pin)
        lay.addWidget(sub)
        root.addWidget(box)

    # ── スロット ──────────────────────────────────────────────────────────

    def _on_connect(self) -> None:
        self.camera_connect_requested.emit(
            self._sb_device.value(), self._sb_width.value(), self._sb_height.value()
        )

    def _on_capture(self) -> None:
        self.capture_requested.emit(
            self._sb_device.value(), self._sb_width.value(), self._sb_height.value()
        )

    def _on_analyze(self) -> None:
        self.analysis_requested.emit(self._collect_params())

    def _on_clip_master_changed(self, text: str) -> None:
        idx = self._clip_master_cb.currentIndex()
        path = self._clip_master_cb.itemData(idx)
        if path is None or idx == 0:
            return
        img = safe_imread(path)
        if img is not None:
            self._lbl_master_file.setText(os.path.basename(path))
            self.master_image_changed.emit(img, ("clip", path))
        else:
            QMessageBox.warning(self, "読み込みエラー", f"画像を読み込めません: {path}")

    def _on_clip_test_changed(self, text: str) -> None:
        idx = self._clip_test_cb.currentIndex()
        path = self._clip_test_cb.itemData(idx)
        if path is None or idx == 0:
            return
        img = safe_imread(path)
        if img is not None:
            self._lbl_test_file.setText(os.path.basename(path))
            self.test_image_changed.emit(img, ("clip", path))
        else:
            QMessageBox.warning(self, "読み込みエラー", f"画像を読み込めません: {path}")

    def _on_master_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "基準画像を選択", "",
            "画像 (*.png *.jpg *.jpeg *.bmp *.heic *.heif)"
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
            img = load_image_from_bytes(data)
            if img is None:
                raise ValueError("画像のデコードに失敗")
            self._lbl_master_file.setText(os.path.basename(path))
            key = (os.path.basename(path), os.path.getsize(path))
            self.master_image_changed.emit(img, key)
        except Exception as exc:
            QMessageBox.warning(self, "読み込みエラー", str(exc))

    def _on_test_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "比較画像を選択", "",
            "画像 (*.png *.jpg *.jpeg *.bmp *.heic *.heif)"
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
            img = load_image_from_bytes(data)
            if img is None:
                raise ValueError("画像のデコードに失敗")
            self._lbl_test_file.setText(os.path.basename(path))
            key = (os.path.basename(path), os.path.getsize(path))
            self.test_image_changed.emit(img, key)
        except Exception as exc:
            QMessageBox.warning(self, "読み込みエラー", str(exc))

    # ── ヘルパー ──────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        """UI から現在のパラメータを収集して返す"""
        return {
            "diff_thresh": self._sb_diff_thresh.value(),
            "min_area": self._sb_min_area.value(),
            "morph_kernel": self._sb_morph_kernel.value(),
            "max_boxes": self._sb_max_boxes.value(),
            "detect_bbox": self._chk_detect_bbox.isChecked(),
            "bbox_type_coloring": self._chk_bbox_coloring.isChecked(),
            "use_foreground_mask": self._chk_fg_mask.isChecked(),
            "use_background_overlap_check": self._chk_bg_overlap.isChecked(),
            "foreground_xor_add": self._chk_fg_xor.isChecked(),
            "use_crop_to_master_fov": self._chk_crop.isChecked(),
            "crop_compare_mode": self._chk_crop_compare.isChecked(),
            "auto_align": self._chk_auto_align.isChecked(),
            "align_mode": self._cb_align_mode.currentData(),
            "preprocess_mode": self._cb_preprocess.currentData(),
            "use_roi": self._chk_roi.isChecked(),
            "roi_x": self._sb_roi_x.value(),
            "roi_y": self._sb_roi_y.value(),
            "roi_w": self._sb_roi_w.value(),
            "roi_h": self._sb_roi_h.value(),
            "crop_method": self._cb_crop_method.currentData(),
            "use_flip_compare": self._chk_flip.isChecked(),
            "quality_check": self._cb_quality_check.isChecked(),
            "camera_enabled": self._cb_camera_enabled.isChecked(),
            # ピンアート外形照合
            "ensemble_thresh": self._sb_ensemble_thresh.value(),
            "pin_profile_enabled": self._chk_pin.isChecked(),
            "pin_burr_threshold": self._sb_pin_burr.value(),
            "pin_chip_threshold": self._sb_pin_chip.value(),
            "pin_noise_erode": self._sb_pin_erode.value(),
            "pin_anchor_band_min_length": self._sb_pin_anchor.value(),
        }

    def set_auto_labels(
        self,
        dt_text: str = "",
        ma_text: str = "",
        mk_text: str = "",
    ) -> None:
        self._lbl_dt_auto.setText(dt_text)
        self._lbl_ma_auto.setText(ma_text)
        self._lbl_mk_auto.setText(mk_text)

    def show_master_warn(self, msg: str) -> None:
        self._lbl_master_warn.setText(msg)
        self._lbl_master_warn.setVisible(bool(msg))

    def show_test_warn(self, msg: str) -> None:
        self._lbl_test_warn.setText(msg)
        self._lbl_test_warn.setVisible(bool(msg))

    def get_camera_enabled(self) -> bool:
        return self._cb_camera_enabled.isChecked()

    def get_quality_check(self) -> bool:
        return self._cb_quality_check.isChecked()
