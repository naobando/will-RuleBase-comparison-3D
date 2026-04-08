"""ユーザーモード左パネル: カメラサムネイル + ボタン"""
from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from ui.widgets.image_label import ImageLabel


def _separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("color: #3a3a3a;")
    return sep


def _detect_cameras() -> list[tuple[int, str]]:
    """接続中のカメラを検出して (device_index, 表示名) のリストを返す"""
    # PySide6.QtMultimedia で名前付き一覧を取得（OpenCVは使わない: macOS権限問題を避けるため）
    try:
        from PySide6.QtMultimedia import QMediaDevices
        devices = QMediaDevices.videoInputs()
        if devices:
            return [(i, d.description()) for i, d in enumerate(devices)]
    except Exception:
        pass
    # フォールバック: OpenCVを使わずデフォルト番号のみ返す
    return [(0, "カメラ 0 (デフォルト)")]


class UserLeftPanel(QWidget):
    """ユーザーモードの左サイドバー (固定幅)"""

    camera_connect_requested     = Signal()
    camera_disconnect_requested  = Signal()
    start_requested              = Signal()
    stop_requested               = Signal()
    save_requested               = Signal()
    master_file_requested        = Signal()
    master_register_requested    = Signal()
    master_from_camera_requested = Signal()
    test_file_requested          = Signal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._master_img = None
        self._camera_connected = False
        self.setFixedWidth(240)
        self.setStyleSheet("background: #1a1a1a;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._build_camera_preview(layout)
        self._build_master_info(layout)
        layout.addWidget(_separator())
        self._build_buttons(layout)
        layout.addStretch()

    # ── カメラサムネイル ──────────────────────────────────────────────────

    def _build_camera_preview(self, layout: QVBoxLayout) -> None:
        self._thumb = ImageLabel("カメラ未接続")
        self._thumb.setFixedHeight(130)
        self._thumb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._thumb)

    # ── 基準画像情報 ──────────────────────────────────────────────────────

    def _build_master_info(self, layout: QVBoxLayout) -> None:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        self._master_lbl = QLabel("基準画像: 未設定")
        self._master_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._master_lbl.setWordWrap(True)

        self._btn_master = QPushButton("基準画像を変更...")
        self._btn_master.setStyleSheet(
            "background: #2a2a2a; color: #aaa; font-size: 10px; padding: 3px;"
        )
        self._btn_master.clicked.connect(self.master_file_requested)

        self._btn_register = QPushButton("📷 マスター登録（自動切り出し）")
        self._btn_register.setStyleSheet(
            "background: #2a3a2a; color: #8c8; font-size: 10px; padding: 3px; border-radius: 3px;"
        )
        self._btn_register.clicked.connect(self.master_register_requested)

        self._btn_from_camera = QPushButton("🎥 現在のフレームで登録")
        self._btn_from_camera.setStyleSheet(
            "background: #2a2a3a; color: #88c; font-size: 10px; padding: 3px; border-radius: 3px;"
        )
        self._btn_from_camera.setEnabled(False)
        self._btn_from_camera.clicked.connect(self.master_from_camera_requested)

        row_layout.addWidget(self._master_lbl)
        row_layout.addWidget(self._btn_master)
        row_layout.addWidget(self._btn_register)
        row_layout.addWidget(self._btn_from_camera)
        layout.addWidget(row)

    # ── 実行ボタン ────────────────────────────────────────────────────────

    def _build_buttons(self, layout: QVBoxLayout) -> None:
        # ── カメラ選択ドロップダウン ──────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.setSpacing(4)

        self._cam_combo = QComboBox()
        self._cam_combo.setStyleSheet(
            "background: #2a2a2a; color: #ddd; font-size: 10px;"
        )
        self._cam_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._cam_combo.setToolTip("使用するカメラを選択")

        self._btn_scan = QPushButton("🔄")
        self._btn_scan.setFixedWidth(28)
        self._btn_scan.setStyleSheet(
            "background: #2a2a2a; color: #aaa; font-size: 11px; padding: 2px; border-radius: 3px;"
        )
        self._btn_scan.setToolTip("カメラを再スキャン")
        self._btn_scan.clicked.connect(self.refresh_cameras)

        sel_row.addWidget(self._cam_combo)
        sel_row.addWidget(self._btn_scan)
        layout.addLayout(sel_row)

        # ── カメラ接続 / 接続解除（横並び）──────────────────────────────
        cam_row = QHBoxLayout()
        cam_row.setSpacing(4)

        self._btn_cam_connect = QPushButton("カメラ接続")
        self._btn_cam_connect.setStyleSheet(
            "background: #1a3a3a; color: #6cc; font-size: 11px; "
            "padding: 5px; border-radius: 4px;"
        )
        self._btn_cam_connect.clicked.connect(self.camera_connect_requested)

        self._btn_cam_disconnect = QPushButton("接続解除")
        self._btn_cam_disconnect.setStyleSheet(
            "background: #2a2a2a; color: #888; font-size: 11px; "
            "padding: 5px; border-radius: 4px;"
        )
        self._btn_cam_disconnect.setEnabled(False)
        self._btn_cam_disconnect.clicked.connect(self.camera_disconnect_requested)

        cam_row.addWidget(self._btn_cam_connect)
        cam_row.addWidget(self._btn_cam_disconnect)
        layout.addLayout(cam_row)

        # ── 入力画像ファイル選択 ───────────────────────────────────────────
        self._btn_test_file = QPushButton("📂 ファイルから検査")
        self._btn_test_file.setStyleSheet(
            "background: #2a3a4a; color: #9cf; padding: 6px; border-radius: 4px;"
        )
        self._btn_test_file.setToolTip("画像ファイルを選択して検査する（カメラ不要）")
        self._btn_test_file.clicked.connect(self.test_file_requested)
        layout.addWidget(self._btn_test_file)

        # ── 検査開始 / 停止 / 保存 ───────────────────────────────────────
        self._btn_start = QPushButton("検査開始")
        self._btn_start.setStyleSheet(
            "background: #1b5e20; color: white; font-weight: bold; "
            "padding: 8px; font-size: 13px; border-radius: 4px;"
        )
        self._btn_start.clicked.connect(self.start_requested)

        self._btn_stop = QPushButton("検査停止")
        self._btn_stop.setStyleSheet(
            "background: #555; color: #ccc; padding: 6px; border-radius: 4px;"
        )
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested)

        self._btn_save = QPushButton("画像保存")
        self._btn_save.setStyleSheet(
            "background: #1a3a5a; color: #9cf; padding: 6px; border-radius: 4px;"
        )
        self._btn_save.clicked.connect(self.save_requested)

        layout.addWidget(self._btn_start)
        layout.addWidget(self._btn_stop)
        layout.addWidget(self._btn_save)

        # ボタン生成後にイベントループ開始後（遅延）スキャン
        QTimer.singleShot(500, self.refresh_cameras)

    # ── 公開メソッド ──────────────────────────────────────────────────────

    def refresh_cameras(self) -> None:
        """カメラ一覧を再スキャンしてドロップダウンを更新する"""
        cameras = _detect_cameras()
        self._cam_combo.clear()
        for dev_idx, name in cameras:
            self._cam_combo.addItem(name, userData=dev_idx)
        has_camera = len(cameras) > 0
        self._btn_cam_connect.setEnabled(has_camera and not self._camera_connected)

    def set_master_info(self, name: str) -> None:
        self._master_lbl.setText(f"基準: {name}")
        self._master_lbl.setStyleSheet("color: #6a6; font-size: 10px;")

    def set_master_thumb(self, img) -> None:
        self._master_img = img
        if not self._camera_connected:
            self._thumb.set_cv2(img)

    def set_camera_connected(self, connected: bool) -> None:
        """カメラ接続状態を反映（映像のみ、検査は別）"""
        self._camera_connected = connected
        self._btn_cam_connect.setEnabled(not connected)
        self._btn_cam_disconnect.setEnabled(connected)
        self._btn_scan.setEnabled(not connected)   # 接続中はスキャン不可
        self._cam_combo.setEnabled(not connected)  # 接続中は切替不可
        self._btn_from_camera.setEnabled(connected)
        if not connected and self._master_img is not None:
            self._thumb.set_cv2(self._master_img)

    def set_running(self, running: bool) -> None:
        """検査（解析）の実行状態を反映"""
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)

    def update_thumb(self, img) -> None:
        self._thumb.set_cv2(img)

    def get_diff_thresh(self) -> int:
        return int(self._config["analysis"].get("diff_thresh", 13))

    def get_preprocess_mode(self) -> str:
        return self._config["analysis"].get("preprocess_mode", "luminance")

    def get_device_index(self) -> int:
        idx = self._cam_combo.currentData()
        return idx if idx is not None else 0

    def get_flip(self) -> bool:
        return bool(self._config["analysis"].get("flip_compare", False))
