"""デバッグモード: 3ペインレイアウト + 全機能"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel, QMessageBox, QSizePolicy, QSplitter,
    QStatusBar, QVBoxLayout, QWidget,
)

from ui.state import AppState
from ui.debug.left_panel import LeftPanel
from ui.debug.center_panel import CenterPanel
from ui.debug.right_panel import RightPanel
from ui.threads.camera_thread import CameraThread, CaptureWorker
from ui.threads.analysis_worker import AnalysisWorker, CalibrationWorker
from src.core.quality_check import check_image_quality


def _trim_black_borders(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    if image is None:
        return image
    mask = np.any(image > threshold, axis=2)
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


class DebugWindow(QWidget):
    """デバッグモードのルートウィジェット (QMainWindow に埋め込まれる)"""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._state = AppState()
        self._camera_thread: CameraThread | None = None
        self._capture_worker: CaptureWorker | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._calibration_worker: CalibrationWorker | None = None

        self._build_layout()
        self._connect_signals()

    # ── レイアウト ────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        self._left = LeftPanel(self._config)
        self._center = CenterPanel()
        self._right = RightPanel()

        self._left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._center.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        splitter.addWidget(self._left)
        splitter.addWidget(self._center)
        splitter.addWidget(self._right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        root.addWidget(splitter, stretch=1)

        # ステータスバー
        self._status = QStatusBar()
        self._status.showMessage("準備完了")
        root.addWidget(self._status)

    # ── シグナル接続 ──────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        lp = self._left

        lp.reset_requested.connect(self._on_reset)
        lp.camera_connect_requested.connect(self._on_camera_connect)
        lp.camera_disconnect_requested.connect(self._on_camera_disconnect)
        lp.capture_requested.connect(self._on_capture)
        lp.analysis_requested.connect(self._on_analyze)
        lp.master_image_changed.connect(self._on_master_changed)
        lp.test_image_changed.connect(self._on_test_changed)

    # ── スロット ──────────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        self._stop_camera()
        self._state.reset()
        self._center.clear_view()
        self._right.set_master(None)
        self._right.set_test(None)
        self._status.showMessage("リセット完了")

    def _on_camera_connect(self, device: int, width: int, height: int) -> None:
        if not self._left.get_camera_enabled():
            self._status.showMessage("カメラ接続が無効です")
            return
        self._stop_camera()
        interval = self._config["camera"].get("live_refresh_ms", 200)
        self._camera_thread = CameraThread(device, width, height, interval)
        self._camera_thread.frame_ready.connect(self._on_live_frame)
        self._camera_thread.error_occurred.connect(self._on_camera_error)
        self._camera_thread.start()
        self._state.live = True
        self._status.showMessage("カメラ接続中...")

    def _on_camera_disconnect(self) -> None:
        self._stop_camera()
        self._status.showMessage("カメラ切断")

    def _on_live_frame(self, frame: np.ndarray) -> None:
        self._state.last_frame = frame
        self._center.set_live_frame(frame)

    def _on_camera_error(self, msg: str) -> None:
        self._stop_camera()
        self._status.showMessage(f"カメラエラー: {msg}")

    def _on_capture(self, device: int, width: int, height: int) -> None:
        if not self._left.get_camera_enabled():
            self._status.showMessage("カメラが無効です")
            return
        self._stop_camera()

        cfg_cam = self._config["camera"]
        qc = self._left.get_quality_check()
        worker = CaptureWorker(
            device_index=device, width=width, height=height,
            quality_check=qc,
            max_retry=cfg_cam.get("quality_auto_retry", 3),
            retry_delay_ms=cfg_cam.get("quality_retry_delay_ms", 300),
            white_ratio=cfg_cam.get("quality_white_ratio", 0.05),
            blur_threshold=cfg_cam.get("quality_blur_threshold", 100.0),
        )
        worker.frame_ready.connect(self._on_capture_done)
        worker.quality_warn.connect(lambda msg: self._status.showMessage(f"⚠ {msg}"))
        worker.error_occurred.connect(lambda msg: self._status.showMessage(f"撮影エラー: {msg}"))
        self._capture_worker = worker
        worker.start()
        self._status.showMessage("撮影中...")

    def _on_capture_done(self, frame: np.ndarray) -> None:
        master = self._state.master_override
        if master is not None:
            mh, mw = master.shape[:2]
            fh, fw = frame.shape[:2]
            if fh >= mh and fw >= mw:
                y0 = (fh - mh) // 2
                x0 = (fw - mw) // 2
                frame = frame[y0:y0 + mh, x0:x0 + mw]
        self._state.test_frame = frame
        self._right.set_test(frame)
        self._status.showMessage("撮影完了")

    def _on_master_changed(self, img: np.ndarray, key: tuple) -> None:
        if self._state.master_file_key == key:
            return
        self._state.set_master(img, key)
        display = _trim_black_borders(img)
        self._right.set_master(display)
        self._center.update_calibration(None)
        self._status.showMessage("基準画像を読み込みました")

        # 品質チェック
        if self._left.get_quality_check():
            ok, reason = check_image_quality(img)
            if not ok:
                self._left.show_master_warn(f"品質不良: {reason}")
            else:
                self._left.show_master_warn("")
        else:
            self._left.show_master_warn("")

        # キャリブレーション
        if self._config["analysis"].get("master_calibration_enabled", False):
            self._run_calibration(img)

    def _on_test_changed(self, img: np.ndarray, key: tuple) -> None:
        if self._state.test_file_key == key:
            return
        self._state.set_test(img, key)
        display = _trim_black_borders(img)
        self._right.set_test(display)
        self._status.showMessage("比較画像を読み込みました")

        if self._left.get_quality_check():
            ok, reason = check_image_quality(img)
            if not ok:
                self._left.show_test_warn(f"品質不良: {reason}")
            else:
                self._left.show_test_warn("")
        else:
            self._left.show_test_warn("")

    def _on_analyze(self, params: dict) -> None:
        master = self._state.master_override
        test = self._state.test_frame
        if master is None:
            self._status.showMessage("⚠ 基準画像を選択してください")
            return
        if test is None:
            self._status.showMessage("⚠ 比較画像を撮影または選択してください")
            return

        if self._analysis_worker and self._analysis_worker.isRunning():
            self._status.showMessage("解析中です...")
            return

        self._state.live = False
        self._stop_camera()

        worker = AnalysisWorker(
            master_frame=master,
            test_frame=test,
            params=params,
            config=self._config,
            calibration_result=self._state.calibration_result,
        )
        worker.result_ready.connect(self._on_analysis_done)
        worker.error_occurred.connect(self._on_analysis_error)
        worker.progress.connect(self._status.showMessage)
        self._analysis_worker = worker
        worker.start()
        self._status.showMessage("解析中...")

    def _on_analysis_done(self, result: tuple) -> None:
        self._state.analysis_result = result
        self._state.analysis_error = None

        fig_bytes = result[5]
        self._center.set_result_figure(fig_bytes)
        self._center.update_result(result)

        bbox_count = len(result[4]) if result[4] else 0
        verdict = "✅ OK" if bbox_count == 0 else f"❌ NG ({bbox_count}箇所)"
        self._status.showMessage(
            f"解析完了 — {verdict}  MSE={result[0]:.2f}  SSIM={result[1]:.4f}"
        )

    def _on_analysis_error(self, msg: str) -> None:
        self._state.analysis_error = msg
        self._center.show_analysis_error(msg)
        self._status.showMessage(f"解析エラー: {msg}")

    # ── キャリブレーション ────────────────────────────────────────────────

    def _run_calibration(self, master: np.ndarray) -> None:
        if self._calibration_worker and self._calibration_worker.isRunning():
            return
        worker = CalibrationWorker(master, self._config)
        worker.result_ready.connect(self._on_calibration_done)
        worker.error_occurred.connect(lambda msg: self._status.showMessage(f"キャリブレーションエラー: {msg}"))
        self._calibration_worker = worker
        worker.start()

    def _on_calibration_done(self, result: dict) -> None:
        self._state.calibration_result = result
        self._center.update_calibration(result)
        self._status.showMessage("キャリブレーション完了")

    # ── ヘルパー ──────────────────────────────────────────────────────────

    def _stop_camera(self) -> None:
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None
        self._state.live = False

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)
