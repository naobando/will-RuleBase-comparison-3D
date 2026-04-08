"""ユーザーモード: シンプル操作UI"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
from src.utils.image_utils import safe_imread, safe_imwrite
from PySide6.QtCore import QObject, Qt, QPoint, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget,
)

from ui.state import AppState
from ui.widgets.image_label import ImageLabel
from ui.user.user_left_panel import UserLeftPanel
from ui.threads.camera_thread import CameraThread, CaptureWorker
from ui.threads.analysis_worker import AnalysisWorker
from src.core.quality_check import check_image_quality
from src.core.master_registration import extract_master, validate_master_registration
from config import save_config_key


def _camera_auth_status() -> int:
    """macOS カメラ権限ステータスを返す。0=未確定, 2=拒否, 3=許可。pyobjc 未インストール時は 3(許可扱い)"""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        return int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))
    except Exception:
        return 3  # pyobjc なし → 許可扱いでそのまま続行


class _PermissionBridge(QObject):
    """GCDスレッド → Qtメインスレッドへ権限結果を安全に渡すブリッジ"""
    granted = Signal(bool)


def _request_camera_permission_async(callback) -> "_PermissionBridge | None":
    """macOS カメラ権限ダイアログを表示し、結果を callback(bool) でメインスレッドに返す。
    戻り値の _PermissionBridge を保持しておかないとGCに回収されるので注意。"""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

        bridge = _PermissionBridge()
        bridge.granted.connect(callback)

        def handler(granted):
            # GCDスレッドから Qt シグナルを emit → Qt が自動でメインスレッドにキュー
            bridge.granted.emit(bool(granted))

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeVideo, handler
        )
        return bridge  # 呼び出し元で保持すること
    except Exception:
        callback(True)  # pyobjc なし → 許可扱い
        return None

# config.yaml のパス（エントリポイントと同じディレクトリ）
_CONFIG_PATH = "config.yaml"


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


class UserWindow(QWidget):
    """ユーザーモードのルートウィジェット"""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._state = AppState()
        self._camera_thread: CameraThread | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._last_result: tuple | None = None
        self._last_fig_bytes: bytes | None = None
        self._camera_connected = False   # カメラ映像が流れている
        self._running = False            # 解析（検査）が走っている
        self._snapshot_taken = False     # スナップショット表示中
        self._permission_bridge = None   # GC回収防止用

        self._build_layout()
        self._connect_signals()
        self._load_master()

    # ── レイアウト ────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._left = UserLeftPanel(self._config)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 4)
        right_layout.setSpacing(8)

        self._verdict_lbl = QLabel("待機中")
        self._verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_lbl.setMinimumHeight(120)
        self._verdict_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._set_verdict_standby()

        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(8)

        self._master_view = _LabeledImage("基準画像")   # 常時表示（左）
        self._camera_view = _LabeledImage("撮影画像")   # スナップショット（中央）
        self._result_view = _LabeledImage("結果画像")   # 解析結果（右）
        images_layout.addWidget(self._master_view, stretch=1)
        images_layout.addWidget(self._camera_view, stretch=1)
        images_layout.addWidget(self._result_view, stretch=1)

        # ── ボトムバー（ステータス + 解析中スピナー）──────────────────────
        # QStatusBar.showMessage() は addWidget() を隠すため
        # 独自の HBox レイアウトで左にステータス・右にスピナーを配置する
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(24)
        bottom_bar.setStyleSheet("background: #1e1e1e;")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 2, 6, 2)
        bottom_layout.setSpacing(6)

        self._status_lbl = QLabel("準備完了")
        self._status_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        bottom_layout.addWidget(self._status_lbl, stretch=1)

        self._spinner_label = QLabel("解析中...")
        self._spinner_label.setStyleSheet("color: #4a9eff; font-size: 11px;")
        self._spinner_label.hide()

        self._spinner = QProgressBar()
        self._spinner.setRange(0, 0)
        self._spinner.setFixedSize(100, 12)
        self._spinner.setTextVisible(False)
        self._spinner.setStyleSheet(
            "QProgressBar { border: 1px solid #444; border-radius: 2px; background: #2a2a2a; }"
            "QProgressBar::chunk { background: #4a9eff; border-radius: 1px; }"
        )
        self._spinner.hide()

        bottom_layout.addWidget(self._spinner_label)
        bottom_layout.addWidget(self._spinner)

        # showMessage 互換ラッパー（QStatusBar の代替）
        self._status = type("_StatusProxy", (), {
            "showMessage": lambda _self, msg: self._status_lbl.setText(msg),
        })()

        right_layout.addWidget(self._verdict_lbl, stretch=0)
        right_layout.addWidget(images_widget, stretch=1)
        right_layout.addWidget(bottom_bar)

        root.addWidget(self._left)
        root.addWidget(right_widget, stretch=1)

    # ── シグナル接続 ──────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        lp = self._left
        lp.camera_connect_requested.connect(self._on_camera_connect)
        lp.camera_disconnect_requested.connect(self._on_camera_disconnect)
        lp.start_requested.connect(self._on_start)
        lp.stop_requested.connect(self._on_stop)
        lp.save_requested.connect(self._on_save)
        lp.master_file_requested.connect(self._on_master_file)
        lp.master_register_requested.connect(self._on_master_register)
        lp.master_from_camera_requested.connect(self._on_master_from_camera)
        lp.test_file_requested.connect(self._on_test_file)

    # ── マスター画像ロード ────────────────────────────────────────────────

    def _load_master(self) -> None:
        master_dir = self._config["guide"].get("master_dir", "master")
        if not os.path.isdir(master_dir):
            return
        for f in sorted(os.listdir(master_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                path = os.path.join(master_dir, f)
                img = safe_imread(path)
                if img is not None:
                    self._state.set_master(img, ("clip", path))
                    self._left.set_master_info(f)
                    self._left.set_master_thumb(img)
                    self._master_view.set_cv2(img)   # 基準画像エリアに表示
                    self._status.showMessage(f"基準画像: {f}")
                    break

    # ── スロット ──────────────────────────────────────────────────────────

    def _on_camera_connect(self) -> None:
        """カメラ映像のみ開始（解析はしない）"""
        if self._camera_connected:
            return
        status = _camera_auth_status()
        if status == 3:
            # 既に許可済み → そのまま接続
            self._do_open_camera()
        elif status == 0:
            # 未確定 → 許可ダイアログを非同期で表示、結果をコールバックで受け取る
            self._status.showMessage("カメラ使用許可を要求中...")
            self._permission_bridge = _request_camera_permission_async(
                self._on_camera_permission_result
            )
        else:
            # 拒否 (2) または制限 (1) → システム設定へ誘導
            self._status.showMessage(
                "カメラアクセスが拒否されています。"
                "システム設定 → プライバシーとセキュリティ → カメラ で許可してください。"
            )
            self._open_camera_settings()

    def _open_camera_settings(self) -> None:
        """システム設定のカメラ権限ページを開き、許可されたら自動接続する"""
        import subprocess
        subprocess.Popen([
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
        ])
        self._status.showMessage(
            "システム設定が開きました。「Python」をオンにして「カメラ接続」を再度押してください。"
        )
        # 許可されるまで定期チェック（最大60秒）して自動接続
        self._perm_check_count = 0
        self._perm_check_timer = QTimer(self)
        self._perm_check_timer.timeout.connect(self._poll_camera_permission)
        self._perm_check_timer.start(1000)

    def _poll_camera_permission(self) -> None:
        """システム設定で権限が付与されたか1秒ごとにチェックし、付与されたら自動接続"""
        self._perm_check_count += 1
        if _camera_auth_status() == 3:
            self._perm_check_timer.stop()
            self._status.showMessage("カメラ権限を確認しました。接続中...")
            self._do_open_camera()
            return
        if self._perm_check_count >= 60:
            self._perm_check_timer.stop()
            self._status.showMessage(
                "タイムアウト。システム設定で「Python」を許可後、「カメラ接続」を押してください。"
            )

    def _on_camera_permission_result(self, granted: bool) -> None:
        """権限ダイアログの結果を受け取ってカメラ接続を継続"""
        self._permission_bridge = None  # 参照解放
        if not granted:
            self._status.showMessage(
                "カメラへのアクセスが拒否されました。"
                "システム設定 → プライバシーとセキュリティ → カメラ で許可してください。"
            )
            self._open_camera_settings()
            return
        self._do_open_camera()

    def _do_open_camera(self) -> None:
        """カメラを実際に開いてスレッドを開始する（権限確認済み前提）"""
        if self._camera_connected:
            return
        dev_idx = self._left.get_device_index()
        # macOS AVFoundation: メインスレッドでカメラを開いてスレッドに渡す
        cap = cv2.VideoCapture(dev_idx)
        if not cap.isOpened():
            cap.release()
            self._status.showMessage(
                f"カメラ(index={dev_idx})を開けませんでした。デバイスを確認してください。"
            )
            return
        cam_cfg = self._config["camera"]
        self._camera_thread = CameraThread(
            device_index=dev_idx,
            width=cam_cfg.get("capture_width", 0),
            height=cam_cfg.get("capture_height", 0),
            interval_ms=cam_cfg.get("live_refresh_ms", 200),
            cap=cap,  # メインスレッドで開いたキャプチャを渡す
        )
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.error_occurred.connect(self._on_camera_error)
        self._camera_thread.start()
        self._camera_connected = True
        self._left.set_camera_connected(True)
        self._status.showMessage("カメラ接続済み")

    def _set_analyzing(self, running: bool) -> None:
        """解析状態を一元管理（_running・ボタン・スピナー）"""
        self._running = running
        self._left.set_running(running)
        if running:
            self._spinner.show()
            self._spinner_label.show()
        else:
            self._spinner.hide()
            self._spinner_label.hide()

    def _on_camera_disconnect(self) -> None:
        """カメラ映像を停止（検査中なら検査も止める）"""
        if self._running:
            self._set_analyzing(False)
        self._snapshot_taken = False
        self._stop_camera()
        self._camera_connected = False
        self._left.set_camera_connected(False)
        self._set_verdict_standby()
        self._status.showMessage("カメラ接続解除")

    def _on_start(self) -> None:
        """検査を1回実行（ボタン押下時点のフレームを使用）"""
        if self._state.master_override is None:
            QMessageBox.warning(self, "警告", "基準画像が設定されていません。")
            return
        if self._running:
            return  # 解析中は多重実行しない
        if not self._camera_connected:
            self._on_camera_connect()
            return  # カメラ接続完了後に再度「検査開始」を押す
        if self._state.last_frame is None:
            self._status.showMessage("カメラフレームがありません。少し待ってから再度押してください。")
            return
        # ボタン押下時点のフレームをキャプチャして撮影画像エリアに固定表示
        frame = self._state.last_frame.copy()
        self._snapshot_taken = True
        self._camera_view.set_cv2(frame)   # カメラビューをスナップショットに差し替え
        self._set_analyzing(True)
        self._set_verdict_standby("解析中...")
        self._status.showMessage("解析中...")
        self._state.test_frame = frame
        self._run_analysis(frame)

    def _on_stop(self) -> None:
        """解析中止（カメラ映像は維持する）"""
        self._set_analyzing(False)
        self._set_verdict_standby()
        self._status.showMessage("解析中止（カメラは接続中）")

    def _on_frame(self, frame: np.ndarray) -> None:
        self._state.last_frame = frame
        self._left.update_thumb(frame)
        # スナップショット未取得中はライブ映像をカメラビューに表示
        if not self._snapshot_taken:
            self._camera_view.set_cv2(frame)

    def _on_camera_error(self, msg: str) -> None:
        self._set_analyzing(False)
        self._camera_connected = False
        self._stop_camera()
        self._left.set_camera_connected(False)
        self._set_verdict_standby()
        self._status.showMessage(f"カメラエラー: {msg}")

    def _on_master_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "基準画像を選択", "",
            "画像 (*.png *.jpg *.jpeg *.bmp *.heic *.heif)"
        )
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            QMessageBox.warning(self, "読み込みエラー", f"画像を読み込めません: {path}")
            return
        self._state.set_master(img, ("file", path))
        self._left.set_master_info(os.path.basename(path))
        self._left.set_master_thumb(img)
        self._master_view.set_cv2(img)   # 基準画像エリアに表示
        self._status.showMessage(f"基準画像: {os.path.basename(path)}")
        self._set_verdict_standby()

    def _on_test_file(self) -> None:
        """画像ファイルを選択して即座に検査を1回実行する（カメラ不要）"""
        # _running 中でも割り込み可能にする（前回の解析を破棄）
        if self._running:
            self._set_analyzing(False)
            if self._analysis_worker and self._analysis_worker.isRunning():
                self._analysis_worker.requestInterruption()

        path, _ = QFileDialog.getOpenFileName(
            self, "検査画像を選択", "",
            "画像 (*.png *.jpg *.jpeg *.bmp *.heic *.heif)"
        )
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            QMessageBox.warning(self, "読み込みエラー", f"画像を読み込めません: {path}")
            return

        # ファイル画像をカメラビューに固定表示
        self._snapshot_taken = True
        self._camera_view.set_cv2(img)
        self._state.test_frame = img
        self._status.showMessage(f"入力: {os.path.basename(path)}")

        # マスター未設定なら表示だけして終了
        if self._state.master_override is None:
            self._status.showMessage(
                f"入力: {os.path.basename(path)}  ※基準画像を設定してから「検査開始」を押してください"
            )
            return

        self._set_analyzing(True)
        self._set_verdict_standby("解析中...")
        self._status.showMessage(f"ファイル検査: {os.path.basename(path)}")
        self._run_analysis(img)

    def _on_master_register(self) -> None:
        """広角画像ファイルを選択してマスター自動登録"""
        path, _ = QFileDialog.getOpenFileName(
            self, "広角画像を選択（マスター自動登録）", "",
            "画像 (*.png *.jpg *.jpeg *.bmp *.heic *.heif)"
        )
        if not path:
            return
        src = safe_imread(path)
        if src is None:
            QMessageBox.warning(self, "読み込みエラー", f"画像を読み込めません:\n{path}")
            return
        self._run_register_dialog(src, os.path.basename(path))

    def _on_master_from_camera(self) -> None:
        """現在のライブフレームをそのままマスター登録ダイアログへ"""
        frame = self._state.last_frame
        if frame is None:
            QMessageBox.warning(self, "警告", "カメラフレームがありません。")
            return
        self._run_register_dialog(frame.copy(), "camera_frame")

    def _run_register_dialog(self, src: np.ndarray, source_name: str) -> None:
        """共通: src から extract_master → ダイアログ表示 → 登録"""
        # config に保存済みの選択範囲があれば自動適用してから初回抽出
        saved_crop = self._config.get("registration_crop")
        initial_src = src
        if saved_crop:
            x, y, w, h = (
                saved_crop["x"], saved_crop["y"],
                saved_crop["w"], saved_crop["h"],
            )
            ih, iw = src.shape[:2]
            x = max(0, min(x, iw - 1))
            y = max(0, min(y, ih - 1))
            w = min(w, iw - x)
            h = min(h, ih - y)
            if w > 0 and h > 0:
                initial_src = src[y:y + h, x:x + w]

        # リトライごとに試すパラメータ (デフォルト → 精密 → 粗め)
        _RETRY_PARAMS = [
            {},
            {"poly_epsilon_ratio": 0.02},
            {"poly_epsilon_ratio": 0.08, "blur_ksize": 9},
        ]
        _VALIDATION_THRESHOLD = 0.65
        _MAX_RETRIES = len(_RETRY_PARAMS)

        best_master: np.ndarray | None = None
        best_info: dict = {}
        best_score: float = 0.0

        self._status.showMessage("部品を自動抽出中...")
        for attempt, params in enumerate(_RETRY_PARAMS):
            try:
                extracted, info = extract_master(initial_src, **params)
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    QMessageBox.critical(self, "自動登録エラー", f"部品抽出に失敗しました:\n{exc}")
                    self._status.showMessage("マスター自動登録: エラー")
                    return
                continue

            score, is_valid, val_msg = validate_master_registration(initial_src, extracted,
                                                                     _VALIDATION_THRESHOLD)
            info["validation_score"] = score
            info["validation_msg"] = val_msg

            if score > best_score:
                best_score = score
                best_master = extracted
                best_info = info

            if is_valid:
                self._status.showMessage(f"自動抽出完了 [{attempt + 1}回目] {val_msg}")
                break
            self._status.showMessage(f"再試行中... [{attempt + 1}/{_MAX_RETRIES}] {val_msg}")

        # 全リトライ後もスコアが低い場合は手動登録を促す
        if best_score < _VALIDATION_THRESHOLD:
            reply = QMessageBox.question(
                self, "自動登録の品質が低い",
                f"自動切り出しの精度スコア: {best_score:.2f}（基準: {_VALIDATION_THRESHOLD}）\n"
                "切り出しがうまくいっていない可能性があります。\n\n"
                "このまま続けますか？（プレビューで手動修正できます）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._status.showMessage("マスター自動登録: 手動登録を推奨")
                return

        extracted, info = best_master, best_info

        dlg = _MasterPreviewDialog(src, extracted, info, saved_crop, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._status.showMessage("マスター自動登録: キャンセル")
            return

        # 選択範囲を保存する指示があれば config.yaml に書き込む
        crop_to_save = dlg.get_crop_to_save()
        if crop_to_save is not None:
            x, y, w, h = crop_to_save
            self._config["registration_crop"] = {"x": x, "y": y, "w": w, "h": h}
            save_config_key("registration_crop", {"x": x, "y": y, "w": w, "h": h},
                            _CONFIG_PATH)
            self._status.showMessage(f"選択範囲を保存しました: ({x},{y}) {w}×{h}")

        final = dlg.get_extracted()
        basename = os.path.splitext(source_name)[0] + "_registered.png"
        self._state.set_master(final, ("register", source_name))
        self._left.set_master_info(basename)
        self._left.set_master_thumb(final)
        self._master_view.set_cv2(final)   # 基準画像エリアに表示
        self._set_verdict_standby()
        self._status.showMessage(f"マスター登録完了: {basename}")

    def _on_save(self) -> None:
        if self._last_result is None:
            self._status.showMessage("保存する結果がありません")
            return
        base_dir = self._config["output"]["base_dir"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.getcwd(), base_dir, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        try:
            if self._state.master_override is not None:
                safe_imwrite(os.path.join(out_dir, "master.png"), self._state.master_override)
            if self._state.test_frame is not None:
                safe_imwrite(os.path.join(out_dir, "test.png"), self._state.test_frame)
            if self._last_fig_bytes:
                with open(os.path.join(out_dir, "result.png"), "wb") as f:
                    f.write(self._last_fig_bytes)
            self._status.showMessage(f"保存完了: {out_dir}")
        except Exception as exc:
            self._status.showMessage(f"保存エラー: {exc}")

    # ── 解析 ──────────────────────────────────────────────────────────────

    def _run_analysis(self, test_frame: np.ndarray) -> None:
        master = self._state.master_override
        if master is None:
            return

        acfg = self._config["analysis"]
        params = {
            "diff_thresh": self._left.get_diff_thresh(),
            "min_area": acfg.get("min_area", 850),
            "morph_kernel": acfg.get("morph_kernel", 9),
            "max_boxes": acfg.get("max_boxes", 10),
            "detect_bbox": True,
            "bbox_type_coloring": acfg.get("bbox_type_coloring", True),
            "use_foreground_mask": acfg.get("foreground_mask", True),
            "use_background_overlap_check": acfg.get("background_overlap_check", False),
            "foreground_xor_add": acfg.get("foreground_xor_add", False),
            "use_crop_to_master_fov": acfg.get("crop_to_master_fov", True),
            "crop_compare_mode": acfg.get("crop_compare_mode", True),
            "auto_align": acfg.get("auto_align", True),
            "align_mode": acfg.get("align_mode", "edge"),
            "preprocess_mode": self._left.get_preprocess_mode(),
            "use_roi": False,
            "roi_x": 0, "roi_y": 0, "roi_w": 0, "roi_h": 0,
            "use_flip_compare": acfg.get("flip_compare", False),
        }

        worker = AnalysisWorker(
            master_frame=master,
            test_frame=test_frame,
            params=params,
            config=self._config,
            calibration_result=self._state.calibration_result,
        )
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_analysis_error)
        self._analysis_worker = worker
        worker.start()

    def _on_result(self, result: tuple) -> None:
        try:
            self._last_result = result
            ssim = result[1]
            bboxes = result[4]
            fig_bytes = result[5]
            visB_img = result[17] if len(result) > 17 else None  # cv2 BGR 画像（比較画像+BBOX）
            self._last_fig_bytes = fig_bytes

            bbox_count = len(bboxes) if bboxes else 0
            is_ok = bbox_count == 0

            if is_ok:
                self._set_verdict_ok()
            else:
                self._set_verdict_ng(bbox_count)

            # 結果エリアには比較画像+BBOX を cv2 画像として直接表示（白背景なし）
            if visB_img is not None:
                self._result_view.set_cv2(visB_img)
            elif fig_bytes:
                pixmap = QPixmap()
                pixmap.loadFromData(fig_bytes)
                self._result_view.set_pixmap(pixmap)

            # bboxes はタプル (x,y,w,h) のリストの場合もあるので .get() は使わない
            self._status.showMessage(
                f"判定: {'OK' if is_ok else 'NG'}  SSIM: {ssim:.3f}  検出: {bbox_count}箇所"
            )
        except Exception as e:
            self._status.showMessage(f"結果表示エラー: {e}")
        finally:
            # 例外が起きても必ずスピナーを止めてボタンを戻す
            self._set_analyzing(False)

    def _on_analysis_error(self, msg: str) -> None:
        self._status.showMessage(f"解析エラー: {msg}")
        self._set_analyzing(False)

    # ── 判定表示 ──────────────────────────────────────────────────────────

    def _set_verdict_ok(self) -> None:
        self._verdict_lbl.setText("OK")
        self._verdict_lbl.setStyleSheet(
            "background: #2e7d32; color: white; font-size: 64px; font-weight: bold; border-radius: 8px;"
        )

    def _set_verdict_ng(self, count: int) -> None:
        self._verdict_lbl.setText(f"NG  ({count}箇所)")
        self._verdict_lbl.setStyleSheet(
            "background: #c62828; color: white; font-size: 64px; font-weight: bold; border-radius: 8px;"
        )

    def _set_verdict_standby(self, text: str = "待機中") -> None:
        self._verdict_lbl.setText(text)
        self._verdict_lbl.setStyleSheet(
            "background: #37474f; color: #aaa; font-size: 48px; font-weight: bold; border-radius: 8px;"
        )

    # ── ヘルパー ──────────────────────────────────────────────────────────

    def _stop_camera(self) -> None:
        if self._camera_thread is not None:
            self._camera_thread.stop()
            self._camera_thread = None

    def closeEvent(self, event) -> None:
        self._running = False
        self._camera_connected = False
        self._stop_camera()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# 汎用ウィジェット
# ─────────────────────────────────────────────────────────────────────────────

class _LabeledImage(QWidget):
    """ラベル + ImageLabel のペア"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel(title)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img = ImageLabel(title)
        layout.addWidget(lbl)
        layout.addWidget(self._img, stretch=1)

    def set_cv2(self, img: np.ndarray | None) -> None:
        self._img.set_cv2(img)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._img.set_pixmap_source(pixmap)


class _SelectableImageLabel(QLabel):
    """マウスドラッグで領域選択できる画像ラベル（選択は画像座標で管理）"""

    selection_changed = Signal()

    def __init__(self, placeholder: str = "画像なし", parent=None):
        super().__init__(parent)
        self._source_img: np.ndarray | None = None
        self._source_pixmap: QPixmap | None = None
        # 選択範囲を画像座標 (x, y, w, h) で保持 → リサイズしても正確
        self._sel_img: tuple[int, int, int, int] | None = None
        self._drag_start_img: tuple[int, int] | None = None
        self._dragging = False

        self.setText(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background: #1a1a1a; color: #666; border: 1px solid #333;")
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── 公開 API ──────────────────────────────────────────────────────────

    def set_cv2(self, img: np.ndarray | None) -> None:
        self._source_img = img
        if img is not None:
            from ui.utils import cv2_to_qpixmap
            self._source_pixmap = cv2_to_qpixmap(img)
        else:
            self._source_pixmap = None
        self._sel_img = None
        self._update_display()

    def set_selection(self, x: int, y: int, w: int, h: int) -> None:
        """画像座標で初期選択範囲を設定する"""
        if self._source_pixmap is None:
            return
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        self._sel_img = (x, y, w, h) if w > 4 and h > 4 else None
        self._update_display()

    def get_crop(self) -> tuple[int, int, int, int] | None:
        return self._sel_img

    def has_selection(self) -> bool:
        return self._sel_img is not None

    def clear_selection(self) -> None:
        self._sel_img = None
        self._update_display()

    # ── 座標変換 ──────────────────────────────────────────────────────────

    def _display_rect(self) -> QRect:
        """表示中の画像矩形（ラベル座標系）"""
        if self._source_pixmap is None:
            return QRect()
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        scale = min(lw / pw, lh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        ox, oy = (lw - sw) // 2, (lh - sh) // 2
        return QRect(ox, oy, sw, sh)

    def _label_pt_to_image(self, pt: QPoint) -> tuple[int, int]:
        """ラベル座標 → 画像座標（クランプあり）"""
        disp = self._display_rect()
        if not disp.isValid() or self._source_pixmap is None:
            return (0, 0)
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        x = (pt.x() - disp.x()) * img_w / disp.width()
        y = (pt.y() - disp.y()) * img_h / disp.height()
        return (
            max(0, min(img_w - 1, int(x))),
            max(0, min(img_h - 1, int(y))),
        )

    def _sel_img_to_label_rect(self) -> QRect | None:
        """_sel_img（画像座標）→ ラベル座標の QRect"""
        if self._sel_img is None or self._source_pixmap is None:
            return None
        disp = self._display_rect()
        if not disp.isValid():
            return None
        img_w = self._source_pixmap.width()
        img_h = self._source_pixmap.height()
        sx = disp.width() / img_w
        sy = disp.height() / img_h
        x, y, w, h = self._sel_img
        return QRect(
            disp.x() + int(x * sx),
            disp.y() + int(y * sy),
            max(1, int(w * sx)),
            max(1, int(h * sy)),
        )

    # ── 描画 ──────────────────────────────────────────────────────────────

    def _update_display(self) -> None:
        if self._source_pixmap is None:
            self.clear()
            return
        lw, lh = max(self.width(), 1), max(self.height(), 1)
        canvas = QPixmap(lw, lh)
        canvas.fill(QColor("#1a1a1a"))

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        disp = self._display_rect()
        scaled = self._source_pixmap.scaled(
            disp.width(), disp.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(disp.x(), disp.y(), scaled)

        r = self._sel_img_to_label_rect()
        if r is not None:
            painter.fillRect(r, QColor(0, 200, 80, 40))
            pen = QPen(QColor(0, 220, 80), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(r)

        painter.end()
        self.setPixmap(canvas)

    # ── マウスイベント ────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._source_pixmap:
            self._drag_start_img = self._label_pt_to_image(event.position().toPoint())
            self._sel_img = None
            self._dragging = True
            self._update_display()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._drag_start_img and self._source_pixmap:
            cur = self._label_pt_to_image(event.position().toPoint())
            x1, y1 = self._drag_start_img
            x2, y2 = cur
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            self._sel_img = (x, y, w, h) if w > 0 and h > 0 else None
            self._update_display()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if self._sel_img and self._sel_img[2] > 4 and self._sel_img[3] > 4:
                self.selection_changed.emit()
            else:
                self._sel_img = None
            self._update_display()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()


# ─────────────────────────────────────────────────────────────────────────────
# マスター登録プレビューダイアログ
# ─────────────────────────────────────────────────────────────────────────────

class _MasterPreviewDialog(QDialog):
    """マスター自動登録: 範囲選択→再抽出→確認ダイアログ"""

    def __init__(
        self,
        src: np.ndarray,
        extracted: np.ndarray,
        info: dict,
        saved_crop: dict | None,
        parent=None,
    ):
        super().__init__(parent)
        self._src = src
        self._extracted = extracted
        self._last_crop: tuple[int, int, int, int] | None = None  # 最後に使った crop

        self.setWindowTitle("マスター自動登録 — 範囲選択・結果確認")
        self.setMinimumSize(1060, 660)
        self.setStyleSheet("background: #1e1e1e; color: #ddd;")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── 操作説明 ──────────────────────────────────────────────────────
        hint = QLabel(
            "左の元画像をドラッグして部品の範囲を囲んでください。"
            "「選択範囲で再抽出」を押すと右に結果が反映されます。"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)

        # ── 情報ラベル ────────────────────────────────────────────────────
        self._info_lbl = QLabel()
        self._info_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_info(info, extracted)
        root.addWidget(self._info_lbl)

        # ── 画像エリア ────────────────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        lbl_src = QLabel("元画像（広角） — ドラッグで範囲指定")
        lbl_src.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sel_img = _SelectableImageLabel("元画像")
        self._sel_img.set_cv2(src)
        self._sel_img.selection_changed.connect(self._on_selection_changed)
        left_col.addWidget(lbl_src)
        left_col.addWidget(self._sel_img, stretch=1)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        lbl_ext = QLabel("抽出結果（マスター候補）")
        lbl_ext.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl_ext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ext_view = ImageLabel("抽出結果")
        self._ext_view.set_cv2(extracted)
        right_col.addWidget(lbl_ext)
        right_col.addWidget(self._ext_view, stretch=1)

        img_row.addLayout(left_col, stretch=1)
        img_row.addLayout(right_col, stretch=1)
        root.addLayout(img_row, stretch=1)

        # ── 保存済み範囲を初期選択として反映 ─────────────────────────────
        if saved_crop:
            self._sel_img.set_selection(
                saved_crop["x"], saved_crop["y"],
                saved_crop["w"], saved_crop["h"],
            )
            self._last_crop = (
                saved_crop["x"], saved_crop["y"],
                saved_crop["w"], saved_crop["h"],
            )

        # ── 再抽出・選択解除ボタン ────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self._btn_reextract = QPushButton("選択範囲で再抽出")
        self._btn_reextract.setEnabled(self._sel_img.has_selection())
        self._btn_reextract.setStyleSheet(
            "background: #1a3a5a; color: #9cf; font-weight: bold; "
            "padding: 7px 18px; border-radius: 4px;"
        )
        self._btn_reextract.clicked.connect(self._on_reextract)

        btn_clear = QPushButton("選択を解除（全体で再試行）")
        btn_clear.setStyleSheet(
            "background: #2a2a2a; color: #aaa; padding: 7px 14px; border-radius: 4px;"
        )
        btn_clear.clicked.connect(self._on_clear_selection)

        # 範囲保存チェックボックス
        self._chk_save = QCheckBox("この選択範囲を次回から自動適用する")
        self._chk_save.setChecked(saved_crop is not None)
        self._chk_save.setStyleSheet("color: #8c8; font-size: 10px;")

        action_row.addWidget(self._btn_reextract)
        action_row.addWidget(btn_clear)
        action_row.addStretch()
        action_row.addWidget(self._chk_save)
        root.addLayout(action_row)

        # ── OK / キャンセル ───────────────────────────────────────────────
        btns = QDialogButtonBox()
        btn_ok = btns.addButton(
            "この画像をマスターに登録", QDialogButtonBox.ButtonRole.AcceptRole
        )
        btn_cancel = btns.addButton("キャンセル", QDialogButtonBox.ButtonRole.RejectRole)
        btn_ok.setStyleSheet(
            "background: #1b5e20; color: white; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px;"
        )
        btn_cancel.setStyleSheet(
            "background: #3a3a3a; color: #aaa; padding: 8px 16px; border-radius: 4px;"
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── 公開メソッド ──────────────────────────────────────────────────────

    def get_extracted(self) -> np.ndarray:
        return self._extracted

    def get_crop_to_save(self) -> tuple[int, int, int, int] | None:
        """チェックボックスが ON かつ選択範囲がある場合に crop を返す"""
        if self._chk_save.isChecked() and self._last_crop is not None:
            return self._last_crop
        return None

    # ── スロット ──────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        self._btn_reextract.setEnabled(True)

    def _on_reextract(self) -> None:
        crop = self._sel_img.get_crop()
        if crop is None:
            return
        x, y, w, h = crop
        cropped = self._src[y:y + h, x:x + w]
        if cropped.size == 0:
            return
        try:
            new_extracted, info = extract_master(cropped)
        except Exception as exc:
            QMessageBox.warning(self, "再抽出エラー", str(exc))
            return
        self._extracted = new_extracted
        self._last_crop = crop
        self._ext_view.set_cv2(new_extracted)
        self._update_info(info, new_extracted)

    def _on_clear_selection(self) -> None:
        self._sel_img.clear_selection()
        self._btn_reextract.setEnabled(False)
        self._last_crop = None
        try:
            new_extracted, info = extract_master(self._src)
            self._extracted = new_extracted
            self._ext_view.set_cv2(new_extracted)
            self._update_info(info, new_extracted)
        except Exception:
            pass

    def _update_info(self, info: dict, img: np.ndarray) -> None:
        method = info.get("method", "?")
        score = info.get("score", 0.0)
        self._info_lbl.setText(
            f"方法: {method}　|　スコア: {score:.3f}　|　"
            f"サイズ: {img.shape[1]} × {img.shape[0]} px"
        )
