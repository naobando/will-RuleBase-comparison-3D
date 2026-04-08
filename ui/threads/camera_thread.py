"""カメラライブキャプチャスレッド"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.quality_check import check_image_quality


def capture_frame(device_index: int, width: int = 0, height: int = 0) -> np.ndarray:
    """単一フレームを取得する（app.py の capture_frame を移植）"""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした。デバイス番号をご確認ください。")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("フレームを取得できませんでした。")
    return frame


class CameraThread(QThread):
    """ライブ映像を連続キャプチャしてフレームをシグナルで送出"""

    frame_ready = Signal(object)    # np.ndarray
    error_occurred = Signal(str)

    def __init__(
        self,
        device_index: int = 0,
        width: int = 0,
        height: int = 0,
        interval_ms: int = 200,
        cap: cv2.VideoCapture | None = None,  # メインスレッドで開いたキャプチャを渡せる
        parent=None,
    ):
        super().__init__(parent)
        self.device_index = device_index
        self.width = width
        self.height = height
        self.interval_ms = interval_ms
        self._cap_from_main = cap   # メインスレッドで既にオープン済みの場合
        self._running = False

    def run(self) -> None:
        self._running = True

        # メインスレッドで開いたキャプチャがあればそれを使う（macOS権限対策）
        if self._cap_from_main is not None and self._cap_from_main.isOpened():
            cap = self._cap_from_main
            owns_cap = False
        else:
            # フォールバック: スレッド内でオープン
            cap = cv2.VideoCapture(self.device_index)
            owns_cap = True
            if not cap.isOpened():
                self.error_occurred.emit(
                    f"カメラ(index={self.device_index})を開けませんでした。"
                    "デバイス番号をご確認ください。"
                )
                return

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.error_occurred.emit("フレームを取得できませんでした。カメラが切断された可能性があります。")
                break
            self.frame_ready.emit(frame)
            time.sleep(self.interval_ms / 1000.0)

        if owns_cap:
            cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait()
        # メインから渡されたキャプチャはここで解放
        if self._cap_from_main is not None:
            self._cap_from_main.release()
            self._cap_from_main = None


class CaptureWorker(QThread):
    """撮影ボタン用の単発キャプチャ（品質チェック + リトライ付き）"""

    frame_ready = Signal(object)        # np.ndarray (成功時)
    quality_warn = Signal(str)          # 品質警告メッセージ
    error_occurred = Signal(str)

    def __init__(
        self,
        device_index: int = 0,
        width: int = 0,
        height: int = 0,
        quality_check: bool = True,
        max_retry: int = 3,
        retry_delay_ms: int = 300,
        white_ratio: float = 0.05,
        blur_threshold: float = 100.0,
        parent=None,
    ):
        super().__init__(parent)
        self.device_index = device_index
        self.width = width
        self.height = height
        self.quality_check = quality_check
        self.max_retry = max_retry
        self.retry_delay_ms = retry_delay_ms
        self.white_ratio = white_ratio
        self.blur_threshold = blur_threshold

    def run(self) -> None:
        last_warn = None
        attempts = self.max_retry if self.quality_check else 1
        for attempt in range(attempts):
            try:
                frame = capture_frame(self.device_index, self.width, self.height)
            except Exception as e:
                self.error_occurred.emit(str(e))
                return

            if not self.quality_check:
                self.frame_ready.emit(frame)
                return

            ok, reason = check_image_quality(
                frame,
                white_ratio_threshold=self.white_ratio,
                blur_threshold=self.blur_threshold,
            )
            if ok:
                self.frame_ready.emit(frame)
                return

            last_warn = f"品質NG: {reason}"
            if attempt < attempts - 1:
                time.sleep(self.retry_delay_ms / 1000.0)

        # リトライ上限に達した場合: 最後のフレームを採用しつつ警告
        if last_warn:
            self.quality_warn.emit(last_warn)
        try:
            frame = capture_frame(self.device_index, self.width, self.height)
            self.frame_ready.emit(frame)
        except Exception as e:
            self.error_occurred.emit(str(e))
