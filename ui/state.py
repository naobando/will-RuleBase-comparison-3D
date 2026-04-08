"""アプリケーション状態管理 (st.session_state の代替)"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AppState:
    # カメラ・フレーム
    live: bool = False
    last_frame: Optional[np.ndarray] = None
    test_frame: Optional[np.ndarray] = None

    # マスター画像
    master_override: Optional[np.ndarray] = None
    master_file_key: Optional[tuple] = None
    test_file_key: Optional[tuple] = None

    # ガイド
    guide_mask: Optional[np.ndarray] = None
    guide_enabled_at_capture: bool = False

    # キャリブレーション
    calibration_result: Optional[dict] = None

    # 解析
    analysis_result: Optional[tuple] = None
    analysis_error: Optional[str] = None
    analysis_save_error: Optional[str] = None
    analysis_params: Optional[tuple] = None  # (diff_thresh, min_area, morph_kernel, max_boxes)

    # 品質チェック推奨メッセージ
    master_quality_recommend: Optional[str] = None
    test_quality_recommend: Optional[str] = None

    def reset(self) -> None:
        """リセットボタン相当: 全状態を初期値に戻す"""
        self.live = False
        self.last_frame = None
        self.test_frame = None
        self.master_override = None
        self.master_file_key = None
        self.test_file_key = None
        self.guide_mask = None
        self.guide_enabled_at_capture = False
        self.calibration_result = None
        self.analysis_result = None
        self.analysis_error = None
        self.analysis_save_error = None
        self.analysis_params = None
        self.master_quality_recommend = None
        self.test_quality_recommend = None

    def set_master(self, img: np.ndarray, key: tuple) -> None:
        self.master_override = img
        self.master_file_key = key
        self.calibration_result = None  # マスター変更でキャリブレーションリセット

    def set_test(self, img: np.ndarray, key: tuple) -> None:
        self.test_frame = img
        self.test_file_key = key
