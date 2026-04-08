import os

import yaml


DEFAULT_CONFIG = {
    "ui": {
        "title": "板金画像比較分析ツール（連係カメラ対応）",
        "description": "iPhoneを連係カメラとして接続し、撮影→比較を行います。",
    },
    "camera": {
        "device_index": 0,
        "live_refresh_ms": 200,
        "live_rotate_deg": 90,
    },
    "guide": {
        "enabled": True,
        "master_dir": "master",
        "master_image": "master_guide.png",
        "rotate_deg": 0,
        "scale": 0.7,
        "alpha": 0.35,
    },
    "analysis": {
        "diff_thresh": 12,
        "min_area": 300,
        "morph_kernel": 20,
        "max_boxes": 6,
    },
    "output": {
        "base_dir": "outputs",
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path="config.yaml"):
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(config_path):
        return config
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_update(config, loaded)
