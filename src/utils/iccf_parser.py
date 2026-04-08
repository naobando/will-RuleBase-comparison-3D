"""
ICCFパーサーモジュール

TIS (The Imaging Source) カメラの設定ファイル (.iccf) を読み込み、
カメラパラメータを辞書として返す。

ICCFファイルはXML形式。device要素のうち name 属性が空でないものが
実カメラの設定を持つ。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ICCFDeviceConfig:
    """ICCFから読み取ったカメラデバイス設定"""

    name: str = ""
    serial: str = ""  # unique タグの値
    video_format: str = ""  # 例: "RGB32 (1920x1080)"
    resolution: tuple[int, int] = (0, 0)  # (width, height)
    fps: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    rotation: int = 0
    offset_x: int = 0
    offset_y: int = 0
    # カメラプロパティ (name -> value)
    properties: dict[str, Any] = field(default_factory=dict)
    # 保存設定
    save_directory: str = ""
    save_prefix: str = ""
    image_format: str = "BMP"  # 0=BMP, 1=JPEG, 2=PNG, 3=TIFF


# ImageSequenceFileType の数値マッピング
_FILE_TYPE_MAP = {0: "BMP", 1: "JPEG", 2: "PNG", 3: "TIFF"}


def parse_iccf(path: str | Path) -> list[ICCFDeviceConfig]:
    """
    ICCFファイルを読み込み、デバイス設定のリストを返す。

    Args:
        path: .iccf ファイルのパス

    Returns:
        ICCFDeviceConfig のリスト（実カメラのみ、name が空のダミーは除外）
    """
    tree = ET.parse(str(path))
    root = tree.getroot()

    configs = []
    for device in root.findall(".//device"):
        name = device.get("name", "")
        unique = _text(device, "unique")

        # name も unique も空のデバイスはダミースロット → スキップ
        if not name and not unique:
            continue

        cfg = ICCFDeviceConfig()
        cfg.name = name
        cfg.serial = unique

        # 映像フォーマット
        vf_elem = device.find("videoformat")
        if vf_elem is not None and vf_elem.text:
            cfg.video_format = vf_elem.text.strip()
            cfg.resolution = _parse_resolution(cfg.video_format)
            cfg.offset_x = int(vf_elem.get("offsetX", "0"))
            cfg.offset_y = int(vf_elem.get("offsetY", "0"))

        cfg.fps = float(_text(device, "fps") or "0")
        cfg.flip_h = _text(device, "fliph") == "1"
        cfg.flip_v = _text(device, "flipv") == "1"
        cfg.rotation = int(_text(device, "rotation") or "0")

        # VCD プロパティ（カメラ固有の露出・ゲイン等）
        cfg.properties = _parse_vcd_properties(device)

        # 保存設定
        seq = device.find(".//Sequencer")
        if seq is not None:
            ft = int(_text(seq, "ImageSequenceFileType") or "0")
            cfg.image_format = _FILE_TYPE_MAP.get(ft, "BMP")
            fng = seq.find("FileNameGen")
            if fng is not None:
                cfg.save_directory = _text(fng, "directory")
                cfg.save_prefix = _text(fng, "prefix")

        configs.append(cfg)

    return configs


def parse_iccf_first(path: str | Path) -> ICCFDeviceConfig | None:
    """ICCFファイルから最初の実カメラ設定を返す。見つからなければ None。"""
    configs = parse_iccf(path)
    return configs[0] if configs else None


def iccf_to_dict(cfg: ICCFDeviceConfig) -> dict[str, Any]:
    """ICCFDeviceConfig を config.yaml 互換の辞書に変換する。"""
    return {
        "camera_name": cfg.name,
        "serial": cfg.serial,
        "video_format": cfg.video_format,
        "resolution_width": cfg.resolution[0],
        "resolution_height": cfg.resolution[1],
        "fps": cfg.fps,
        "flip_h": cfg.flip_h,
        "flip_v": cfg.flip_v,
        "rotation": cfg.rotation,
        "offset_x": cfg.offset_x,
        "offset_y": cfg.offset_y,
        "image_format": cfg.image_format,
        "save_directory": cfg.save_directory,
        "save_prefix": cfg.save_prefix,
        "properties": cfg.properties,
    }


# --- 内部ヘルパー ---


def _text(parent: ET.Element, tag: str) -> str:
    """子要素のテキストを取得。なければ空文字列。"""
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""


def _parse_resolution(video_format: str) -> tuple[int, int]:
    """
    "RGB32 (1920x1080)" → (1920, 1080)
    "Y800 (640x480)"    → (640, 480)
    """
    import re

    m = re.search(r"\((\d+)x(\d+)\)", video_format)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def _parse_vcd_properties(device: ET.Element) -> dict[str, Any]:
    """
    VCD プロパティを {name: value} の辞書に変換する。

    構造:
      <item name="Exposure">
        <element name="Value">
          <itf value="0.053721" />
        </element>
        <element name="Auto">
          <itf value="1" />
        </element>
      </item>

    → {"Exposure": 0.053721, "Exposure_Auto": True, ...}
    """
    props: dict[str, Any] = {}
    for item in device.findall(".//vcdpropertyitems/item"):
        item_name = item.get("name", "")
        if not item_name:
            continue

        for element in item.findall("element"):
            elem_name = element.get("name", "")
            itf = element.find("itf")
            if itf is None:
                continue

            raw_value = itf.get("value", "")
            value = _coerce_value(raw_value)

            if elem_name in ("Value", "Open"):
                props[item_name] = value
            elif elem_name == "Auto":
                props[f"{item_name}_Auto"] = bool(int(float(raw_value)))
            elif elem_name == "Enable":
                props[f"{item_name}_Enable"] = bool(int(float(raw_value)))
            else:
                # White Balance Red, Auto Reference 等
                key = f"{item_name}_{elem_name}".replace(" ", "_")
                props[key] = value

    return props


def _coerce_value(raw: str) -> int | float | str:
    """文字列を int / float / str に変換。"""
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
