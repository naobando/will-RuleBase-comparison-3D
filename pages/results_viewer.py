"""テスト結果一覧ビューア"""
import glob
import os

import cv2
from src.utils.image_utils import safe_imread
import numpy as np
import streamlit as st
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, "test_6patterns_output")
PATTERNS_YAML = os.path.join(BASE, "test_patterns.yaml")


@st.cache_data
def load_patterns():
    with open(PATTERNS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["patterns"]


def find_latest_dir(pattern_name):
    pat_dir = os.path.join(OUTPUT_DIR, pattern_name)
    if not os.path.isdir(pat_dir):
        return None
    subdirs = sorted(glob.glob(os.path.join(pat_dir, "2*")))
    return subdirs[-1] if subdirs else None


def load_image_rgb(path):
    if path is None or not os.path.exists(path):
        return None
    img = safe_imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="テスト結果一覧", layout="wide")
st.title("テスト結果一覧")

patterns = load_patterns()

# フィルタ
col1, col2 = st.columns(2)
with col1:
    prefix = st.selectbox("パターン種別", ["全て", "K (筐体)", "P (トリミング)"])
with col2:
    expected_filter = st.selectbox("期待値", ["全て", "normal", "anomaly"])

filtered = patterns
if prefix == "K (筐体)":
    filtered = [p for p in filtered if p["name"].startswith("K")]
elif prefix == "P (トリミング)":
    filtered = [p for p in filtered if p["name"].startswith("P")]
if expected_filter != "全て":
    filtered = [p for p in filtered if p["expected"] == expected_filter]

# 表示モード
view_mode = st.radio("表示", ["結果のみ", "マスター＋テスト＋結果"], horizontal=True)

st.divider()

for pat in filtered:
    name = pat["name"]
    expected = pat["expected"]
    latest = find_latest_dir(name)

    if latest is None:
        continue

    bbox_img = load_image_rgb(os.path.join(latest, "12_bbox_B.png"))
    if bbox_img is None:
        continue

    # BBOX数をカウント（result.jsonがあれば読む）
    result_json = os.path.join(latest, "result.json")
    bbox_count = "?"
    status = ""
    if os.path.exists(result_json):
        import json
        with open(result_json, "r") as f:
            res = json.load(f)
        bbox_count = res.get("bbox_count", res.get("num_bboxes", "?"))
        structure = res.get("structure_judgment", "")
        if structure:
            status = f" | {structure}"

    badge_color = "green" if expected == "normal" else "red"
    badge = f":{badge_color}[{expected}]"

    st.subheader(f"{name}  {badge}  — BBOX: {bbox_count}{status}")

    if view_mode == "結果のみ":
        st.image(bbox_img, use_container_width=True)
    else:
        master = load_image_rgb(os.path.join(BASE, pat["master"]))
        test_img = load_image_rgb(os.path.join(BASE, pat["test"]))

        cols = st.columns(3)
        with cols[0]:
            st.caption("マスター")
            if master is not None:
                st.image(master, use_container_width=True)
            else:
                st.warning("画像なし")
        with cols[1]:
            st.caption("テスト")
            if test_img is not None:
                st.image(test_img, use_container_width=True)
            else:
                st.warning("画像なし")
        with cols[2]:
            st.caption("検出結果")
            st.image(bbox_img, use_container_width=True)

    st.divider()
