"""デモ用資料画像を生成"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.core.master_registration import chromakey_crop

out_dir = "20260407/デモ資料"
os.makedirs(out_dir, exist_ok=True)


def draw_label(img, text, pos=(10, 30), scale=0.8, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# === 1. クロマキー切り出しの可視化（全サンプル） ===
print("=== クロマキー切り出し可視化 ===")

samples = [
    ("平面", "20260407/正常/平面サンプル正常確定.bmp"),
    ("立体", "20260407/正常/立体サンプル正常画像確定.bmp"),
    ("線対称", "20260407/正常/線対称サンプル正常確定.bmp"),
    ("A", "20260407/正常/Aサンプル正常確定.bmp"),
    ("B", "20260407/正常/Bサンプル正常確定.bmp"),
]

for name, path in samples:
    img = cv2.imread(path)
    if img is None:
        continue

    # クロマキー切り出し
    cropped, info = chromakey_crop(img)
    bbox = info.get("bbox", (0, 0, 0, 0))
    center = info.get("fg_center", (0, 0))

    # 元画像に切り出し範囲を描画
    vis = img.copy()
    bx, by, bw, bh = bbox
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
    cv2.circle(vis, center, 10, (0, 0, 255), -1)
    draw_label(vis, f"Raw {img.shape[1]}x{img.shape[0]}", (10, 30))
    draw_label(vis, f"Crop: ({bx},{by}) {bw}x{bh}", (10, 60), scale=0.6, color=(0, 255, 0))

    # 切り出し結果
    cropped_vis = cropped.copy()
    draw_label(cropped_vis, f"Cropped {cropped.shape[1]}x{cropped.shape[0]}", (10, 30))

    # 横に並べる
    h1, w1 = vis.shape[:2]
    h2, w2 = cropped_vis.shape[:2]
    target_h = max(h1, h2)
    if h1 != target_h:
        vis = cv2.resize(vis, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        cropped_vis = cv2.resize(cropped_vis, (int(w2 * target_h / h2), target_h))

    # 矢印用の隙間
    arrow_w = 60
    arrow = np.zeros((target_h, arrow_w, 3), dtype=np.uint8)
    mid_y = target_h // 2
    cv2.arrowedLine(arrow, (5, mid_y), (arrow_w - 5, mid_y), (255, 255, 255), 3, tipLength=0.4)

    combined = np.hstack([vis, arrow, cropped_vis])
    out_path = f"{out_dir}/chromakey_{name}.png"
    cv2.imwrite(out_path, combined)
    print(f"  {name}: {out_path}")


# === 2. 処理フロー可視化（平面サンプルで代表） ===
print("\n=== 処理フロー可視化 ===")

import yaml
from src.pipeline.symmetry import compare_images

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
analysis = cfg["analysis"]
crop_fixed = cfg.get("crop_fixed_params")

master = cv2.imread("20260407/マスター画像/平面サンプル_マスター.png")
abnormal = cv2.imread("20260407/異常/平面サンプル異常確定.bmp")

params = {**analysis, "show_plot": False, "use_crop_to_master_fov": True}
if crop_fixed:
    params["crop_fixed_params"] = crop_fixed
result = compare_images(master, abnormal, **params)

# デバッグ画像から各段階を取得
dbg_dirs = sorted([d for d in os.listdir("debug_pipeline") if d.startswith("2026")])
dbg = f"debug_pipeline/{dbg_dirs[-1]}"

stages = [
    ("01_initial_A.png", "1. Master"),
    ("01_initial_B.png", "1. Input (raw)"),
    ("02_after_crop_B.png", "2. After Crop"),
    ("04_after_auto_align_B.png", "3. After Align"),
    ("08_diff.png", "4. Diff"),
    ("11_mask.png", "5. BBOX Mask"),
]

stage_imgs = []
for fname, label in stages:
    fpath = f"{dbg}/{fname}"
    if not os.path.exists(fpath):
        continue
    img = cv2.imread(fpath)
    if img is None:
        continue
    # リサイズして統一
    target_h = 400
    scale = target_h / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), target_h))
    draw_label(img, label, (5, 25), scale=0.5)
    stage_imgs.append(img)

if stage_imgs:
    # 矢印を挟んで横に並べる
    arrow_w = 30
    parts = []
    for i, img in enumerate(stage_imgs):
        parts.append(img)
        if i < len(stage_imgs) - 1:
            arrow = np.zeros((img.shape[0], arrow_w, 3), dtype=np.uint8)
            mid_y = img.shape[0] // 2
            cv2.arrowedLine(arrow, (3, mid_y), (arrow_w - 3, mid_y), (255, 255, 255), 2, tipLength=0.5)
            parts.append(arrow)
    flow = np.hstack(parts)
    out_path = f"{out_dir}/pipeline_flow.png"
    cv2.imwrite(out_path, flow)
    print(f"  {out_path}")

print("\nDone.")
