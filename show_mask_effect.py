"""線対称異常: 前景マスクANDの効果を可視化"""
import cv2
import numpy as np
import yaml
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(__file__))

# 日本語フォント設定
for fp in ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
           "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        prop = font_manager.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
        break

from src.core.segmentation import get_foreground_mask
from src.core.preprocessing import preprocess_image

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
analysis = cfg.get("analysis", {})

master = cv2.imread("20260407/マスター画像/線対称サンプル_マスター.png")
abnormal = cv2.imread("20260407/異常/線対称サンプル異常確定.bmp")

# crop_fixed_params で切り出し（パイプラインと同じロジック）
gray_b = cv2.cvtColor(abnormal, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_b, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=3)
binary = cv2.dilate(binary, kern, iterations=1)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
main_c = max(contours, key=cv2.contourArea)
bx, by, bw, bh = cv2.boundingRect(main_c)
tw, th = master.shape[1], master.shape[0]
cx, cy = bx + bw // 2, by + bh // 2
x = max(0, min(cx - tw // 2, abnormal.shape[1] - tw))
y = max(0, min(cy - th // 2, abnormal.shape[0] - th))
cropped = abnormal[y:y+th, x:x+tw]
imageB = cv2.resize(cropped, (tw, th))
imageA = master

# alignment (bbox-based)
from src.core.alignment import align_bbox_based, align_ecc_refine
aligned, *_ = align_bbox_based(imageA, imageB)
imageB = aligned

# サイズ揃え
if imageB.shape[:2] != imageA.shape[:2]:
    imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

# 前景マスク生成
fg_a = get_foreground_mask(imageA, blur_ksize=5, keep_largest_ratio=0.02, preclose_kernel=15)
fg_b = get_foreground_mask(imageB, blur_ksize=5, keep_largest_ratio=0.02, preclose_kernel=15)
fg_and = cv2.bitwise_and(fg_a, fg_b) if fg_a is not None and fg_b is not None else fg_a

# 前処理 + 差分
procA = preprocess_image(imageA, mode="luminance", blur_ksize=3)
procB = preprocess_image(imageB, mode="luminance", blur_ksize=3)
diff_raw = cv2.absdiff(procA, procB)

# マスクなしの差分
diff_master_only = cv2.bitwise_and(diff_raw, diff_raw, mask=fg_a) if fg_a is not None else diff_raw
# ANDマスクの差分
diff_and = cv2.bitwise_and(diff_raw, diff_raw, mask=fg_and) if fg_and is not None else diff_raw

# 可視化
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("線対称 異常: 前景マスクANDの効果", fontsize=16, fontweight="bold")

def show(ax, img, title, cmap=None):
    if img is None:
        ax.set_title(title)
        ax.axis("off")
        return
    if len(img.shape) == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=12)
    ax.axis("off")

# Row 1: 入力と前景マスク
show(axes[0, 0], imageA, "Master (A)")
show(axes[0, 1], imageB, "Compare (B)")
show(axes[0, 2], fg_a, "FG Mask: Master のみ")
show(axes[0, 3], fg_and, "FG Mask: AND (共通部分)")

# Row 2: 差分の比較
show(axes[1, 0], diff_raw, "差分 (マスクなし)", cmap="hot")
show(axes[1, 1], diff_master_only, "差分 (Master FG のみ)", cmap="hot")
show(axes[1, 2], diff_and, "差分 (AND マスク)", cmap="hot")

# XOR部分を可視化（ANDで除外された領域）
if fg_a is not None and fg_and is not None:
    excluded = cv2.bitwise_xor(fg_a, fg_and)
    excluded_color = np.zeros((*excluded.shape, 3), dtype=np.uint8)
    excluded_color[excluded > 0] = [0, 0, 255]  # 赤で表示
    # imageA に重ねる
    overlay = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB).copy() if len(imageA.shape) == 3 else cv2.cvtColor(imageA, cv2.COLOR_GRAY2RGB).copy()
    overlay[excluded > 0] = overlay[excluded > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[1, 3].imshow(overlay.astype(np.uint8))
    axes[1, 3].set_title("除外された領域 (赤)", fontsize=12)
    axes[1, 3].axis("off")
else:
    axes[1, 3].axis("off")

plt.tight_layout()
out = "20260407/結果画像/線対称_異常_mask_effect.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

# 数値比較
print(f"\n差分合計値:")
print(f"  マスクなし:      {np.sum(diff_raw):,}")
print(f"  Master FGのみ:   {np.sum(diff_master_only):,}")
print(f"  AND マスク:      {np.sum(diff_and):,}")
if fg_a is not None and fg_and is not None:
    excluded_pixels = np.count_nonzero(cv2.bitwise_xor(fg_a, fg_and))
    print(f"  除外ピクセル数:  {excluded_pixels:,} ({excluded_pixels / fg_a.size * 100:.1f}%)")
