"""v8パイプラインのデバッグ画像から前景マスクANDの効果を可視化"""
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 日本語フォント設定
for fp in ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
           "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        prop = font_manager.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
        break

DBG = "debug_pipeline/20260408_194728"  # 線対称 異常 v8

# v8 パイプラインが実際に使った画像を読み込む
imageA = cv2.imread(f"{DBG}/05_after_size_align2_A.png")
imageB = cv2.imread(f"{DBG}/05_after_size_align2_B.png")
diff_raw = cv2.imread(f"{DBG}/08_diff.png", cv2.IMREAD_GRAYSCALE)
fg_mask_and = cv2.imread(f"{DBG}/09_fg_mask.png", cv2.IMREAD_GRAYSCALE)
diff_after_mask = cv2.imread(f"{DBG}/09_after_fg_mask.png", cv2.IMREAD_GRAYSCALE)
final_mask = cv2.imread(f"{DBG}/11_mask.png", cv2.IMREAD_GRAYSCALE)
bbox_A = cv2.imread(f"{DBG}/12_bbox_A.png")
bbox_B = cv2.imread(f"{DBG}/12_bbox_B.png")

# 個別の前景マスクも生成（比較用）
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.core.segmentation import get_foreground_mask

fg_a = get_foreground_mask(imageA, blur_ksize=5, keep_largest_ratio=0.02, preclose_kernel=15)
fg_b = get_foreground_mask(imageB, blur_ksize=5, keep_largest_ratio=0.02, preclose_kernel=15)

# Master-only でマスクした差分（旧方式の再現）
diff_master_only = cv2.bitwise_and(diff_raw, diff_raw, mask=fg_a) if fg_a is not None else diff_raw

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.suptitle("線対称 異常: v8パイプライン — 前景マスクANDの効果", fontsize=16, fontweight="bold")

def show(ax, img, title, cmap=None, vmax=None):
    if img is None:
        ax.set_title(title); ax.axis("off"); return
    if len(img.shape) == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=vmax or 255)
    ax.set_title(title, fontsize=11); ax.axis("off")

# Row 1: 入力画像とマスク比較
show(axes[0, 0], imageA, "Master (位置合わせ後)")
show(axes[0, 1], imageB, "Compare (位置合わせ後)")
show(axes[0, 2], fg_a, "FG Mask: Master のみ (旧)")
show(axes[0, 3], fg_mask_and, "FG Mask: AND 共通部分 (新)")

# Row 2: 差分の比較
show(axes[1, 0], diff_raw, "差分 (マスクなし)", cmap="hot")
show(axes[1, 1], diff_master_only, "差分 (Master FGのみ=旧)", cmap="hot")
show(axes[1, 2], diff_after_mask, "差分 (ANDマスク後=新)", cmap="hot")

# 除外された領域の可視化
if fg_a is not None and fg_mask_and is not None:
    excluded = cv2.subtract(fg_a, fg_mask_and)
    overlay = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB).copy()
    overlay[excluded > 0] = (overlay[excluded > 0].astype(float) * 0.4 + np.array([255, 50, 50]) * 0.6).astype(np.uint8)
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("赤: ANDで除外された領域", fontsize=11)
    axes[1, 3].axis("off")

# Row 3: 最終検出結果
show(axes[2, 0], final_mask, "最終 Diff Mask")
show(axes[2, 1], bbox_A, "BBOX on Master")
show(axes[2, 2], bbox_B, "BBOX on Compare")
axes[2, 3].axis("off")

# 数値情報
vals = {
    "マスクなし": int(np.sum(diff_raw)),
    "Master FGのみ": int(np.sum(diff_master_only.astype(int))),
    "AND マスク": int(np.sum(diff_after_mask.astype(int))),
}
excluded_px = int(np.count_nonzero(cv2.subtract(fg_a, fg_mask_and))) if fg_a is not None and fg_mask_and is not None else 0
txt = f"差分合計:\n"
txt += f"  マスクなし:    {vals['マスクなし']:>12,}\n"
txt += f"  Master FGのみ: {vals['Master FGのみ']:>12,}\n"
txt += f"  AND マスク:    {vals['AND マスク']:>12,}\n"
txt += f"\n除外ピクセル: {excluded_px:,} ({excluded_px / fg_a.size * 100:.1f}%)" if fg_a is not None else ""
axes[2, 3].text(0.05, 0.95, txt, transform=axes[2, 3].transAxes, fontsize=12,
               verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = "20260407/結果画像/線対称_異常_mask_effect_v8.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)
