"""サンプル比較テスト"""
import cv2
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.symmetry import compare_images

# config.yaml 読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

analysis = cfg.get("analysis", {})

sample_type = os.environ.get("SAMPLE_TYPE", "平面")

# ファイルパス解決（立体サンプルはファイル名が異なる）
master_candidates = [
    f"20260407/マスター画像/{sample_type}サンプル_マスター.png",
    f"20260407/マスター画像/{sample_type}サンプル_マスター2.png",
]
normal_candidates = [
    f"20260407/正常/{sample_type}サンプル正常確定.bmp",
    f"20260407/正常/{sample_type}サンプル正常画像確定.bmp",
    f"20260407/正常/{sample_type}サンプル正常確定掃除.bmp",
]
abnormal_candidates = [
    f"20260407/異常/{sample_type}サンプル異常確定.bmp",
    f"20260407/異常/{sample_type}サンプル異常画像確定.bmp",
    f"20260407/異常/{sample_type}サンプル異常確定掃除.bmp",
]

def find_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

master_path = find_file(master_candidates)
normal_path = find_file(normal_candidates)
abnormal_path = find_file(abnormal_candidates)

if not master_path:
    print(f"ERROR: master not found for {sample_type}")
    sys.exit(1)

master = cv2.imread(master_path)
print(f"Master: {master.shape} ({master_path})")

params = {
    **analysis,
    "show_plot": False,
    "crop_to_master_fov": analysis.get("crop_to_master_fov", True),
    "use_crop_to_master_fov": analysis.get("crop_to_master_fov", True),
}

crop_fixed = cfg.get("crop_fixed_params")
if crop_fixed:
    params["crop_fixed_params"] = crop_fixed

out_dir = "20260407/結果画像"
os.makedirs(out_dir, exist_ok=True)

test_pairs = []
if abnormal_path:
    test_pairs.append(("異常", cv2.imread(abnormal_path), abnormal_path))

for label, test_img, path in test_pairs:
    print(f"\n=== {sample_type}_{label} ({path}) ===")
    print(f"  Test: {test_img.shape}")
    result = compare_images(master, test_img, **params)
    mse_val, ssim_val, diff, mask, bboxes, fig = result[:6]
    n_bbox = len(bboxes) if bboxes else 0
    print(f"  SSIM={ssim_val:.4f}, MSE={mse_val:.2f}, BBOX={n_bbox}")

    if fig is not None:
        from datetime import datetime
        version = os.environ.get("VERSION", "latest")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{sample_type}_{label}_{version}_{ts}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)

print("\nDone.")
