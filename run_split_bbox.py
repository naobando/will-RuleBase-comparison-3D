"""差分を方向別に分離してBBOX生成→合成するテスト"""
import cv2
import yaml
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.symmetry import SymmetryPipeline, compare_images
from src.core.bbox_detection import merge_nearby_bboxes

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
analysis = cfg["analysis"]
crop_fixed = cfg.get("crop_fixed_params")

sample_type = os.environ.get("SAMPLE_TYPE", "立体")
version = os.environ.get("VERSION", "v14split")

# ファイル検索
master_candidates = [
    f"20260407/マスター画像/{sample_type}サンプル_マスター.png",
    f"20260407/マスター画像/{sample_type}サンプル_マスター2.png",
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
abnormal_path = find_file(abnormal_candidates)
if not master_path or not abnormal_path:
    print(f"ERROR: files not found for {sample_type}")
    sys.exit(1)

master = cv2.imread(master_path)
abnormal = cv2.imread(abnormal_path)

# 通常のパイプラインを実行（デバッグ画像を保存させるため）
params = {**analysis, "show_plot": False, "use_crop_to_master_fov": True}
if crop_fixed:
    params["crop_fixed_params"] = crop_fixed
result_normal = compare_images(master, abnormal, **params)
normal_bboxes = result_normal[4]

# デバッグ画像から位置合わせ済み画像とensembleマスクを取得
dbg_dirs = sorted([d for d in os.listdir("debug_pipeline") if d.startswith("2026")])
dbg = f"debug_pipeline/{dbg_dirs[-1]}"
imgA = cv2.imread(f"{dbg}/05_after_size_align2_A.png")
imgB = cv2.imread(f"{dbg}/05_after_size_align2_B.png")
emask = cv2.imread(f"{dbg}/11_mask.png", cv2.IMREAD_GRAYSCALE)

if emask is None:
    print("ERROR: ensemble mask not found")
    sys.exit(1)

# グレースケール差分の方向
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY).astype(np.float32)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY).astype(np.float32)
diff_signed = grayA - grayB  # 正=master明るい=欠損, 負=test明るい=余剰

# ensemble maskの connected component ごとに方向判定して分離
n, labels, stats, _ = cv2.connectedComponentsWithStats(emask, connectivity=8)

missing_mask = np.zeros_like(emask)  # マスターにあって入力にない
extra_mask = np.zeros_like(emask)    # 入力にあってマスターにない

emin_area = max(200, int(imgA.shape[0] * imgA.shape[1] * 0.0003))

for i in range(1, n):
    bx, by, bw, bh, area = stats[i]
    if area < emin_area:
        continue
    component = (labels == i)
    vals = diff_signed[component]
    mean_dir = float(np.mean(vals))
    if mean_dir > 0:
        missing_mask[component] = 255
    else:
        extra_mask[component] = 255

# 帯フィルタ用のアスペクト比閾値
band_aspect = float(analysis.get("bbox_drop_band_aspect", 4.0))
merge_dist = int(analysis.get("bbox_merge_distance", 40))

def extract_bboxes(mask, label):
    """マスクからBBOXを生成（帯フィルタ+マージ）"""
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    for i in range(1, n2):
        bx, by, bw, bh, area = stats2[i]
        if area < emin_area:
            continue
        # 帯フィルタ
        aspect = max(bw, 1) / max(bh, 1)
        aspect = max(aspect, max(bh, 1) / max(bw, 1))
        if band_aspect > 0 and aspect >= band_aspect:
            continue
        bboxes.append((bx, by, bw, bh))
    # マージ（同じカテゴリ内のみ）
    if merge_dist > 0 and len(bboxes) > 1:
        bboxes = merge_nearby_bboxes(bboxes, distance_thresh=merge_dist, diff_mask=mask)
    return bboxes

missing_bboxes = extract_bboxes(missing_mask, "欠損")
extra_bboxes = extract_bboxes(extra_mask, "余剰")

# 合成
all_bboxes = missing_bboxes + extra_bboxes

h, w = imgA.shape[:2]
def pos(bx, by, bw, bh):
    lr = "L" if (bx + bw // 2) < w // 2 else "R"
    cy = by + bh // 2
    if cy < h * 0.2: return f"TOP-{lr}"
    elif cy < h * 0.4: return f"UPPER-{lr}"
    elif cy < h * 0.6: return f"MID-{lr}"
    elif cy < h * 0.75: return f"LOWER-{lr}"
    else: return f"BOTTOM-{lr}"

print(f"\n=== {sample_type} 方向別分離BBOX ===")
print(f"欠損(masterにあり入力にない): {len(missing_bboxes)}個")
for i, (bx, by, bw, bh) in enumerate(missing_bboxes):
    print(f"  M#{i}: ({bx},{by}) {bw}x{bh} area={bw*bh} [{pos(bx,by,bw,bh)}]")
print(f"余剰(入力にありmasterにない): {len(extra_bboxes)}個")
for i, (bx, by, bw, bh) in enumerate(extra_bboxes):
    print(f"  E#{i}: ({bx},{by}) {bw}x{bh} area={bw*bh} [{pos(bx,by,bw,bh)}]")
print(f"合成BBOX: {len(all_bboxes)}個")
print(f"通常パイプライン(v14): {len(normal_bboxes)}個")
