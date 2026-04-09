# アーキテクチャ概要

## パイプライン処理フロー

```
入力画像(テスト) + マスター画像
    │
    ├─ 1. Crop (画角合わせ)
    │    crop_fixed_params あり → chromakey_crop で前景中心検出 → マスターサイズで切り出し
    │    crop_fixed_params なし → crop_to_master_fov (特徴点マッチング)
    │
    ├─ 2. サイズ合わせ
    │    マスターと同じピクセルサイズに resize
    │
    ├─ 3. 位置合わせ (auto_align)
    │    align_mode: bbox → 外接矩形ベース (estimateAffinePartial2D, 4自由度)
    │    align_mode: feature_points → 特徴点マッチング
    │    + ECC精密合わせ (euclidean, 3自由度)
    │
    ├─ 4. 前処理 (preprocess)
    │    luminance → グレースケール + ブラー
    │
    ├─ 5. 差分計算
    │    SSIM map + 絶対差分
    │
    ├─ 6. 前景マスク
    │    両画像の前景OR → 両方とも背景の領域だけ除外
    │
    ├─ 7. BBOX検出 (E+SIFT ensemble)
    │    Multi-scale SSIM ensemble → connected components
    │    → 帯フィルタ → BG CHROMA(AND条件) → edge suppress
    │
    └─ 8. 結果出力
         SSIM, MSE, BBOX一覧, 可視化図
```

## 主要モジュール

| ファイル | 役割 |
|---------|------|
| `src/pipeline/symmetry.py` | メインパイプライン (SymmetryPipeline) |
| `src/core/crop.py` | 画角合わせ (crop_to_master_fov) |
| `src/core/alignment.py` | 位置合わせ (auto_align, ECC, bbox_based) |
| `src/core/master_registration.py` | マスター登録 (chromakey_crop, extract_master) |
| `src/core/bbox_detection.py` | BBOX検出・マージ (merge_nearby_bboxes) |
| `src/core/segmentation.py` | 前景マスク生成 (get_foreground_mask) |
| `src/core/preprocessing.py` | 画像前処理 |
| `ui/debug/` | デバッグUI (スライダーパラメータ調整) |
| `ui/user/` | ユーザーUI (マスター登録・比較実行) |
| `ui/threads/analysis_worker.py` | バックグラウンド解析ワーカー |

## 変換の自由度

| 処理 | 方式 | 自由度 | 許容する変形 |
|------|------|--------|-------------|
| crop_to_master_fov | estimateAffinePartial2D | 4 | 回転+等方スケール+平行移動 |
| auto_align (bbox) | estimateAffinePartial2D | 4 | 回転+等方スケール+平行移動 |
| ECC refine | MOTION_EUCLIDEAN | 3 | 回転+平行移動 |

シアー・透視歪みは全て排除。カメラ固定環境では回転+平行移動で十分。
