# パラメータリファレンス

## config.yaml 主要パラメータ

### 画角合わせ (crop)

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| crop_to_master_fov | true | 画角合わせを有効化 |
| crop_fixed_params | {scale, rotation, x, y} | 設定されていると chromakey_crop 方式を使用 |
| crop_compare_mode | true | 切り出し前後のSSIMを比較して採用判定 |

### 位置合わせ (alignment)

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| auto_align | true | 自動位置合わせ有効 |
| align_mode | bbox | 外接矩形ベースの位置合わせ |
| align_ecc_refine_enabled | true | ECC精密合わせ有効 |
| align_ecc_warp_mode | euclidean | 3自由度（回転+平行移動のみ） |
| align_min_rotation | 3.0 | 補正を適用する最小回転角度(度) |
| align_min_translation | 30 | 補正を適用する最小平行移動量(px) |

### BBOX検出

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| bbox_method | ensemble_sift | E+SIFT ensemble 方式 |
| ensemble_thresh | 190 | ensemble 二値化閾値 |
| ensemble_min_area | 200 | ensemble最小面積(固定下限) |
| ensemble_min_area_ratio | 0.0003 | ensemble最小面積(画像面積比) |
| ensemble_min_dim | 6 | 最小幅/高さ(px) |
| ensemble_bbox_min_mean | 60 | BBOX内ensemble平均の最低値 |
| ensemble_bg_brightness_max | 30 | 背景判定の最大輝度（AND条件） |
| diff_thresh | 38 | 差分閾値 |
| min_area | 850 | BBOX最小面積(auto計算で上書き) |
| morph_kernel | 9 | モルフォロジーカーネルサイズ(auto計算) |
| max_boxes | 10 | BBOX最大数 |
| bbox_merge_distance | 40 | BBOXマージ距離(ratio自動計算で上書き) |
| bbox_merge_distance_ratio | 0.03 | マージ距離(短辺の3%) |
| bbox_drop_band_aspect | 4.0 | 帯状BBOX除去のアスペクト比閾値 |

### 前景マスク

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| foreground_mask | true | 前景マスク有効 |
| foreground_mask_dilate_iter | 2 | マスク膨張回数 |
| foreground_mask_kernel | 15 | マスクカーネルサイズ |

### 自動計算

| パラメータ | 計算式 | 説明 |
|-----------|--------|------|
| min_area | 画像面積 × 0.008 | 画像サイズに応じた最小面積 |
| morph_kernel | sqrt(面積) / coeff | 画像面積に応じたカーネル |
| bbox_merge_distance | 短辺 × 0.03 | 短辺の3%でマージ距離 |
| ensemble_min_area | max(200, 画像面積 × 0.0003) | 大画像では ratio でスケール |

## デバッグUIスライダー

| スライダー | 範囲 | 説明 |
|-----------|------|------|
| diff_thresh | 1-50 | 差分閾値 |
| min_area | 10-5000 | BBOX最小面積 |
| morph_kernel | 3-51 (奇数) | モルフォロジーカーネル |
| max_boxes | 1-10 | BBOX最大数 |
| ensemble_thresh | 10-255 | ensemble閾値 |
| bbox_merge_dist | 0-200 | BBOXマージ距離 |
