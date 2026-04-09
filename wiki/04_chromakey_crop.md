# クロマキー自動切り出し (2026-04-09)

## 概要

黒背景を利用したクロマキー方式で部品を自動切り出す。
マスター登録とテスト画像cropの両方で同じ `chromakey_crop` 関数を共有。

## 処理フロー

```
入力画像（黒背景+部品）
    │
    ├─ 1. グレースケール + GaussianBlur(5,5)
    │
    ├─ 2. Otsu二値化（前景/背景分離）
    │
    ├─ 3. morphological closing (31x31, 3回) + dilate (1回)
    │    → 部品内部の穴を埋める
    │
    ├─ 4. 輪郭検出 → 画像中心に最も近い大きな輪郭を選択
    │
    ├─ 5. 外接矩形 + padding(4px) で切り出し
    │
    └─ 出力: 切り出し画像 + info {method, bbox, fg_center, score}
```

## 検証結果

### ステップ1: 手動マスター + 黒余白追加 → クロマキー再切り出し (v21)
手動切り出しマスターに150pxの黒余白を追加し、クロマキーで再切り出し。
全サンプルでオリジナルと同等のBBOX数・SSIM。

### ステップ2: 広角正常画像 → クロマキー切り出し (v22)
1568x1356の広角画像をそのままクロマキーで切り出してマスターとして使用。

| サンプル | 手動マスター | クロマキー自動 | 差 |
|----------|-------------|---------------|----|
| 平面 | BBOX=8 | BBOX=8 | 同じ |
| 立体 | BBOX=9 | BBOX=10 | +1 |
| 線対称 | BBOX=7 | BBOX=7 | 同じ |
| A | BBOX=1 | BBOX=1 | 同じ |
| B | BBOX=4 | BBOX=4 | 同じ |

### ステップ3: パイプライン統合 (v24)
`chromakey_crop` を `master_registration.py` に新設し、
パイプラインの crop ブランチでも同じ関数を使用。
全サンプルでv20と完全一致を確認。

## 自動マスター登録の方針

1. カメラ撮影 or ファイル選択（広角画像）
2. `chromakey_crop` で前景検出・外接矩形切り出し
3. 切り出し結果をマスターとして保存
4. 比較時: テスト画像も `chromakey_crop` で前景中心を検出、マスターサイズで切り出し

## `chromakey_crop` 関数

```python
from src.core.master_registration import chromakey_crop

result_image, info = chromakey_crop(image, padding=4, min_area_ratio=0.01)
# info = {
#     "method": "chromakey",
#     "bbox": (x, y, w, h),      # 切り出し矩形
#     "fg_center": (cx, cy),     # 前景中心座標
#     "score": float,            # 切り出し面積比
# }
```

## コミット履歴

- `674b694` クロマキー自動切り出しの検証完了
- `3e1e1a3` マスター登録とパイプラインcropをchromakey_cropに統一
