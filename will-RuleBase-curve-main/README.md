# Defect Template Project

画像テンプレートマッチングによる欠陥検出システムです。マスタ画像と検査画像を比較し、輪郭差分やROI差分を用いて欠陥を検出します。

## 機能

- **テンプレート作成**: マスタ画像からテンプレートを生成
- **欠陥検出**: 検査画像とテンプレートを比較して欠陥を検出
  - 輪郭差分検出（band_match方式）
  - ROI差分検出（エッジ差分/画素差分）
- **画像アライメント**: ECC（Enhanced Correlation Coefficient）による位置合わせ
- **スケール正規化**: サイズ差を補正
- **可視化**: 検出結果をオーバーレイ画像として出力

## セットアップ

### 必要な環境

- Python 3.7以上
- macOS / Linux

### クイックスタート

```bash
cd defect_template_project
bash setup_and_run.sh
```

出力は `data/outputs/` に保存されます。

### 手動セットアップ

```bash
# 仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 使い方

### 1. テンプレートの作成

マスタ画像からテンプレートを作成します：

```bash
python -m src.main make-template \
  --config configs/config.yaml \
  --name master \
  --image data/inputs/master.png
```

- `--config`: 設定ファイルのパス
- `--name`: テンプレート名（`templates/` 配下に保存されます）
- `--image`: マスタ画像のパス

### 2. 検査の実行

検査画像をテンプレートと比較します：

```bash
python -m src.main inspect \
  --config configs/config.yaml \
  --template master \
  --image data/inputs/test.jpg
```

- `--config`: 設定ファイルのパス
- `--template`: 使用するテンプレート名
- `--image`: 検査画像のパス

### 出力

検査結果は `data/outputs/{template_name}_{timestamp}/` に保存されます：

- `result.json`: 検出結果の詳細（判定、スコア、パラメータなど）
- `overlay_on_input_{OK|NG}.png`: 検出結果の可視化画像
- `ref_crop.png`, `test_crop.png`: クロップ後の画像
- `test_aligned.png`: アライメント後の画像
- `ref_outline.png`, `test_outline.png`: 輪郭画像
- `miss_map.png`, `extra_map.png`: 差分マップ（輪郭差分の場合）
- `roi_diff_bin.png`, `roi_boxes.png`: ROI差分結果（ROI差分が有効な場合）

## 設定ファイル

`configs/config.yaml` で各種パラメータを設定できます。

### 主要な設定項目

#### アライメント設定

- `alignment.coarse_crop`: 粗いクロップ設定
- `alignment.scale_normalize`: スケール正規化設定
- `alignment.ecc`: ECCアライメント設定

#### 検出設定

- `detection.contour_diff`: 輪郭差分検出設定
  - `method`: `band_match`（バンドマッチング方式）
  - `band_width`: バンド幅
  - `miss_ratio_th`: 欠落比率の閾値
  - `extra_ratio_th`: 余剰比率の閾値

- `detection.roi_diff`: ROI差分検出設定
  - `enabled`: 有効/無効
  - `method`: `edge`（エッジ差分）または `abs`（画素差分）
  - `rois`: ROI領域のリスト（比率で指定: [x0, y0, x1, y1]）
  - `canny1`, `canny2`: Cannyエッジ検出の閾値
  - `diff_th`: 差分の閾値
  - `min_area`: 最小ブロブ面積
  - `sum_ratio_th`: 差分比率の閾値

#### 可視化設定

- `visualization.save_intermediate`: 中間画像の保存有無
- `visualization.overlay_base`: オーバーレイのベース画像（`ref`, `test_aligned`, `test_crop`）

詳細は `configs/config.yaml` を参照してください。

## プロジェクト構成

```
defect_template_project/
├── configs/              # 設定ファイル
│   └── config.yaml
├── data/
│   ├── inputs/          # 入力画像
│   └── outputs/         # 出力結果
├── templates/           # テンプレート（自動生成）
├── src/
│   ├── main.py         # メイン処理
│   ├── pipeline.py     # 画像処理パイプライン
│   └── config_loader.py # 設定読み込み
├── requirements.txt     # 依存パッケージ
├── setup_and_run.sh    # セットアップスクリプト
└── README.md           # このファイル
```

## 依存パッケージ

- `opencv-python`: 画像処理
- `numpy`: 数値計算
- `pyyaml`: 設定ファイル読み込み

## 判定結果

`result.json` に以下の情報が記録されます：

- `judge`: 最終判定（`OK` または `NG`）
- `contour_judge`: 輪郭差分の判定
- `contour_diff`: 輪郭差分の詳細（miss_ratio, extra_ratio など）
- `roi_diff`: ROI差分の詳細（各ROIの判定結果など）
- `scale_factor`: スケール補正係数
- `ecc_cc`: ECC相関係数

## 注意事項

- マスタ画像と検査画像は同じ部品タイプである必要があります
- 照明条件や撮影角度の違いは検出精度に影響します
- 設定パラメータは対象物に応じて調整が必要です

## ライセンス

（ライセンス情報を記載してください）
