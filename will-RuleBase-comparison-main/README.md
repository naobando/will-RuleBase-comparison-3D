## Environment

- Python 3.9+
- OS: Windows / macOS / Linux（動作確認はmacOS）

Setup:
macOS:
brew install libheif

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Usage:
- `master/` フォルダを作成しマスター画像を配置（デフォルト: `master/master_guide.png`）
- Googleドライブの "https://drive.google.com/drive/folders/1o3U-zg9r9h89trAFpkd1dEw3hDSHl5OT?usp=share_link" から
- 左の「カメラ接続を有効化」→「カメラ接続」
- 「比較画像を撮影」→「解析実行」

Config:
- 設定は `config.yaml` を編集
- 出力は `outputs/` 配下に保存

Guide:
- `master/` にガイド用のマスター画像を配置（デフォルト: `master/master_guide.png`）
- 画像名を変更する場合は `config.yaml` の `guide.master_image` を編集
- ガイドの角度は `guide.rotate_deg` で指定（度数）
- ライブ映像の中央にガイドとして重ねて表示

Master:
- 比較の基準画像は `master/` 配下のマスター画像を使用
- UIの「比較画像を撮影」でライブ映像側の画像を取得

Live:
- ライブ表示の回転角は `camera.live_rotate_deg` で指定（度数）

Outputs:
- `outputs/日時/` に `master.png` / `test.png` / `diff.png` / `mask.png` / `result.png` / `result.json`
