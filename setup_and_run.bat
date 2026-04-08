@echo off
chcp 65001 >nul
echo ========================================
echo  板金画像比較分析ツール - Windows セットアップ
echo ========================================
echo.

REM Python確認
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Python が見つかりません。Python 3.11以上をインストールしてください。
    pause
    exit /b 1
)

REM uv確認・インストール
uv --version >nul 2>&1
if errorlevel 1 (
    echo [情報] uv をインストールしています...
    pip install uv
)

REM 依存パッケージインストール
echo.
echo [1/2] 依存パッケージをインストール中...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo [エラー] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

REM 起動
echo.
echo [2/2] アプリケーションを起動します...
echo.
uv run python app_qt.py

pause
