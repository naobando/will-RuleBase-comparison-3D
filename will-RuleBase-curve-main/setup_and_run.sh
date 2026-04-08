#!/usr/bin/env bash
set -euo pipefail

# Run from project root
cd "$(dirname "$0")"

echo "[1/4] Create venv"
python3 -m venv .venv

echo "[2/4] Activate venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/4] Upgrade pip + install requirements"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Make template + inspect (sample inputs)"
python -m src.main make-template --config configs/config.yaml --name master --image data/inputs/master.png
python -m src.main inspect --config configs/config.yaml --template master --image data/inputs/test.jpg

echo "Done. Check outputs under data/outputs/."
