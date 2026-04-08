# defect_template_project (verified bundle)

## Quick start (macOS / Linux)

1) Unzip this project
2) Run:

```bash
cd defect_template_project
bash setup_and_run.sh
```

Outputs will be written under `data/outputs/`.

## Manual run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m src.main make-template --config configs/config.yaml --name master --image data/inputs/master.png
python -m src.main inspect --config configs/config.yaml --template master --image data/inputs/test.jpg
```
