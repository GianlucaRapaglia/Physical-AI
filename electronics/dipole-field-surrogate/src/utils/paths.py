from pathlib import Path

# Project root = folder containing pyproject.toml or .git
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
