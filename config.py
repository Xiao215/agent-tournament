from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
CONFIG_DIR = PROJECT_DIR / "configs"
MODEL_WEIGHTS_DIR = Path("/h") / os.getenv("USER") / "model-weights"
CACHE_DIR = PROJECT_DIR / "caches"
