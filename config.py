from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
MODEL_WEIGHTS_DIR = Path("/h") / os.getenv("USER") / "model-weights"
