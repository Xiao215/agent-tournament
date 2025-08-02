from typing import Any
from datetime import datetime

import wandb
from matplotlib.figure import Figure


class WandBLogger:
    """Minimal wrapper so your plotting code never touches wandb directly."""

    def __init__(self, project: str, config: dict[str, Any] | None = None):
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            save_code=True,
            reinit=True
        )

    def log_figure(self, fig: Figure, name: str):
        """Log a Matplotlib figure as an image artefact."""
        self.run.log({name: wandb.Image(fig)})
