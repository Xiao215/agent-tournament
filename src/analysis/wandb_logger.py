import wandb
from typing import Any


class WandBLogger:
    """Minimal wrapper so your plotting code never touches wandb directly."""

    def __init__(self, project: str, config: dict[str, Any] | None = None):
        self.run = wandb.init(
            project=project, config=config or {}, save_code=True, reinit=True
        )

    def log_figure(self, fig, name: str, step: int | None = None):
        """Log a Matplotlib figure as an image artefact."""
        self.run.log({name: wandb.Image(fig)}, step=step)
