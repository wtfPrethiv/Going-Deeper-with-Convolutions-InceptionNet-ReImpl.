from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
    is_best: bool = False,
    best_path: str | Path | None = None,
) -> None:
    import shutil

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)

    if is_best:
        if best_path is None:
            raise ValueError("`best_path` must be provided when `is_best=True`.")
        shutil.copyfile(path, best_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint