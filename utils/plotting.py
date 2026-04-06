from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax_loss.plot(epochs, history["val_loss"], label="Val loss", linewidth=2, linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct = [a * 100 for a in history["val_acc"]]

    ax_acc.plot(epochs, train_acc_pct, label="Train acc", linewidth=2)
    ax_acc.plot(epochs, val_acc_pct, label="Val acc", linewidth=2, linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("InceptionNet — Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)