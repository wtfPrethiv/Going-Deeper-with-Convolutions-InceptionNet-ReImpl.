from __future__ import annotations
import torch

class AverageMeter:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, count={self.count})"


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)