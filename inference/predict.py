from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models import InceptionNet
from data.dataset import CIFAR10DataModule
from utils import load_config, load_checkpoint, get_logger


class Predictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "configs/configs.yaml",
        device: str = "auto",
    ) -> None:
        self.cfg = load_config(config_path)
        self.logger = get_logger("predictor")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = InceptionNet(
            num_classes=self.cfg.model.num_classes,
            aux_logits=False,
            dropout=self.cfg.model.dropout,
        ).to(self.device)

        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.cfg.data.mean,
                std=self.cfg.data.std,
            ),
        ])

        self.classes = CIFAR10DataModule.CLASSES

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        label = self.classes[pred_idx.item()]
        return label, confidence.item()

    def predict_topk(
        self, image_path: str | Path, k: int = 3
    ) -> list[tuple[str, float]]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(k, dim=1)

        results = [
            (self.classes[idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
        return results

    def predict_dir(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> list[tuple[Path, str, float]]:
        directory = Path(directory)
        image_paths = [
            p for p in sorted(directory.iterdir())
            if p.suffix.lower() in extensions
        ]

        if not image_paths:
            self.logger.warning(f"No images found in {directory}")
            return []

        results = []
        for path in image_paths:
            label, conf = self.predict(path)
            results.append((path, label, conf))
            self.logger.info(f"{path.name:<30s} → {label:<12s} ({conf:.1%})")

        return results