from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InceptionNet v1 — CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Run training or inference.",
    )
    parser.add_argument(
        "--config",
        default="configs/configs.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to use: "auto", "cuda", or "cpu".',
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="[train] Path to a checkpoint to resume training from.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="[predict] Path to a .pth checkpoint file.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="[predict] Path to an image file or directory.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="[predict] Number of top predictions to display.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "train":
        from training.train import Trainer

        trainer = Trainer(
            config_path=args.config,
            resume_from=args.resume,
            device=args.device,
        )
        trainer.fit()

    elif args.mode == "predict":
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required for predict mode.", file=sys.stderr)
            sys.exit(1)
        if args.image is None:
            print("ERROR: --image is required for predict mode.", file=sys.stderr)
            sys.exit(1)

        from pathlib import Path
        from inference.predict import Predictor

        predictor = Predictor(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
        )

        image_path = Path(args.image)

        if image_path.is_dir():
            predictor.predict_dir(image_path)
        elif args.topk > 1:
            results = predictor.predict_topk(image_path, k=args.topk)
            print(f"\nTop-{args.topk} predictions for {image_path.name}:")
            for rank, (label, conf) in enumerate(results, 1):
                print(f"  {rank}. {label:<12s} {conf:.1%}")
        else:
            label, conf = predictor.predict(image_path)
            print(f"\n{image_path.name} → {label} ({conf:.1%})")


if __name__ == "__main__":
    main()