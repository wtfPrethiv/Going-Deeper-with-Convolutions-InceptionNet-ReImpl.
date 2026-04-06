from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _Namespace:
    def __init__(self, d: dict[str, Any]) -> None:
        for key, value in d.items():
            setattr(self, key, _Namespace(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Namespace({attrs})"


def load_config(path: str | Path = "configs/configs.yaml") -> _Namespace:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return _Namespace(raw)