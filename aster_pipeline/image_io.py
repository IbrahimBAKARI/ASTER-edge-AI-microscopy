from __future__ import annotations

from pathlib import Path

from PIL import Image


def discover_images(input_path: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    candidates = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    images = [p for p in candidates if p.suffix.lower() in extensions]
    if not images:
        raise ValueError(f"No supported images found in: {path}")
    return images


def load_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid or unreadable image: {path}") from exc
