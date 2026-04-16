"""
download_models.py
Downloads SAM2.1 model checkpoints from Meta's servers.

Usage:
    python download_models.py                      # downloads tiny (default)
    python download_models.py --model tiny
    python download_models.py --model small
    python download_models.py --model large
    python download_models.py --model base_plus
    python download_models.py --model all
"""

import argparse
import sys
from pathlib import Path

import requests

CHECKPOINTS = {
    "tiny": (
        "sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ),
    "small": (
        "sam2.1_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ),
    "base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ),
    "large": (
        "sam2.1_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
}

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def download_checkpoint(name: str) -> None:
    filename, url = CHECKPOINTS[name]
    dest = CHECKPOINTS_DIR / filename
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[skip] {filename} already exists at {dest}")
        return

    print(f"[download] {name}: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[error] Failed to download {name}: {e}", file=sys.stderr)
        sys.exit(1)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                print(f"\r  {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    print(f"\n[done] saved to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download SAM2.1 model checkpoints")
    parser.add_argument(
        "--model",
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default="tiny",
        help="Which checkpoint to download (default: tiny)",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in CHECKPOINTS:
            download_checkpoint(name)
    else:
        download_checkpoint(args.model)


if __name__ == "__main__":
    main()
