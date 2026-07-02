import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def dataset_stats(data_dir: Path):
    png_files = sorted(data_dir.glob("location_*.png"))
    countries = set()
    for png_path in png_files:
        json_path = png_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            countries.add(meta.get("country_name", "Unknown"))
    print(f"Images: {len(png_files)}")
    print(f"Countries: {len(countries)}")
    print(f"Countries list: {sorted(countries)}")
    return len(png_files), len(countries)
