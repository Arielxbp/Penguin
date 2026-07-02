import json
import random
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from config import (
    AUGMENTATIONS_PER_IMAGE,
    AUGMENTATION_CONFIG,
    DATA_DIR,
    DATA_MAPPED_DIR,
    NUM_IMAGES_MAX,
    SEED,
    SUBSET_DIR,
    USE_SUBSET,
)

random.seed(SEED)
torch.manual_seed(SEED)


def gather_samples(data_dir: Path, limit: Optional[int] = None):
    pngs = sorted(data_dir.glob("location_*.png"))
    samples = []
    for png_path in pngs:
        json_path = png_path.with_suffix(".json")
        if json_path.exists():
            samples.append((png_path, json_path))
    if limit:
        samples = samples[:limit]
    return samples


class GeoSamples:
    def __init__(self, data_dir: Optional[Path] = None):
        data_dir = data_dir or SUBSET_DIR
        self.samples = gather_samples(data_dir, NUM_IMAGES_MAX)

    def __len__(self):
        return len(self.samples)


class CountryEncoder:
    def __init__(self, data_dir: Optional[Path] = None):
        data_dir = data_dir or SUBSET_DIR
        countries = set()
        for json_path in sorted(data_dir.glob("location_*.json")):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    countries.add(meta.get("country_name", "Unknown"))
            except (json.JSONDecodeError, Exception):
                countries.add("Unknown")
        self.country_list = sorted(countries)
        self.country_to_idx = {c: i for i, c in enumerate(self.country_list)}

    def encode(self, country_name: str) -> int:
        return self.country_to_idx.get(country_name, -1)

    def __len__(self):
        return len(self.country_list)


def parse_metadata(json_path: Path):
    try:
        with open(json_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            meta = {}
    except (json.JSONDecodeError, Exception):
        meta = {}
    return {
        "coordinates": meta.get("coordinates", [0.0, 0.0]),
        "country_name": meta.get("country_name", "Unknown"),
        "country_code": meta.get("country_code", "XX"),
        "regions": meta.get("regions", []),
    }


PIL_AUG_TRANSFORM = T.Compose(
    [
        T.RandomResizedCrop(
            336,
            scale=AUGMENTATION_CONFIG["random_crop_scale"],
            ratio=AUGMENTATION_CONFIG["random_crop_ratio"],
        ),
        T.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip_prob"]),
        T.ColorJitter(
            brightness=AUGMENTATION_CONFIG["color_jitter_brightness"],
            contrast=AUGMENTATION_CONFIG["color_jitter_contrast"],
            saturation=AUGMENTATION_CONFIG["color_jitter_saturation"],
            hue=AUGMENTATION_CONFIG["color_jitter_hue"],
        ),
        T.RandomRotation(degrees=AUGMENTATION_CONFIG["rotation_degrees"]),
    ]
)

TENSOR_AUG_TRANSFORM = T.Compose(
    [
        T.GaussianBlur(
            kernel_size=AUGMENTATION_CONFIG["blur_kernel_size"],
            sigma=AUGMENTATION_CONFIG["blur_sigma"],
        ),
    ]
)

BASE_TRANSFORM = T.Compose(
    [
        T.Resize((336, 336)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


class GeoSampleDataset(Dataset):
    def __init__(
        self,
        samples: list,
        country_encoder: CountryEncoder,
        augment: bool = False,
    ):
        self.samples = samples
        self.country_encoder = country_encoder
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        meta = parse_metadata(json_path)
        image = Image.open(img_path).convert("RGB")
        country_idx = self.country_encoder.encode(meta["country_name"])
        augmented_images = []
        if self.augment:
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                aug_img = PIL_AUG_TRANSFORM(image)
                aug_tensor = T.ToTensor()(aug_img)
                aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
                aug_tensor = T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )(aug_tensor)
                augmented_images.append(aug_tensor)
        image = BASE_TRANSFORM(image)
        result = {
            "image": image,
            "image_path": str(img_path),
            "json_path": str(json_path),
            "coordinates": torch.tensor(meta["coordinates"], dtype=torch.float32),
            "country_name": meta["country_name"],
            "country_code": meta["country_code"],
            "country_idx": torch.tensor(country_idx, dtype=torch.long),
        }
        if augmented_images:
            result["augmented_images"] = torch.stack(augmented_images, dim=0)
        return result


def create_dataset(data_dir: Optional[Path] = None, augment: bool = False):
    data_dir = data_dir or SUBSET_DIR
    samples = gather_samples(data_dir, NUM_IMAGES_MAX)
    country_encoder = CountryEncoder(data_dir)
    return GeoSampleDataset(samples, country_encoder, augment=augment), country_encoder


def split_dataset(
    full_dataset: GeoSampleDataset,
    train_ratio: float = 0.85,
):
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]
    return train_indices, val_indices
