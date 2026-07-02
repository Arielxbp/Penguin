import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    OBJ_FEATURE_DIM,
    VEG_FEATURE_DIM,
    OUTPUT_DIR,
)
from dataset import (
    CountryEncoder,
    gather_samples,
    parse_metadata,
    BASE_TRANSFORM,
    SUBSET_DIR,
)
from features import create_feature_extractors
from model import StreetCLIPFusion

CENTROID_CACHE_DIR = OUTPUT_DIR / "centroids"
CENTROID_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _centroid_cache_path(data_dir: Path, checkpoint_path: str) -> Path:
    key = f"{data_dir.resolve()}_{checkpoint_path}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CENTROID_CACHE_DIR / f"centroids_{h}.pt"


def load_model(checkpoint_path: Optional[str] = None, device: str = "cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model = StreetCLIPFusion(freeze_backbone=False)
    if checkpoint_path:
        state = torch.load(checkpoint_path, weights_only=True)
    else:
        best_path = CHECKPOINT_DIR / "best_model.pt"
        checkpoint_path = str(best_path)
        if best_path.exists():
            state = torch.load(best_path, weights_only=True)
        else:
            raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_DIR}")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def compute_country_centroids(
    model,
    data_dir: Path,
    country_encoder: CountryEncoder,
    device: str = "cuda",
    max_per_country: int = 200,
    force_refresh: bool = False,
    checkpoint_path: str = "",
):
    device = device if torch.cuda.is_available() else "cpu"
    cache_path = _centroid_cache_path(data_dir, checkpoint_path)

    if cache_path.exists() and not force_refresh:
        centroids = torch.load(cache_path, weights_only=True)
        for k in centroids:
            if centroids[k].ndim == 2 and centroids[k].shape[0] == 1:
                centroids[k] = centroids[k].squeeze(0)
        cached = {c for c in country_encoder.country_list if c in centroids}
        missing = [c for c in country_encoder.country_list if c not in centroids]
        if not missing:
            print(f"Loaded {len(centroids)} centroids from cache: {cache_path}")
            return centroids
        print(f"Loaded {len(centroids)} cached centroids, {len(missing)} new countries to compute")
    else:
        centroids = {}

    samples = gather_samples(data_dir)
    country_embs = {c: [] for c in country_encoder.country_list}
    for c in centroids:
        country_embs[c] = []

    for img_path, json_path in tqdm(samples, desc="Building centroids"):
        meta = parse_metadata(json_path)
        country = meta["country_name"]
        if country not in country_embs:
            continue
        if len(country_embs[country]) >= max_per_country:
            continue
        image = Image.open(img_path).convert("RGB")
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)
        road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
        veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)
        emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)
        country_embs[country].append(emb.cpu())

    for country, embs in country_embs.items():
        if embs:
                centroids[country] = torch.stack(embs).mean(dim=0).squeeze(0)

    torch.save(centroids, cache_path)
    print(f"Saved {len(centroids)} centroids to {cache_path}")
    return centroids


@torch.no_grad()
def evaluate(
    model,
    data_dir: Path,
    country_encoder: CountryEncoder,
    centroids: dict,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    use_features: bool = True,
):
    device = device if torch.cuda.is_available() else "cpu"
    samples = gather_samples(data_dir)
    if max_samples:
        samples = samples[:max_samples]

    extractor = None
    if use_features:
        try:
            extractor = create_feature_extractors(
                road_model="grounding_dino", veg_model="clip", device=device
            )
        except Exception:
            extractor = None

    correct_1 = 0
    correct_5 = 0
    total = 0

    centroid_matrix = torch.stack([centroids[c] for c in country_encoder.country_list]).to(device)

    for img_path, json_path in tqdm(samples, desc="Evaluating"):
        meta = parse_metadata(json_path)
        true_country = meta["country_name"]
        if true_country not in centroids:
            continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)

        if extractor is not None:
            features = extractor.extract(image)
            road_f = features["road_features"].unsqueeze(0).to(device)
            veg_f = features["veg_features"].unsqueeze(0).to(device)
        else:
            road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
            veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)

        emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)
        sim = torch.matmul(emb, centroid_matrix.T).squeeze(0)
        top5_indices = sim.argsort(descending=True)[:5].cpu().numpy()
        top5_countries = [country_encoder.country_list[i] for i in top5_indices]

        if top5_countries[0] == true_country:
            correct_1 += 1
        if true_country in top5_countries:
            correct_5 += 1
        total += 1

    print(f"Samples evaluated: {total}")
    print(f"Top-1 accuracy: {correct_1/total*100:.2f}%")
    print(f"Top-5 accuracy: {correct_5/total*100:.2f}%")
    return correct_1 / total, correct_5 / total


@torch.no_grad()
def predict_single(
    model,
    image_path: str,
    country_encoder: CountryEncoder,
    centroids: dict,
    device: str = "cuda",
    top_k: int = 5,
):
    device = device if torch.cuda.is_available() else "cpu"
    image = Image.open(image_path).convert("RGB")
    pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)
    road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
    veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)
    emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)

    centroid_matrix = torch.stack([centroids[c] for c in country_encoder.country_list]).to(device)
    sim = torch.matmul(emb, centroid_matrix.T).squeeze(0)
    topk = sim.argsort(descending=True)[:top_k].cpu().numpy()
    results = [(country_encoder.country_list[i], sim[i].item()) for i in topk]
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--refresh-centroids", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else SUBSET_DIR
    print(f"Data dir: {data_dir}")

    model, ckpt_path = load_model(args.checkpoint, args.device)
    country_encoder = CountryEncoder(data_dir)
    print(f"Countries: {len(country_encoder)}")

    centroids = compute_country_centroids(
        model, data_dir, country_encoder, args.device,
        force_refresh=args.refresh_centroids,
        checkpoint_path=ckpt_path,
    )

    if args.image:
        results = predict_single(model, args.image, country_encoder, centroids, args.device)
        print("\nPredictions:")
        for country, score in results:
            print(f"  {country}: {score:.4f}")
    else:
        evaluate(
            model, data_dir, country_encoder, centroids, args.device,
            max_samples=args.max_samples,
            use_features=not args.no_features,
        )
