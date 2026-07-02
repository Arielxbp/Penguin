import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import (
    AUGMENTATIONS_PER_IMAGE,
    EMBEDDING_DIR,
    FEATURE_DIR,
    STREETCLIP_MODEL,
    PRECOMPUTE_EMBED_BATCH,
    PRECOMPUTE_FEATURE_BATCH,
    PRECOMPUTE_NUM_WORKERS,
    USE_SUBSET,
    SHARD_SIZE,
)

from dataset import (
    GeoSampleDataset,
    CountryEncoder,
    gather_samples,
    parse_metadata,
    BASE_TRANSFORM,
    PIL_AUG_TRANSFORM,
    TENSOR_AUG_TRANSFORM,
    SUBSET_DIR,
)
import torchvision.transforms as T


class PrecomputeDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        img_id = img_path.stem
        image = Image.open(img_path).convert("RGB")

        orig_tensor = BASE_TRANSFORM(image)
        variants = generate_augmented_variants(image)
        variant_tensor = torch.stack(variants)

        return img_id, image, orig_tensor, variant_tensor


def _embed_collate(batch):
    ids = [item[0] for item in batch]
    orig = torch.stack([item[2] for item in batch])
    variants = torch.stack([item[3] for item in batch])
    return ids, orig, variants


def _feature_collate(batch):
    ids = [item[0] for item in batch]
    pil_images = [item[1] for item in batch]
    variants = torch.stack([item[3] for item in batch])
    return ids, pil_images, variants


def load_streetclip_vision_encoder(device: str = "cuda"):
    model = CLIPModel.from_pretrained(STREETCLIP_MODEL).to(device)
    model.eval()
    return model.vision_model, model.visual_projection


def extract_streetclip_embedding(image, vision_model, visual_projection, device="cuda"):
    if isinstance(image, Image.Image):
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)
    else:
        pixel_values = image.unsqueeze(0).to(device)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
    with torch.inference_mode():
        vision_outputs = vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        embedding = visual_projection(pooled)
    return embedding.squeeze(0).cpu()


def generate_augmented_variants(image: Image.Image, n_variants: int = AUGMENTATIONS_PER_IMAGE):
    variants = []
    for _ in range(n_variants):
        aug_img = PIL_AUG_TRANSFORM(image)
        aug_tensor = T.ToTensor()(aug_img)
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        aug_tensor = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )(aug_tensor)
        variants.append(aug_tensor)
    return variants


def precompute_embeddings(
    data_dir: Optional[Path] = None,
    device: str = "cuda",
    shard_size: int = SHARD_SIZE,
):
    data_dir = data_dir or SUBSET_DIR
    emb_dir = EMBEDDING_DIR
    emb_dir.mkdir(parents=True, exist_ok=True)
    index_path = emb_dir / "embedding_index.json"

    vision_model, visual_projection = load_streetclip_vision_encoder(device)

    samples = gather_samples(data_dir)

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        valid = {k: v for k, v in index.items() if (emb_dir / v).exists()}
        index = valid
        completed = set(index.keys())
        samples = [(p, j) for p, j in samples if p.stem not in completed]
        shard_idx = len(set(index.values()))
        total_done = len(completed)
        print(f"Resuming embeddings: {total_done} completed, {len(samples)} remaining")
    else:
        index = {}
        shard_idx = 0
        total_done = 0
        print(f"Precomputing StreetCLIP embeddings for {len(samples)} images...")

    if not samples:
        print("All embeddings already precomputed.")
        return index

    batch_orig = []
    batch_aug = [[] for _ in range(AUGMENTATIONS_PER_IMAGE)]
    batch_img_ids = []

    def flush_shard():
        nonlocal shard_idx, index
        if not batch_img_ids:
            return
        shard_name = f"emb_shard_{shard_idx:04d}.pt"
        shard_data = {}
        for i, img_id in enumerate(batch_img_ids):
            shard_data[img_id] = batch_orig[i]
            for j in range(AUGMENTATIONS_PER_IMAGE):
                shard_data[f"{img_id}_aug{j}"] = batch_aug[j][i]

        shard_tmp = emb_dir / f".{shard_name}.tmp"
        torch.save(shard_data, shard_tmp)
        shard_tmp.replace(emb_dir / shard_name)

        for img_id in batch_img_ids:
            index[img_id] = shard_name
        tmp_idx = index_path.with_suffix(".tmp")
        with open(tmp_idx, "w") as f:
            json.dump(index, f)
        tmp_idx.replace(index_path)

        batch_img_ids.clear()
        batch_orig.clear()
        for b in batch_aug:
            b.clear()
        shard_idx += 1

    dataset = PrecomputeDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=PRECOMPUTE_EMBED_BATCH,
        shuffle=False,
        num_workers=PRECOMPUTE_NUM_WORKERS,
        collate_fn=_embed_collate,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )

    VIEWS = 1 + AUGMENTATIONS_PER_IMAGE
    total = total_done
    for ids, orig_batch, var_batch in tqdm(loader, total=len(loader)):
        B = len(ids)

        all_tensors = []
        for i in range(B):
            all_tensors.append(orig_batch[i])
            for j in range(AUGMENTATIONS_PER_IMAGE):
                all_tensors.append(var_batch[i, j])

        all_tensor = torch.stack(all_tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            vision_outputs = vision_model(pixel_values=all_tensor)
            all_embs = visual_projection(vision_outputs.pooler_output).cpu()

        for i in range(B):
            batch_orig.append(all_embs[i * VIEWS])
            for j in range(AUGMENTATIONS_PER_IMAGE):
                batch_aug[j].append(all_embs[i * VIEWS + 1 + j])

        batch_img_ids.extend(ids)
        total += B
        if len(batch_img_ids) >= shard_size:
            flush_shard()
            print(f"  Wrote {shard_idx} shards ({total} samples)")

    flush_shard()

    total_shards = len(set(index.values()))
    print(f"Embeddings saved to {emb_dir} ({total_shards} shard(s))")
    print(f"Index saved to {index_path}")
    return index


def precompute_features(
    data_dir: Optional[Path] = None,
    device: str = "cuda",
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
    shard_size: int = SHARD_SIZE,
):
    from features import create_feature_extractors

    data_dir = data_dir or SUBSET_DIR
    feat_dir = FEATURE_DIR
    feat_dir.mkdir(parents=True, exist_ok=True)
    index_path = feat_dir / "feature_index.json"

    extractor = create_feature_extractors(road_model=road_model, veg_model=veg_model, device=device)
    samples = gather_samples(data_dir)

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        valid = {k: v for k, v in index.items() if (feat_dir / v).exists()}
        index = valid
        completed = set(index.keys())
        samples = [(p, j) for p, j in samples if p.stem not in completed]
        shard_idx = len(set(index.values()))
        total_done = len(completed)
        print(f"Resuming features: {total_done} completed, {len(samples)} remaining")
    else:
        index = {}
        shard_idx = 0
        total_done = 0
        print(f"Precomputing features ({road_model} + {veg_model}) for {len(samples)} images...")

    if not samples:
        print("All features already precomputed.")
        return index

    batch_img_ids = []
    batch_feats = []
    batch_aug_feats = [[] for _ in range(AUGMENTATIONS_PER_IMAGE)]

    def flush_shard():
        nonlocal shard_idx, index
        if not batch_img_ids:
            return
        shard_name = f"feat_shard_{shard_idx:04d}.pt"
        shard_data = {}
        for i, img_id in enumerate(batch_img_ids):
            shard_data[img_id] = batch_feats[i]
            for j in range(AUGMENTATIONS_PER_IMAGE):
                shard_data[f"{img_id}_aug{j}"] = batch_aug_feats[j][i]

        shard_tmp = feat_dir / f".{shard_name}.tmp"
        torch.save(shard_data, shard_tmp)
        shard_tmp.replace(feat_dir / shard_name)

        for img_id in batch_img_ids:
            index[img_id] = shard_name
        tmp_idx = index_path.with_suffix(".tmp")
        with open(tmp_idx, "w") as f:
            json.dump(index, f)
        tmp_idx.replace(index_path)

        batch_img_ids.clear()
        batch_feats.clear()
        for b in batch_aug_feats:
            b.clear()
        shard_idx += 1

    dataset = PrecomputeDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=PRECOMPUTE_FEATURE_BATCH,
        shuffle=False,
        num_workers=PRECOMPUTE_NUM_WORKERS,
        collate_fn=_feature_collate,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )

    VIEWS = 1 + AUGMENTATIONS_PER_IMAGE
    total = total_done
    for ids, pil_images, var_batch in tqdm(loader, total=len(loader)):
        B = len(ids)

        all_inputs = list(pil_images)
        for i in range(B):
            for j in range(AUGMENTATIONS_PER_IMAGE):
                all_inputs.append(var_batch[i, j])

        all_features = extractor.extract_batch(all_inputs)

        for i in range(B):
            batch_feats.append({
                "road_features": all_features["road_features"][i * VIEWS],
                "veg_features": all_features["veg_features"][i * VIEWS],
            })
            for j in range(AUGMENTATIONS_PER_IMAGE):
                idx = i * VIEWS + 1 + j
                batch_aug_feats[j].append({
                    "road_features": all_features["road_features"][idx],
                    "veg_features": all_features["veg_features"][idx],
                })

        batch_img_ids.extend(ids)
        total += B
        if len(batch_img_ids) >= shard_size:
            flush_shard()
            print(f"  Wrote {shard_idx} shards ({total} samples)")

    flush_shard()

    total_shards = len(set(index.values()))
    print(f"Features saved to {feat_dir} ({total_shards} shard(s))")
    print(f"Index saved to {index_path}")
    return index


def precompute_all(
    data_dir: Optional[Path] = None,
    device: str = "cuda",
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
    shard_size: int = SHARD_SIZE,
):
    print("=" * 60)
    print("PRECOMPUTING STREETCLIP EMBEDDINGS")
    print("=" * 60)
    precompute_embeddings(data_dir=data_dir, device=device, shard_size=shard_size)

    print()
    print("=" * 60)
    print("PRECOMPUTING ROAD + VEGETATION FEATURES")
    print("=" * 60)
    precompute_features(data_dir=data_dir, device=device, road_model=road_model, veg_model=veg_model, shard_size=shard_size)

    print()
    print("=" * 60)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["embeddings", "features", "all"], default="all")
    parser.add_argument("--road-model", choices=["grounding_dino", "yolo_world"], default="grounding_dino")
    parser.add_argument("--veg-model", choices=["ram++", "clip"], default="clip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    if args.mode == "embeddings":
        precompute_embeddings(data_dir=args.data_dir, device=args.device, shard_size=args.shard_size)
    elif args.mode == "features":
        precompute_features(
            data_dir=args.data_dir,
            device=args.device,
            road_model=args.road_model,
            veg_model=args.veg_model,
            shard_size=args.shard_size,
        )
    else:
        precompute_all(
            data_dir=args.data_dir,
            device=args.device,
            road_model=args.road_model,
            veg_model=args.veg_model,
            shard_size=args.shard_size,
        )
