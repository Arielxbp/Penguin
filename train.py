import json
from pathlib import Path
from typing import Optional

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    FEATURE_DIR,
    FUSION_OUTPUT_DIM,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOG_DIR,
    MAX_GRAD_NORM,
    NUM_EPOCHS,
    NUM_WORKERS,
    OBJ_FEATURE_DIM,
    SEED,
    STREETCLIP_EMBED_DIM,
    TEMPERATURE,
    TRAIN_SPLIT,
    VEG_FEATURE_DIM,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

from dataset import (
    GeoSampleDataset,
    CountryEncoder,
    gather_samples,
    parse_metadata,
    split_dataset,
    SUBSET_DIR,
)

from model import StreetCLIPFusion, ContrastiveLoss, MultiModalContrastiveLoss

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PrecomputedDataset(Dataset):
    def __init__(
        self,
        sample_indices: list,
        all_samples: list,
        country_encoder: CountryEncoder,
        embedding_dir: Path,
        feature_dir: Optional[Path] = None,
        plonkit_embeddings: Optional[dict] = None,
    ):
        self.indices = sample_indices
        self.all_samples = all_samples
        self.country_encoder = country_encoder
        self.embedding_dir = embedding_dir
        self.feature_dir = feature_dir
        self.plonkit_embeddings = plonkit_embeddings

        with open(embedding_dir / "embedding_index.json") as f:
            self.emb_index = json.load(f)

        self.feat_index = {}
        if feature_dir:
            feat_index_path = feature_dir / "feature_index.json"
            if feat_index_path.exists():
                with open(feat_index_path) as f:
                    self.feat_index = json.load(f)

        needed_emb_shards = set()
        needed_feat_shards = set()
        for idx in self.indices:
            img_path, _ = self.all_samples[idx]
            img_id = img_path.stem
            if img_id in self.emb_index:
                needed_emb_shards.add(self.emb_index[img_id])
            if img_id in self.feat_index:
                needed_feat_shards.add(self.feat_index[img_id])

        self._emb_shards = {}
        for shard_name in needed_emb_shards:
            self._emb_shards[shard_name] = torch.load(
                self.embedding_dir / shard_name, weights_only=False
            )

        self._feat_shards = {}
        for shard_name in needed_feat_shards:
            self._feat_shards[shard_name] = torch.load(
                self.feature_dir / shard_name, weights_only=False
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        img_path, json_path = self.all_samples[sample_idx]
        img_id = img_path.stem
        meta = parse_metadata(json_path)
        country_name = meta["country_name"]
        country_idx = self.country_encoder.encode(country_name)

        emb = torch.zeros(STREETCLIP_EMBED_DIM)
        shard = self.emb_index.get(img_id)
        if shard:
            emb = self._emb_shards[shard][img_id].float()

        road_feat = torch.zeros(OBJ_FEATURE_DIM)
        veg_feat = torch.zeros(VEG_FEATURE_DIM)
        feat_shard = self.feat_index.get(img_id)
        if feat_shard:
            feat = self._feat_shards[feat_shard].get(img_id)
            if feat:
                road_feat = feat["road_features"].float()
                veg_feat = feat["veg_features"].float()

        result = {
            "embedding": emb,
            "road_features": road_feat,
            "veg_features": veg_feat,
            "country_idx": torch.tensor(country_idx, dtype=torch.long),
            "country_name": country_name,
            "img_id": img_id,
        }

        if self.plonkit_embeddings and country_name in self.plonkit_embeddings:
            result["country_text_emb"] = self.plonkit_embeddings[country_name].float()

        return result


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_plonkit(model, dataloader, optimizer, criterion, device,
                        scheduler=None, gradient_accumulation_steps=1,
                        max_grad_norm=None, epoch=0):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Plonkit]")
    for step, batch in enumerate(pbar):
        embeddings = batch["embedding"].to(device)
        road_features = batch["road_features"].to(device)
        veg_features = batch["veg_features"].to(device)
        text_embeddings = batch["country_text_emb"].to(device)
        labels = batch["country_idx"].to(device)

        image_emb = model(
            embeddings=embeddings,
            road_features=road_features,
            veg_features=veg_features,
        )
        loss = criterion(image_emb, text_embeddings, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


def train_epoch(
    model, dataloader, optimizer, criterion, device,
    scheduler=None, gradient_accumulation_steps=1,
    max_grad_norm=None, epoch=0,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        embeddings = batch["embedding"].to(device)
        road_features = batch["road_features"].to(device)
        veg_features = batch["veg_features"].to(device)
        labels = batch["country_idx"].to(device)

        output = model(
            embeddings=embeddings,
            road_features=road_features,
            veg_features=veg_features,
        )
        loss = criterion(output, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_plonkit: bool = False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Eval"):
        embeddings = batch["embedding"].to(device)
        road_features = batch["road_features"].to(device)
        veg_features = batch["veg_features"].to(device)
        labels = batch["country_idx"].to(device)

        output = model(
            embeddings=embeddings,
            road_features=road_features,
            veg_features=veg_features,
        )

        if use_plonkit and "country_text_emb" in batch:
            text_embs = batch["country_text_emb"].to(device)
            loss = criterion(output, text_embs, labels)
        else:
            loss = criterion(output, labels)

        total_loss += loss.item()
        total += labels.size(0)

        sim = torch.matmul(output, output.T)
        sim = sim.masked_fill(torch.eye(sim.size(0), dtype=torch.bool, device=sim.device), -1e9)
        pred = sim.argmax(dim=1)
        correct += (labels == labels[pred]).sum().item()

    return total_loss / len(dataloader), correct / max(total, 1)


def train(
    data_dir: Optional[Path] = None,
    device: str = "cuda",
    use_precomputed: bool = True,
    use_plonkit: bool = True,
    resume_from: Optional[str] = None,
):
    device = device if torch.cuda.is_available() else "cpu"
    data_dir = data_dir or SUBSET_DIR
    samples = gather_samples(data_dir)
    country_encoder = CountryEncoder(data_dir)
    print(f"Dataset: {len(samples)} samples, {len(country_encoder)} countries")

    train_indices, val_indices = split_dataset(
        GeoSampleDataset(samples, country_encoder, augment=False),
        train_ratio=TRAIN_SPLIT,
    )
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    plonkit_embs = None
    if use_plonkit:
        from plonkit_integration import PlonkitCountryEncoder
        pk_encoder = PlonkitCountryEncoder(device=device)
        pk_encoder.precompute_embeddings(country_encoder.country_list)
        plonkit_embs = pk_encoder.country_embeddings
        matched = sum(1 for c in country_encoder.country_list if c in plonkit_embs)
        print(f"Plonkit text embeddings: {matched}/{len(country_encoder.country_list)} countries")

    if use_precomputed:
        train_dataset = PrecomputedDataset(
            train_indices, samples, country_encoder,
            EMBEDDING_DIR, FEATURE_DIR, plonkit_embs,
        )
        val_dataset = PrecomputedDataset(
            val_indices, samples, country_encoder,
            EMBEDDING_DIR, FEATURE_DIR, plonkit_embs,
        )
    else:
        train_dataset = GeoSampleDataset(
            [samples[i] for i in train_indices], country_encoder, augment=True
        )
        val_dataset = GeoSampleDataset(
            [samples[i] for i in val_indices], country_encoder, augment=False
        )

    num_workers = 0 if use_precomputed else NUM_WORKERS

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model = StreetCLIPFusion(freeze_backbone=True, fusion_output_dim=FUSION_OUTPUT_DIM)
    if resume_from:
        state = torch.load(resume_from, weights_only=True)
        model.load_state_dict(state, strict=False)
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if use_plonkit:
        criterion = MultiModalContrastiveLoss(temperature=TEMPERATURE)
        train_fn = train_epoch_plonkit
        eval_use_plonkit = True
    else:
        criterion = ContrastiveLoss(temperature=TEMPERATURE)
        train_fn = train_epoch
        eval_use_plonkit = False

    criterion = criterion.to(device)

    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_linear_warmup_scheduler(optimizer, WARMUP_STEPS, total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            epoch=epoch,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            use_plonkit=eval_use_plonkit,
        )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved (val_loss={val_loss:.4f})")

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-precomputed", action="store_true")
    parser.add_argument("--no-plonkit", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        device=args.device,
        use_precomputed=not args.no_precomputed,
        use_plonkit=not args.no_plonkit,
        resume_from=args.resume,
    )
