# Import the json module for reading embedding and feature index files.
import json
# Import Path from pathlib for cross-platform filesystem path handling.
from pathlib import Path
# Import Optional for optional type hints.
from typing import Optional

# Import numpy for numerical operations and random seeding.
import numpy as np
# Import random for Python-level random seeding.
import random
# Import PyTorch for tensor operations, GPU acceleration, and neural network training.
import torch
# Import DataLoader and Dataset for batched data loading.
from torch.utils.data import DataLoader, Dataset
# Import the AdamW optimizer for training with decoupled weight decay.
from torch.optim import AdamW
# Import tqdm for progress bar visualization during training and evaluation loops.
from tqdm import tqdm

# Import configuration constants from the project's config module.
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

# Import dataset-related utilities from the project's dataset module.
from dataset import (
    GeoSampleDataset,
    CountryEncoder,
    gather_samples,
    parse_metadata,
    split_dataset,
    SUBSET_DIR,
)

# Import the fusion model and contrastive loss classes from the project's model module.
from model import StreetCLIPFusion, ContrastiveLoss, MultiModalContrastiveLoss

# Seed the Python random module for reproducibility.
random.seed(SEED)
# Seed the NumPy random number generator for reproducibility.
np.random.seed(SEED)
# Seed the PyTorch random number generator for reproducibility.
torch.manual_seed(SEED)


# A PyTorch Dataset that loads precomputed embeddings and features from sharded .pt files.
class PrecomputedDataset(Dataset):
    # Initialize the dataset with indices, samples, encoder, and paths to precomputed data.
    def __init__(
        # Self reference.
        self,
        # List of integer indices into all_samples for the samples assigned to this dataset split.
        sample_indices: list,
        # The full list of (image_path, json_path) sample tuples.
        all_samples: list,
        # An encoder that maps country names to integer indices.
        country_encoder: CountryEncoder,
        # Path to the directory containing embedding shard files and the embedding index JSON.
        embedding_dir: Path,
        # Optional path to the directory containing feature shard files and the feature index JSON.
        feature_dir: Optional[Path] = None,
        # Optional dictionary mapping country names to precomputed Plonkit text embeddings.
        plonkit_embeddings: Optional[dict] = None,
    ):
        # Store the list of sample indices assigned to this dataset split.
        self.indices = sample_indices
        # Store the full list of samples for lookup by index.
        self.all_samples = all_samples
        # Store the country encoder for mapping country names to indices.
        self.country_encoder = country_encoder
        # Store the path to the embedding directory.
        self.embedding_dir = embedding_dir
        # Store the optional path to the feature directory.
        self.feature_dir = feature_dir
        # Store the optional Plonkit text embeddings dictionary.
        self.plonkit_embeddings = plonkit_embeddings

        # Open and read the embedding index JSON file that maps image ids to shard filenames.
        with open(embedding_dir / "embedding_index.json") as f:
            # Parse the JSON into the emb_index dictionary.
            self.emb_index = json.load(f)

        # Initialize an empty feature index dictionary (will be populated if feature_dir is provided).
        self.feat_index = {}
        # If a feature directory is provided, load the feature index.
        if feature_dir:
            # Build the path to the feature index JSON file.
            feat_index_path = feature_dir / "feature_index.json"
            # Only proceed if the feature index file actually exists.
            if feat_index_path.exists():
                # Open and read the feature index JSON file.
                with open(feat_index_path) as f:
                    # Parse the JSON into the feat_index dictionary.
                    self.feat_index = json.load(f)

        # Collect the set of embedding shard filenames that this dataset's samples reference.
        needed_emb_shards = set()
        # Collect the set of feature shard filenames that this dataset's samples reference.
        needed_feat_shards = set()
        # Iterate over the sample indices assigned to this dataset split.
        for idx in self.indices:
            # Unpack the (image_path, json_path) tuple for this sample.
            img_path, _ = self.all_samples[idx]
            # Extract the filename stem as the image identifier.
            img_id = img_path.stem
            # If this image id is in the embedding index, add its shard to the needed set.
            if img_id in self.emb_index:
                # Add the shard filename to the set of needed embedding shards.
                needed_emb_shards.add(self.emb_index[img_id])
            # If this image id is in the feature index, add its shard to the needed set.
            if img_id in self.feat_index:
                # Add the shard filename to the set of needed feature shards.
                needed_feat_shards.add(self.feat_index[img_id])

        # Initialize a dictionary to cache loaded embedding shards in memory.
        self._emb_shards = {}
        # Load each required embedding shard from disk.
        for shard_name in needed_emb_shards:
            # Load the shard file with weights_only=False (contains dicts, not just model weights).
            self._emb_shards[shard_name] = torch.load(
                # Build the full path to the embedding shard file.
                self.embedding_dir / shard_name, weights_only=False
            )

        # Initialize a dictionary to cache loaded feature shards in memory.
        self._feat_shards = {}
        # Load each required feature shard from disk.
        for shard_name in needed_feat_shards:
            # Load the shard file with weights_only=False (contains dicts, not just model weights).
            self._feat_shards[shard_name] = torch.load(
                # Build the full path to the feature shard file (only if feature_dir is set).
                self.feature_dir / shard_name, weights_only=False
            )

    # Return the total number of samples in this dataset split.
    def __len__(self):
        # Return the length of the indices list.
        return len(self.indices)

    # Retrieve the idx-th sample: precomputed embedding, features, country label, and metadata.
    def __getitem__(self, idx):
        # Map the dataset-local index to the global sample index in all_samples.
        sample_idx = self.indices[idx]
        # Unpack the (image_path, json_path) tuple for this sample.
        img_path, json_path = self.all_samples[sample_idx]
        # Extract the filename stem as the image identifier.
        img_id = img_path.stem
        # Parse the JSON metadata file to extract country information.
        meta = parse_metadata(json_path)
        # Get the country name from the parsed metadata.
        country_name = meta["country_name"]
        # Encode the country name as an integer index using the country encoder.
        country_idx = self.country_encoder.encode(country_name)

        # Initialize a zero embedding tensor as fallback for missing data.
        emb = torch.zeros(STREETCLIP_EMBED_DIM)
        # Look up the embedding shard for this image id.
        shard = self.emb_index.get(img_id)
        # If a shard was found for this image id, retrieve its embedding.
        if shard:
            # Load the precomputed embedding from the cached shard and cast to float32.
            emb = self._emb_shards[shard][img_id].float()

        # Initialize a zero road feature tensor as fallback for missing data.
        road_feat = torch.zeros(OBJ_FEATURE_DIM)
        # Initialize a zero vegetation feature tensor as fallback for missing data.
        veg_feat = torch.zeros(VEG_FEATURE_DIM)
        # Look up the feature shard for this image id.
        feat_shard = self.feat_index.get(img_id)
        # If a feature shard was found for this image id, retrieve its feature dict.
        if feat_shard:
            # Get the feature dict for this image id from the cached shard.
            feat = self._feat_shards[feat_shard].get(img_id)
            # If a feature dict was found, extract road and vegetation features.
            if feat:
                # Load the precomputed road features and cast to float32.
                road_feat = feat["road_features"].float()
                # Load the precomputed vegetation features and cast to float32.
                veg_feat = feat["veg_features"].float()

        # Build the result dictionary containing all data for this sample.
        result = {
            # The precomputed StreetCLIP embedding.
            "embedding": emb,
            # The precomputed road detection features.
            "road_features": road_feat,
            # The precomputed vegetation features.
            "veg_features": veg_feat,
            # The country index as an integer tensor for the classification/contrastive target.
            "country_idx": torch.tensor(country_idx, dtype=torch.long),
            # The raw country name string for debugging and Plonkit lookup.
            "country_name": country_name,
            # The image identifier string for debugging.
            "img_id": img_id,
        }

        # If Plonkit text embeddings are available and this country has one, include it.
        if self.plonkit_embeddings and country_name in self.plonkit_embeddings:
            # Add the precomputed country text embedding to the result dict.
            result["country_text_emb"] = self.plonkit_embeddings[country_name].float()

        # Return the assembled result dictionary.
        return result


# Create a learning rate scheduler with linear warmup followed by cosine decay.
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    # Define the lambda function that computes the learning rate multiplier for a given step.
    def lr_lambda(current_step):
        # During the warmup phase, linearly ramp up from 0 to 1.
        if current_step < warmup_steps:
            # Return the linear warmup ratio (current / warmup), protected against division by zero.
            return float(current_step) / float(max(1, warmup_steps))
        # After warmup, compute the progress through the remaining steps.
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Return the cosine decay multiplier, starting from 1.0 and decaying to 0.0.
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    # Return a LambdaLR scheduler that multiplies the base LR by lr_lambda at each step.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Train for one epoch using the Plonkit multimodal contrastive loss (image embeddings vs country text embeddings).
def train_epoch_plonkit(model, dataloader, optimizer, criterion, device,
                        scheduler=None, gradient_accumulation_steps=1,
                        max_grad_norm=None, epoch=0):
    # Set the model to training mode (enables gradients, dropout, batch norm updates).
    model.train()
    # Initialize the cumulative loss accumulator for this epoch.
    total_loss = 0.0
    # Zero out any accumulated gradients from previous steps.
    optimizer.zero_grad()

    # Create a progress bar over the dataloader with a descriptive label.
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Plonkit]")
    # Iterate over batches with a step counter.
    for step, batch in enumerate(pbar):
        # Move the precomputed StreetCLIP embeddings to the target device.
        embeddings = batch["embedding"].to(device)
        # Move the precomputed road features to the target device.
        road_features = batch["road_features"].to(device)
        # Move the precomputed vegetation features to the target device.
        veg_features = batch["veg_features"].to(device)
        # Move the Plonkit country text embeddings to the target device.
        text_embeddings = batch["country_text_emb"].to(device)
        # Move the country index labels to the target device.
        labels = batch["country_idx"].to(device)

        # Forward pass: fuse the embedding, road features, and veg features into a joint image embedding.
        image_emb = model(
            # Pass the StreetCLIP embedding.
            embeddings=embeddings,
            # Pass the road detection features.
            road_features=road_features,
            # Pass the vegetation features.
            veg_features=veg_features,
        )
        # Compute the Plonkit contrastive loss between image embeddings and country text embeddings.
        loss = criterion(image_emb, text_embeddings, labels)
        # Scale the loss by the number of gradient accumulation steps.
        loss = loss / gradient_accumulation_steps
        # Backpropagate the scaled loss (accumulates gradients).
        loss.backward()

        # If enough steps have been accumulated (or it's the final step), perform an optimizer step.
        if (step + 1) % gradient_accumulation_steps == 0:
            # If a maximum gradient norm is set, clip gradients to prevent explosion.
            if max_grad_norm is not None:
                # Clip gradients of trainable parameters by the max norm.
                torch.nn.utils.clip_grad_norm_(
                    # Select only parameters that require gradients.
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm
                )
            # Perform the optimizer step to update model weights.
            optimizer.step()
            # If a learning rate scheduler is provided, step it.
            if scheduler is not None:
                # Step the LR scheduler.
                scheduler.step()
            # Zero gradients for the next accumulation cycle.
            optimizer.zero_grad()

        # Accumulate the un-scaled loss for reporting.
        total_loss += loss.item() * gradient_accumulation_steps
        # Update the progress bar's displayed loss value.
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    # Return the average loss over all batches in the epoch.
    return total_loss / len(dataloader)


# Train for one epoch using standard contrastive loss on image embeddings only.
def train_epoch(
    # The StreetCLIPFusion model.
    model,
    # The training dataloader.
    dataloader,
    # The optimizer (AdamW).
    optimizer,
    # The loss criterion (ContrastiveLoss).
    criterion,
    # The target device ("cuda" or "cpu").
    device,
    # Optional learning rate scheduler.
    scheduler=None,
    # Number of forward/backward passes to accumulate before an optimizer step.
    gradient_accumulation_steps=1,
    # Optional maximum gradient norm for clipping.
    max_grad_norm=None,
    # The current epoch number (for logging).
    epoch=0,
):
    # Set the model to training mode (enables gradients, dropout, batch norm updates).
    model.train()
    # Initialize the cumulative loss accumulator for this epoch.
    total_loss = 0.0
    # Zero out any accumulated gradients from previous steps.
    optimizer.zero_grad()

    # Create a progress bar over the dataloader with a descriptive label.
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    # Iterate over batches with a step counter.
    for step, batch in enumerate(pbar):
        # Move the precomputed StreetCLIP embeddings to the target device.
        embeddings = batch["embedding"].to(device)
        # Move the precomputed road features to the target device.
        road_features = batch["road_features"].to(device)
        # Move the precomputed vegetation features to the target device.
        veg_features = batch["veg_features"].to(device)
        # Move the country index labels to the target device.
        labels = batch["country_idx"].to(device)

        # Forward pass: fuse the embedding, road features, and veg features into a joint output.
        output = model(
            # Pass the StreetCLIP embedding.
            embeddings=embeddings,
            # Pass the road detection features.
            road_features=road_features,
            # Pass the vegetation features.
            veg_features=veg_features,
        )
        # Compute the contrastive loss between the output embeddings and country labels.
        loss = criterion(output, labels)
        # Scale the loss by the number of gradient accumulation steps.
        loss = loss / gradient_accumulation_steps
        # Backpropagate the scaled loss (accumulates gradients).
        loss.backward()

        # If enough steps have been accumulated (or it's the final step), perform an optimizer step.
        if (step + 1) % gradient_accumulation_steps == 0:
            # If a maximum gradient norm is set, clip gradients to prevent explosion.
            if max_grad_norm is not None:
                # Clip gradients of all model parameters by the max norm.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Perform the optimizer step to update model weights.
            optimizer.step()
            # If a learning rate scheduler is provided, step it.
            if scheduler is not None:
                # Step the LR scheduler.
                scheduler.step()
            # Zero gradients for the next accumulation cycle.
            optimizer.zero_grad()

        # Accumulate the un-scaled loss for reporting.
        total_loss += loss.item() * gradient_accumulation_steps
        # Update the progress bar's displayed loss value.
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    # Return the average loss over all batches in the epoch.
    return total_loss / len(dataloader)


# Evaluate the model on a validation dataloader, returning average loss and top-1 accuracy.
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_plonkit: bool = False):
    # Set the model to evaluation mode (disables dropout, uses running batch norm stats).
    model.eval()
    # Initialize the cumulative loss accumulator.
    total_loss = 0.0
    # Initialize the count of correct nearest-neighbor predictions.
    correct = 0
    # Initialize the total count of evaluated samples.
    total = 0

    # Iterate over validation batches with a progress bar.
    for batch in tqdm(dataloader, desc="Eval"):
        # Move the precomputed StreetCLIP embeddings to the target device.
        embeddings = batch["embedding"].to(device)
        # Move the precomputed road features to the target device.
        road_features = batch["road_features"].to(device)
        # Move the precomputed vegetation features to the target device.
        veg_features = batch["veg_features"].to(device)
        # Move the country index labels to the target device.
        labels = batch["country_idx"].to(device)

        # Forward pass: fuse the embedding, road features, and veg features into a joint output.
        output = model(
            # Pass the StreetCLIP embedding.
            embeddings=embeddings,
            # Pass the road detection features.
            road_features=road_features,
            # Pass the vegetation features.
            veg_features=veg_features,
        )

        # If using Plonkit and the batch contains country text embeddings, use multimodal loss.
        if use_plonkit and "country_text_emb" in batch:
            # Move the Plonkit country text embeddings to the device.
            text_embs = batch["country_text_emb"].to(device)
            # Compute the Plonkit contrastive loss (image vs text embeddings, with labels).
            loss = criterion(output, text_embs, labels)
        # Otherwise, use the standard contrastive loss on image embeddings only.
        else:
            # Compute the standard contrastive loss.
            loss = criterion(output, labels)

        # Accumulate the batch loss.
        total_loss += loss.item()
        # Add the number of samples in this batch to the total count.
        total += labels.size(0)

        # Compute the pairwise cosine similarity matrix between all outputs in the batch.
        sim = torch.matmul(output, output.T)
        # Mask the diagonal entries (self-similarity) with a large negative value so they are not chosen as nearest neighbors.
        sim = sim.masked_fill(torch.eye(sim.size(0), dtype=torch.bool, device=sim.device), -1e9)
        # For each sample, find the index of its most similar other sample.
        pred = sim.argmax(dim=1)
        # Count how many nearest neighbors share the same country label as the query sample.
        correct += (labels == labels[pred]).sum().item()

    # Return the average loss and the nearest-neighbor accuracy.
    return total_loss / len(dataloader), correct / max(total, 1)


# Main training function: sets up datasets, model, optimizer, scheduler, and runs the training loop.
def train(
    # Optional path to the data directory; defaults to SUBSET_DIR.
    data_dir: Optional[Path] = None,
    # Device to run training on ("cuda" or "cpu"); defaults to "cuda".
    device: str = "cuda",
    # Whether to use precomputed embeddings and features (instead of recomputing on-the-fly).
    use_precomputed: bool = True,
    # Whether to use Plonkit country text embeddings for multimodal contrastive training.
    use_plonkit: bool = True,
    # Optional path to a checkpoint file to resume training from.
    resume_from: Optional[str] = None,
):
    # Fall back to CPU if CUDA is unavailable for the specified device.
    device = device if torch.cuda.is_available() else "cpu"
    # Use the provided data directory or fall back to the configured subset directory.
    data_dir = data_dir or SUBSET_DIR
    # Gather all (image_path, json_path) sample tuples from the data directory.
    samples = gather_samples(data_dir)
    # Create a country encoder from the list of unique country names in the dataset.
    country_encoder = CountryEncoder(data_dir)
    # Print dataset statistics.
    print(f"Dataset: {len(samples)} samples, {len(country_encoder)} countries")

    # Split the dataset into training and validation indices using the configured ratio.
    train_indices, val_indices = split_dataset(
        # Pass a non-augmenting dataset for the purpose of splitting indices.
        GeoSampleDataset(samples, country_encoder, augment=False),
        # Use the configured training split ratio.
        train_ratio=TRAIN_SPLIT,
    )
    # Print the number of training and validation samples.
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Initialize Plonkit embeddings as None (will be populated if use_plonkit is True).
    plonkit_embs = None
    # If Plonkit multimodal training is enabled, load and precompute country text embeddings.
    if use_plonkit:
        # Import the Plonkit country encoder (lazy import to avoid dependency when not used).
        from plonkit_integration import PlonkitCountryEncoder
        # Instantiate the Plonkit country encoder on the target device.
        pk_encoder = PlonkitCountryEncoder(device=device)
        # Precompute text embeddings for all countries in the dataset.
        pk_encoder.precompute_embeddings(country_encoder.country_list)
        # Store the precomputed country-to-embedding mapping.
        plonkit_embs = pk_encoder.country_embeddings
        # Count how many of the dataset's countries have Plonkit text embeddings.
        matched = sum(1 for c in country_encoder.country_list if c in plonkit_embs)
        # Print the Plonkit coverage statistics.
        print(f"Plonkit text embeddings: {matched}/{len(country_encoder.country_list)} countries")

    # If using precomputed embeddings and features, load them from disk.
    if use_precomputed:
        # Create the training dataset from precomputed shards and Plonkit embeddings.
        train_dataset = PrecomputedDataset(
            # Training sample indices.
            train_indices,
            # Full list of samples for index lookup.
            samples,
            # Country encoder for label mapping.
            country_encoder,
            # Directory containing embedding shards.
            EMBEDDING_DIR,
            # Directory containing feature shards.
            FEATURE_DIR,
            # Optional Plonkit text embeddings dict.
            plonkit_embs,
        )
        # Create the validation dataset from precomputed shards and Plonkit embeddings.
        val_dataset = PrecomputedDataset(
            # Validation sample indices.
            val_indices,
            # Full list of samples for index lookup.
            samples,
            # Country encoder for label mapping.
            country_encoder,
            # Directory containing embedding shards.
            EMBEDDING_DIR,
            # Directory containing feature shards.
            FEATURE_DIR,
            # Optional Plonkit text embeddings dict.
            plonkit_embs,
        )
    # Otherwise, use the GeoSampleDataset that loads images and computes embeddings on-the-fly.
    else:
        # Create the training dataset with augmentation enabled.
        train_dataset = GeoSampleDataset(
            # Select only the training samples from the full list.
            [samples[i] for i in train_indices],
            # Country encoder for label mapping.
            country_encoder,
            # Enable data augmentation for training.
            augment=True,
        )
        # Create the validation dataset without augmentation.
        val_dataset = GeoSampleDataset(
            # Select only the validation samples from the full list.
            [samples[i] for i in val_indices],
            # Country encoder for label mapping.
            country_encoder,
            # Disable data augmentation for validation.
            augment=False,
        )

    # If using precomputed data, set num_workers to 0 (data is already in memory); otherwise use config value.
    num_workers = 0 if use_precomputed else NUM_WORKERS

    # Create the training DataLoader.
    train_loader = DataLoader(
        # The training dataset.
        train_dataset,
        # Batch size from config.
        batch_size=BATCH_SIZE,
        # Shuffle training samples each epoch.
        shuffle=True,
        # Number of subprocess workers (0 for precomputed).
        num_workers=num_workers,
        # Pin memory for faster GPU transfer.
        pin_memory=True,
    )
    # Create the validation DataLoader.
    val_loader = DataLoader(
        # The validation dataset.
        val_dataset,
        # Batch size from config.
        batch_size=BATCH_SIZE,
        # No shuffling needed for validation.
        shuffle=False,
        # Number of subprocess workers (0 for precomputed).
        num_workers=num_workers,
        # Pin memory for faster GPU transfer.
        pin_memory=True,
    )

    # Instantiate the StreetCLIPFusion model with a frozen backbone and configured output dimension.
    model = StreetCLIPFusion(freeze_backbone=True, fusion_output_dim=FUSION_OUTPUT_DIM)
    # If a checkpoint path was provided, load the model state dict into the model.
    if resume_from:
        # Load the checkpoint file with weights_only=True for safety.
        state = torch.load(resume_from, weights_only=True)
        # Load the state dict with strict=False to allow for partial checkpoints.
        model.load_state_dict(state, strict=False)
    # Move the model to the target device.
    model = model.to(device)

    # Collect only the trainable parameters (backbone is frozen, only fusion head learns).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Print the number of trainable parameters for diagnostics.
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # Create the AdamW optimizer with the configured learning rate and weight decay.
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Choose loss function and training function based on whether Plonkit is enabled.
    if use_plonkit:
        # Use multi-modal contrastive loss that includes country text embeddings.
        criterion = MultiModalContrastiveLoss(temperature=TEMPERATURE)
        # Assign the Plonkit-specific training function.
        train_fn = train_epoch_plonkit
        # Flag that evaluation should use Plonkit loss when text embeddings are available.
        eval_use_plonkit = True
    # Otherwise, use standard image-only contrastive loss.
    else:
        # Use standard contrastive loss on image embeddings.
        criterion = ContrastiveLoss(temperature=TEMPERATURE)
        # Assign the standard training function.
        train_fn = train_epoch
        # Flag that evaluation should use standard loss only.
        eval_use_plonkit = False

    # Move the criterion (loss function) to the target device.
    criterion = criterion.to(device)

    # Compute total training steps across all epochs for the LR scheduler.
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    # Create the linear warmup + cosine decay learning rate scheduler.
    scheduler = get_linear_warmup_scheduler(optimizer, WARMUP_STEPS, total_steps)

    # Initialize the best validation loss tracker with infinity.
    best_val_loss = float("inf")

    # Loop over epochs from 1 to NUM_EPOCHS inclusive.
    for epoch in range(1, NUM_EPOCHS + 1):
        # Run one training epoch and return the average training loss.
        train_loss = train_fn(
            # The fusion model.
            model,
            # The training dataloader.
            train_loader,
            # The AdamW optimizer.
            optimizer,
            # The loss criterion.
            criterion,
            # The target device.
            device,
            # The LR scheduler.
            scheduler=scheduler,
            # Number of gradient accumulation steps.
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            # Maximum gradient norm for clipping.
            max_grad_norm=MAX_GRAD_NORM,
            # Current epoch number for logging.
            epoch=epoch,
        )
        # Evaluate the model on the validation set and return validation loss and accuracy.
        val_loss, val_acc = evaluate(
            # The fusion model.
            model,
            # The validation dataloader.
            val_loader,
            # The loss criterion.
            criterion,
            # The target device.
            device,
            # Whether to use Plonkit loss (if text embeddings are available in the batch).
            use_plonkit=eval_use_plonkit,
        )

        # Print the epoch summary with training loss, validation loss, and validation accuracy.
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Build the path for the epoch checkpoint file.
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
        # Save the current model state dict as an epoch checkpoint.
        torch.save(model.state_dict(), checkpoint_path)

        # If this epoch's validation loss is the best so far, save it as the best model.
        if val_loss < best_val_loss:
            # Update the best validation loss.
            best_val_loss = val_loss
            # Build the path for the best model checkpoint.
            best_path = CHECKPOINT_DIR / "best_model.pt"
            # Save the model state dict as the best model.
            torch.save(model.state_dict(), best_path)
            # Print notification that a new best model was saved.
            print(f"  New best model saved (val_loss={val_loss:.4f})")

    # Print the final training summary with the best validation loss achieved.
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    # Return the trained model.
    return model


# Entry point when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Import argparse for command-line argument parsing.
    import argparse

    # Create an argument parser with a description.
    parser = argparse.ArgumentParser()
    # Add an optional argument for a custom data directory path.
    parser.add_argument("--data-dir", default=None)
    # Add an optional argument for the device; auto-detect CUDA availability for default.
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Add a flag to disable the use of precomputed embeddings/features.
    parser.add_argument("--no-precomputed", action="store_true")
    # Add a flag to disable the use of Plonkit text embeddings.
    parser.add_argument("--no-plonkit", action="store_true")
    # Add an optional argument to specify a checkpoint path for resuming training.
    parser.add_argument("--resume", default=None)
    # Parse the command-line arguments into an args namespace.
    args = parser.parse_args()

    # Call the main training function with the parsed command-line arguments.
    train(
        # Pass the data directory argument.
        data_dir=args.data_dir,
        # Pass the device argument.
        device=args.device,
        # Use precomputed data unless --no-precomputed flag is set.
        use_precomputed=not args.no_precomputed,
        # Use Plonkit unless --no-plonkit flag is set.
        use_plonkit=not args.no_plonkit,
        # Pass the resume checkpoint path.
        resume_from=args.resume,
    )
