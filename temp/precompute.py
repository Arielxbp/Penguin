# Import the json module for reading/writing JSON index files.
import json
# Import Path from pathlib for cross-platform filesystem path handling.
from pathlib import Path
# Import Optional for optional type hints.
from typing import Optional

# Import PyTorch for tensor operations and GPU acceleration.
import torch
# Import PIL Image for image loading and manipulation.
from PIL import Image
# Import DataLoader and Dataset for batched data loading.
from torch.utils.data import DataLoader, Dataset
# Import tqdm for progress bar visualization of long-running loops.
from tqdm import tqdm
# Import CLIPModel and CLIPProcessor from HuggingFace transformers for vision encoding.
from transformers import CLIPModel, CLIPProcessor

# Import configuration constants from the project's config module.
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

# Import dataset-related utilities and constants from the project's dataset module.
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
# Import torchvision transforms for tensor-level augmentations (ToTensor, Normalize).
import torchvision.transforms as T


# Define a PyTorch Dataset that loads images and generates augmented variants for precomputation.
class PrecomputeDataset(Dataset):
    # Initialize the dataset with a list of (image_path, json_path) sample tuples.
    def __init__(self, samples):
        # Store the samples list on the instance.
        self.samples = samples

    # Return the total number of samples in the dataset.
    def __len__(self):
        # Return the length of the samples list.
        return len(self.samples)

    # Load the idx-th sample: read image, apply base transform, generate augmented variants.
    def __getitem__(self, idx):
        # Unpack the (image_path, json_path) tuple for this sample index.
        img_path, _ = self.samples[idx]
        # Extract the filename stem (without extension) as the image identifier.
        img_id = img_path.stem
        # Open the image file and convert it to RGB color space.
        image = Image.open(img_path).convert("RGB")

        # Apply the base transform (resize, center crop, normalize) for the original image.
        orig_tensor = BASE_TRANSFORM(image)
        # Generate multiple augmented variants of this image (random crops, flips, color jitter).
        variants = generate_augmented_variants(image)
        # Stack the list of augmented tensors into a single tensor (N_aug, C, H, W).
        variant_tensor = torch.stack(variants)

        # Return the image id, PIL image, base-transformed tensor, and stacked augmented variants.
        return img_id, image, orig_tensor, variant_tensor


# Collate function for the embedding dataloader: extracts ids, original tensors, and variant tensors from a batch.
def _embed_collate(batch):
    # Collect the image id strings from each batch item.
    ids = [item[0] for item in batch]
    # Stack the base-transformed originals into a batch tensor (B, C, H, W).
    orig = torch.stack([item[2] for item in batch])
    # Stack the augmented variant batches into a consolidated tensor (B, N_aug, C, H, W).
    variants = torch.stack([item[3] for item in batch])
    # Return the collated batch.
    return ids, orig, variants


# Collate function for the feature dataloader: extracts ids, PIL images, and variant tensors from a batch.
def _feature_collate(batch):
    # Collect the image id strings from each batch item.
    ids = [item[0] for item in batch]
    # Collect the PIL Image objects (kept for feature extractors that need PIL input).
    pil_images = [item[1] for item in batch]
    # Stack the augmented variant batches into a consolidated tensor (B, N_aug, C, H, W).
    variants = torch.stack([item[3] for item in batch])
    # Return the collated batch.
    return ids, pil_images, variants


# Load the StreetCLIP vision model and its visual projection head, placing them on the given device.
def load_streetclip_vision_encoder(device: str = "cuda"):
    # Fall back to CPU if CUDA is not available (or if requested device is unavailable).
    device = device if torch.cuda.is_available() else "cpu"
    # Load the pretrained CLIP model from the configured model identifier and move it to the device.
    model = CLIPModel.from_pretrained(STREETCLIP_MODEL).to(device)
    # Set the model to evaluation mode (disables dropout, batch norm running stats updates).
    model.eval()
    # Return the vision model and visual projection layer separately.
    return model.vision_model, model.visual_projection


# Extract a StreetCLIP embedding from a single image (PIL or pre-transformed tensor).
def extract_streetclip_embedding(image, vision_model, visual_projection, device="cuda"):
    # Fall back to CPU if CUDA is not available (or if requested device is unavailable).
    device = device if torch.cuda.is_available() else "cpu"
    # If the input is a PIL Image, apply the base transform and add a batch dimension.
    if isinstance(image, Image.Image):
        # Apply base transform (resize, center crop, normalize), add batch dim, move to device.
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)
    # Otherwise the input is expected to be a pre-transformed tensor.
    else:
        # Add a batch dimension and move to the target device.
        pixel_values = image.unsqueeze(0).to(device)
        # If the tensor is 3D (C, H, W), add another batch dimension so it becomes (1, C, H, W).
        if pixel_values.dim() == 3:
            # Unsqueeze at dim 0 to go from (C, H, W) to (1, C, H, W).
            pixel_values = pixel_values.unsqueeze(0)
    # Use inference mode (no gradient tracking) for efficiency during feature extraction.
    with torch.inference_mode():
        # Forward pass through the vision transformer model.
        vision_outputs = vision_model(pixel_values=pixel_values)
        # Extract the pooled (CLS token) output representing the image embedding.
        pooled = vision_outputs.pooler_output
        # Project the pooled output through the visual projection layer.
        embedding = visual_projection(pooled)
    # Remove the batch dimension, move to CPU, and return the embedding tensor.
    return embedding.squeeze(0).cpu()


# Generate multiple augmented variants of a PIL Image using the configured transform pipeline.
def generate_augmented_variants(image: Image.Image, n_variants: int = AUGMENTATIONS_PER_IMAGE):
    # Initialize an empty list to collect the augmented tensor variants.
    variants = []
    # Loop n_variants times to produce the requested number of augmentations.
    for _ in range(n_variants):
        # Apply PIL-level augmentations (e.g., random resized crop, horizontal flip, color jitter).
        aug_img = PIL_AUG_TRANSFORM(image)
        # Convert the augmented PIL Image to a PyTorch tensor (C, H, W) in [0, 1].
        aug_tensor = T.ToTensor()(aug_img)
        # Apply tensor-level augmentations (e.g., additional augmentations after conversion).
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        # Apply CLIP-style normalization with ImageNet-Sketch / CLIP mean and std.
        aug_tensor = T.Normalize(
            # CLIP normalization mean values for RGB channels.
            mean=[0.48145466, 0.4578275, 0.40821073],
            # CLIP normalization std values for RGB channels.
            std=[0.26862954, 0.26130258, 0.27577711],
        )(aug_tensor)
        # Append the normalized augmented tensor to the variants list.
        variants.append(aug_tensor)
    # Return the list of augmented tensor variants.
    return variants


# Precompute StreetCLIP embeddings for all images in the dataset, saved in sharded .pt files with a JSON index.
def precompute_embeddings(
    # Path to the data directory containing images and metadata; defaults to SUBSET_DIR.
    data_dir: Optional[Path] = None,
    # Device to run inference on ("cuda" or "cpu"); defaults to "cuda".
    device: str = "cuda",
    # Maximum number of samples per embedding shard file; defaults to SHARD_SIZE from config.
    shard_size: int = SHARD_SIZE,
):
    # Fall back to CPU if CUDA is unavailable for the specified device.
    device = device if torch.cuda.is_available() else "cpu"
    # Use the provided data directory or fall back to the configured subset directory.
    data_dir = data_dir or SUBSET_DIR
    # Get the directory where embedding shards will be stored.
    emb_dir = EMBEDDING_DIR
    # Create the embedding output directory if it does not already exist.
    emb_dir.mkdir(parents=True, exist_ok=True)
    # Build the path to the JSON index file that maps image ids to shard filenames.
    index_path = emb_dir / "embedding_index.json"

    # Load the StreetCLIP vision encoder model on the target device.
    vision_model, visual_projection = load_streetclip_vision_encoder(device)

    # Collect all (image_path, json_path) sample tuples from the data directory.
    samples = gather_samples(data_dir)

    # Check if a previous embedding index file exists (resume support).
    if index_path.exists():
        # Open and read the existing JSON index file.
        with open(index_path) as f:
            # Parse the JSON to reconstruct the index dictionary.
            index = json.load(f)
        # Filter the index to only keep entries whose shard files still exist on disk.
        valid = {k: v for k, v in index.items() if (emb_dir / v).exists()}
        # Update the index to only contain valid entries.
        index = valid
        # Build a set of already-completed image ids for filtering.
        completed = set(index.keys())
        # Filter out samples whose embeddings have already been precomputed.
        samples = [(p, j) for p, j in samples if p.stem not in completed]
        # Determine the next shard index based on how many unique shards already exist.
        shard_idx = len(set(index.values()))
        # Track the total number of completed samples for progress reporting.
        total_done = len(completed)
        # Print a resume status message.
        print(f"Resuming embeddings: {total_done} completed, {len(samples)} remaining")
    # Otherwise, initialize a fresh precomputation run.
    else:
        # Start with an empty index dictionary.
        index = {}
        # Start shard numbering at 0.
        shard_idx = 0
        # No samples have been completed yet.
        total_done = 0
        # Print an initial status message.
        print(f"Precomputing StreetCLIP embeddings for {len(samples)} images...")

    # If no samples remain (all already precomputed), return early.
    if not samples:
        # Print a completion message.
        print("All embeddings already precomputed.")
        # Return the existing index.
        return index

    # Accumulator list for original image embeddings in the current shard.
    batch_orig = []
    # Accumulator for augmented embeddings: one list per augmentation slot, each collecting per-image vectors.
    batch_aug = [[] for _ in range(AUGMENTATIONS_PER_IMAGE)]
    # Accumulator list for image identifiers in the current shard.
    batch_img_ids = []

    # Inner function to flush the current in-memory batch to a shard file and update the index.
    def flush_shard():
        # Declare shard_idx and index as nonlocal so they can be modified from the enclosing scope.
        nonlocal shard_idx, index
        # If there is nothing accumulated, return without writing.
        if not batch_img_ids:
            # Early exit: no data to flush.
            return
        # Build the shard filename using zero-padded shard index.
        shard_name = f"emb_shard_{shard_idx:04d}.pt"
        # Initialize an empty dictionary to hold all embeddings for this shard.
        shard_data = {}
        # Iterate over each image in the current shard batch.
        for i, img_id in enumerate(batch_img_ids):
            # Store the original (non-augmented) embedding for this image.
            shard_data[img_id] = batch_orig[i]
            # Store each augmented variant embedding under a key like "imgid_aug0", "imgid_aug1", etc.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Map the j-th augmented embedding of the i-th image.
                shard_data[f"{img_id}_aug{j}"] = batch_aug[j][i]

        # Build a temporary filename for atomic save (write to tmp, then rename).
        shard_tmp = emb_dir / f".{shard_name}.tmp"
        # Save the shard data dictionary to the temporary file.
        torch.save(shard_data, shard_tmp)
        # Atomically rename the temporary file to the final shard filename.
        shard_tmp.replace(emb_dir / shard_name)

        # Update the index with each image id pointing to the current shard.
        for img_id in batch_img_ids:
            # Map the image id to the shard filename.
            index[img_id] = shard_name
        # Build a temporary path for the index file for atomic write.
        tmp_idx = index_path.with_suffix(".tmp")
        # Open the temporary index file for writing.
        with open(tmp_idx, "w") as f:
            # Serialize the index dictionary as JSON to the temporary file.
            json.dump(index, f)
        # Atomically replace the old index with the newly written one.
        tmp_idx.replace(index_path)

        # Clear the accumulated image ids for the next shard.
        batch_img_ids.clear()
        # Clear the accumulated original embeddings for the next shard.
        batch_orig.clear()
        # Clear each augmented embedding accumulator list for the next shard.
        for b in batch_aug:
            # Clear the list in place.
            b.clear()
        # Increment the shard index for the next flush.
        shard_idx += 1

    # Create a PrecomputeDataset wrapping the (possibly filtered) samples list.
    dataset = PrecomputeDataset(samples)
    # Create a DataLoader for batched embedding extraction.
    loader = DataLoader(
        # The PrecomputeDataset instance.
        dataset,
        # Batch size for embedding precomputation from config.
        batch_size=PRECOMPUTE_EMBED_BATCH,
        # Do not shuffle; order doesn't matter for precomputation.
        shuffle=False,
        # Number of subprocess workers for data loading.
        num_workers=PRECOMPUTE_NUM_WORKERS,
        # Use the embedding-specific collate function.
        collate_fn=_embed_collate,
        # Pin memory for faster GPU transfer.
        pin_memory=True,
        # Prefetch factor for data loading (loads 2 batches ahead).
        prefetch_factor=2,
        # Do not drop the last incomplete batch.
        drop_last=False,
    )

    # Compute number of views per sample: 1 original + N augmentations.
    VIEWS = 1 + AUGMENTATIONS_PER_IMAGE
    # Track the cumulative total number of processed samples.
    total = total_done
    # Iterate over batches from the dataloader with a progress bar.
    for ids, orig_batch, var_batch in tqdm(loader, total=len(loader)):
        # Get the number of actual samples in this batch (may differ for the last batch).
        B = len(ids)

        # Collect all tensors (original + variants) in interleaved order for a single forward pass.
        all_tensors = []
        # For each sample in the batch, append its original tensor and all augmented variants.
        for i in range(B):
            # Append the i-th original (base-transformed) tensor.
            all_tensors.append(orig_batch[i])
            # Append each of the augmented variant tensors for this sample.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Append the j-th augmented variant of the i-th sample.
                all_tensors.append(var_batch[i, j])

        # Stack all collected tensors into a single large batch and move to the device.
        all_tensor = torch.stack(all_tensors).to(device, non_blocking=True)
        # Run the vision encoder in inference mode (no gradients).
        with torch.inference_mode():
            # Forward pass through the vision model on the full stacked batch.
            vision_outputs = vision_model(pixel_values=all_tensor)
            # Apply visual projection to pooled outputs and move result to CPU.
            all_embs = visual_projection(vision_outputs.pooler_output).cpu()

        # Distribute the computed embeddings back to per-image accumulators.
        for i in range(B):
            # The original embedding for the i-th image is at position i * VIEWS in the flat result.
            batch_orig.append(all_embs[i * VIEWS])
            # Distribute each augmented variant embedding to the corresponding augmentation slot.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Append the j-th augmented embedding of the i-th image to the j-th accumulator.
                batch_aug[j].append(all_embs[i * VIEWS + 1 + j])

        # Add the current batch's image ids to the shard accumulator.
        batch_img_ids.extend(ids)
        # Update the total count of processed samples.
        total += B
        # If the accumulated number of image ids meets or exceeds the shard size, flush to disk.
        if len(batch_img_ids) >= shard_size:
            # Flush the current batch to a shard file.
            flush_shard()
            # Print a progress message showing shard count and total samples.
            print(f"  Wrote {shard_idx} shards ({total} samples)")

    # Flush any remaining accumulated data after the loop ends (final partial shard).
    flush_shard()

    # Count the number of unique shards written.
    total_shards = len(set(index.values()))
    # Print a completion message with the output directory and shard count.
    print(f"Embeddings saved to {emb_dir} ({total_shards} shard(s))")
    # Print the location of the JSON index file.
    print(f"Index saved to {index_path}")
    # Return the embedding index dictionary.
    return index


# Precompute road and vegetation features for all images, saved in sharded .pt files with a JSON index.
def precompute_features(
    # Path to the data directory containing images and metadata; defaults to SUBSET_DIR.
    data_dir: Optional[Path] = None,
    # Device to run inference on ("cuda" or "cpu"); defaults to "cuda".
    device: str = "cuda",
    # Type of road detection model ("grounding_dino" or "yolo_world"); defaults to "grounding_dino".
    road_model: str = "grounding_dino",
    # Type of vegetation model ("ram++" or "clip"); defaults to "clip".
    veg_model: str = "clip",
    # Maximum number of samples per feature shard file; defaults to SHARD_SIZE from config.
    shard_size: int = SHARD_SIZE,
):
    # Fall back to CPU if CUDA is unavailable for the specified device.
    device = device if torch.cuda.is_available() else "cpu"
    # Lazily import the feature extractor factory to avoid loading heavy models until needed.
    from features import create_feature_extractors

    # Use the provided data directory or fall back to the configured subset directory.
    data_dir = data_dir or SUBSET_DIR
    # Get the directory where feature shards will be stored.
    feat_dir = FEATURE_DIR
    # Create the feature output directory if it does not already exist.
    feat_dir.mkdir(parents=True, exist_ok=True)
    # Build the path to the JSON index file that maps image ids to feature shard filenames.
    index_path = feat_dir / "feature_index.json"

    # Create the feature extractor object with the specified models.
    extractor = create_feature_extractors(road_model=road_model, veg_model=veg_model, device=device)
    # Collect all (image_path, json_path) sample tuples from the data directory.
    samples = gather_samples(data_dir)

    # Check if a previous feature index file exists (resume support).
    if index_path.exists():
        # Open and read the existing JSON index file.
        with open(index_path) as f:
            # Parse the JSON to reconstruct the index dictionary.
            index = json.load(f)
        # Filter the index to only keep entries whose shard files still exist on disk.
        valid = {k: v for k, v in index.items() if (feat_dir / v).exists()}
        # Update the index to only contain valid entries.
        index = valid
        # Build a set of already-completed image ids for filtering.
        completed = set(index.keys())
        # Filter out samples whose features have already been precomputed.
        samples = [(p, j) for p, j in samples if p.stem not in completed]
        # Determine the next shard index based on how many unique shards already exist.
        shard_idx = len(set(index.values()))
        # Track the total number of completed samples for progress reporting.
        total_done = len(completed)
        # Print a resume status message.
        print(f"Resuming features: {total_done} completed, {len(samples)} remaining")
    # Otherwise, initialize a fresh precomputation run.
    else:
        # Start with an empty index dictionary.
        index = {}
        # Start shard numbering at 0.
        shard_idx = 0
        # No samples have been completed yet.
        total_done = 0
        # Print an initial status message with the selected model names.
        print(f"Precomputing features ({road_model} + {veg_model}) for {len(samples)} images...")

    # If no samples remain (all already precomputed), return early.
    if not samples:
        # Print a completion message.
        print("All features already precomputed.")
        # Return the existing index.
        return index

    # Accumulator list for image identifiers in the current shard.
    batch_img_ids = []
    # Accumulator list for feature dictionaries in the current shard.
    batch_feats = []
    # Accumulator for augmented feature dicts: one list per augmentation slot, each collecting per-image feature dicts.
    batch_aug_feats = [[] for _ in range(AUGMENTATIONS_PER_IMAGE)]

    # Inner function to flush the current in-memory batch to a feature shard file and update the index.
    def flush_shard():
        # Declare shard_idx and index as nonlocal so they can be modified from the enclosing scope.
        nonlocal shard_idx, index
        # If there is nothing accumulated, return without writing.
        if not batch_img_ids:
            # Early exit: no data to flush.
            return
        # Build the shard filename using zero-padded shard index.
        shard_name = f"feat_shard_{shard_idx:04d}.pt"
        # Initialize an empty dictionary to hold all features for this shard.
        shard_data = {}
        # Iterate over each image in the current shard batch.
        for i, img_id in enumerate(batch_img_ids):
            # Store the original (non-augmented) feature dict for this image.
            shard_data[img_id] = batch_feats[i]
            # Store each augmented variant feature dict under a key like "imgid_aug0", "imgid_aug1", etc.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Map the j-th augmented features of the i-th image.
                shard_data[f"{img_id}_aug{j}"] = batch_aug_feats[j][i]

        # Build a temporary filename for atomic save (write to tmp, then rename).
        shard_tmp = feat_dir / f".{shard_name}.tmp"
        # Save the shard data dictionary to the temporary file.
        torch.save(shard_data, shard_tmp)
        # Atomically rename the temporary file to the final shard filename.
        shard_tmp.replace(feat_dir / shard_name)

        # Update the index with each image id pointing to the current shard.
        for img_id in batch_img_ids:
            # Map the image id to the shard filename.
            index[img_id] = shard_name
        # Build a temporary path for the index file for atomic write.
        tmp_idx = index_path.with_suffix(".tmp")
        # Open the temporary index file for writing.
        with open(tmp_idx, "w") as f:
            # Serialize the index dictionary as JSON to the temporary file.
            json.dump(index, f)
        # Atomically replace the old index with the newly written one.
        tmp_idx.replace(index_path)

        # Clear the accumulated image ids for the next shard.
        batch_img_ids.clear()
        # Clear the accumulated feature dicts for the next shard.
        batch_feats.clear()
        # Clear each augmented feature accumulator list for the next shard.
        for b in batch_aug_feats:
            # Clear the list in place.
            b.clear()
        # Increment the shard index for the next flush.
        shard_idx += 1

    # Create a PrecomputeDataset wrapping the (possibly filtered) samples list.
    dataset = PrecomputeDataset(samples)
    # Create a DataLoader for batched feature extraction.
    loader = DataLoader(
        # The PrecomputeDataset instance.
        dataset,
        # Batch size for feature precomputation from config.
        batch_size=PRECOMPUTE_FEATURE_BATCH,
        # Do not shuffle; order doesn't matter for precomputation.
        shuffle=False,
        # Number of subprocess workers for data loading.
        num_workers=PRECOMPUTE_NUM_WORKERS,
        # Use the feature-specific collate function.
        collate_fn=_feature_collate,
        # Pin memory for faster GPU transfer.
        pin_memory=True,
        # Prefetch factor for data loading (loads 2 batches ahead).
        prefetch_factor=2,
        # Do not drop the last incomplete batch.
        drop_last=False,
    )

    # Compute number of views per sample: 1 original + N augmentations.
    VIEWS = 1 + AUGMENTATIONS_PER_IMAGE
    # Track the cumulative total number of processed samples.
    total = total_done
    # Iterate over batches from the dataloader with a progress bar.
    for ids, pil_images, var_batch in tqdm(loader, total=len(loader)):
        # Get the number of actual samples in this batch (may differ for the last batch).
        B = len(ids)

        # Start the inputs list with the original PIL images for each sample in the batch.
        all_inputs = list(pil_images)
        # For each sample, append its augmented variant tensors to the inputs list.
        for i in range(B):
            # Append each augmented variant tensor for this sample.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Append the j-th augmented variant of the i-th sample to the flat inputs list.
                all_inputs.append(var_batch[i, j])

        # Run the feature extractor on all images (originals + variants) in one batch call.
        all_features = extractor.extract_batch(all_inputs)

        # Distribute the computed feature dicts back to per-image accumulators.
        for i in range(B):
            # Append the original (non-augmented) feature dict for the i-th image.
            batch_feats.append({
                # Extract the road features for the i-th original.
                "road_features": all_features["road_features"][i * VIEWS],
                # Extract the vegetation features for the i-th original.
                "veg_features": all_features["veg_features"][i * VIEWS],
            })
            # Distribute each augmented variant's feature dict to the corresponding augmentation slot.
            for j in range(AUGMENTATIONS_PER_IMAGE):
                # Compute the flat index of this augmented variant in the results.
                idx = i * VIEWS + 1 + j
                # Append the j-th augmented feature dict to the j-th augmented accumulator.
                batch_aug_feats[j].append({
                    # Road features for the j-th augmented variant of the i-th image.
                    "road_features": all_features["road_features"][idx],
                    # Vegetation features for the j-th augmented variant of the i-th image.
                    "veg_features": all_features["veg_features"][idx],
                })

        # Add the current batch's image ids to the shard accumulator.
        batch_img_ids.extend(ids)
        # Update the total count of processed samples.
        total += B
        # If the accumulated number of image ids meets or exceeds the shard size, flush to disk.
        if len(batch_img_ids) >= shard_size:
            # Flush the current batch to a shard file.
            flush_shard()
            # Print a progress message showing shard count and total samples.
            print(f"  Wrote {shard_idx} shards ({total} samples)")

    # Flush any remaining accumulated data after the loop ends (final partial shard).
    flush_shard()

    # Count the number of unique shards written.
    total_shards = len(set(index.values()))
    # Print a completion message with the output directory and shard count.
    print(f"Features saved to {feat_dir} ({total_shards} shard(s))")
    # Print the location of the JSON index file.
    print(f"Index saved to {index_path}")
    # Return the feature index dictionary.
    return index


# Convenience function to run both embedding and feature precomputation sequentially.
def precompute_all(
    # Path to the data directory containing images and metadata; defaults to SUBSET_DIR.
    data_dir: Optional[Path] = None,
    # Device to run inference on ("cuda" or "cpu"); defaults to "cuda".
    device: str = "cuda",
    # Type of road detection model; defaults to "grounding_dino".
    road_model: str = "grounding_dino",
    # Type of vegetation model; defaults to "clip".
    veg_model: str = "clip",
    # Maximum number of samples per shard file; defaults to SHARD_SIZE from config.
    shard_size: int = SHARD_SIZE,
):
    # Fall back to CPU if CUDA is unavailable for the specified device.
    device = device if torch.cuda.is_available() else "cpu"
    # Print a separator banner for the embeddings phase.
    print("=" * 60)
    # Print the section header.
    print("PRECOMPUTING STREETCLIP EMBEDDINGS")
    # Print the closing separator.
    print("=" * 60)
    # Run the embedding precomputation.
    precompute_embeddings(data_dir=data_dir, device=device, shard_size=shard_size)

    # Print a blank line for visual separation between phases.
    print()
    # Print a separator banner for the features phase.
    print("=" * 60)
    # Print the section header.
    print("PRECOMPUTING ROAD + VEGETATION FEATURES")
    # Print the closing separator.
    print("=" * 60)
    # Run the feature precomputation with the specified road and vegetation model types.
    precompute_features(data_dir=data_dir, device=device, road_model=road_model, veg_model=veg_model, shard_size=shard_size)

    # Print a blank line for visual separation.
    print()
    # Print a separator banner for the completion message.
    print("=" * 60)
    # Print the completion header.
    print("PRECOMPUTATION COMPLETE")
    # Print the closing separator.
    print("=" * 60)


# Entry point when the script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Import argparse for command-line argument parsing.
    import argparse

    # Create an argument parser with a description.
    parser = argparse.ArgumentParser()
    # Add an optional argument to select which precomputation mode to run.
    parser.add_argument("--mode", choices=["embeddings", "features", "all"], default="all")
    # Add an optional argument to select the road detection model type.
    parser.add_argument("--road-model", choices=["grounding_dino", "yolo_world"], default="grounding_dino")
    # Add an optional argument to select the vegetation model type.
    parser.add_argument("--veg-model", choices=["ram++", "clip"], default="clip")
    # Add an optional argument for the device; auto-detect CUDA availability for default.
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Add an optional argument for a custom data directory path.
    parser.add_argument("--data-dir", default=None)
    # Add an optional argument to override the shard size with a custom integer value.
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    # Parse the command-line arguments into an args namespace.
    args = parser.parse_args()

    # If the mode is "embeddings", run only embedding precomputation.
    if args.mode == "embeddings":
        # Call precompute_embeddings with the parsed arguments.
        precompute_embeddings(data_dir=args.data_dir, device=args.device, shard_size=args.shard_size)
    # If the mode is "features", run only feature precomputation.
    elif args.mode == "features":
        # Call precompute_features with the parsed arguments (line continuation for readability).
        precompute_features(
            # Pass the data directory argument.
            data_dir=args.data_dir,
            # Pass the device argument.
            device=args.device,
            # Pass the road model type argument.
            road_model=args.road_model,
            # Pass the vegetation model type argument.
            veg_model=args.veg_model,
            # Pass the shard size argument.
            shard_size=args.shard_size,
        )
    # Otherwise (mode is "all"), run both embedding and feature precomputation.
    else:
        # Call precompute_all with the parsed arguments (line continuation for readability).
        precompute_all(
            # Pass the data directory argument.
            data_dir=args.data_dir,
            # Pass the device argument.
            device=args.device,
            # Pass the road model type argument.
            road_model=args.road_model,
            # Pass the vegetation model type argument.
            veg_model=args.veg_model,
            # Pass the shard size argument.
            shard_size=args.shard_size,
        )
