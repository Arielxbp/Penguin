# Import hashlib to generate deterministic hash keys for cache file names.
import hashlib
# Import json for parsing JSON metadata files associated with images.
import json
# Import Path from pathlib for cross-platform filesystem path handling.
from pathlib import Path
# Import Optional for type annotations that allow None values.
from typing import Optional

# Import numpy for numerical array operations and tensor manipulation.
import numpy as np
# Import PyTorch for deep learning model inference and tensor computations.
import torch
# Import PIL Image for loading and preprocessing street-level photographs.
from PIL import Image
# Import tqdm for displaying progress bars during long-running loops.
from tqdm import tqdm

# Import configuration constants: checkpoint directory, feature dimensions, and output directory.
from config import (
    CHECKPOINT_DIR,
    OBJ_FEATURE_DIM,
    VEG_FEATURE_DIM,
    OUTPUT_DIR,
)
# Import dataset utilities: country encoder, sample gathering, metadata parsing, transforms, and subset directory.
from dataset import (
    CountryEncoder,
    gather_samples,
    parse_metadata,
    BASE_TRANSFORM,
    SUBSET_DIR,
)
# Import the factory function that creates road and vegetation feature extractors.
from features import create_feature_extractors
# Import the StreetCLIPFusion model class for cross-modal geolocation.
from model import StreetCLIPFusion

# Define the directory path where cached country centroid tensors are stored.
CENTROID_CACHE_DIR = OUTPUT_DIR / "centroids"
# Create the centroid cache directory (and any missing parents) if it does not already exist.
CENTROID_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _centroid_cache_path(data_dir: Path, checkpoint_path: str,
                         use_features: bool = False,
                         road_model: str = "grounding_dino",
                         veg_model: str = "clip") -> Path:
    # Build a unique string key from the data directory, checkpoint path, feature flag, and detector model names.
    key = f"{data_dir.resolve()}_{checkpoint_path}_feat{int(use_features)}_{road_model}_{veg_model}"
    # Compute a short MD5 hex digest (12 characters) of the key for a compact, deterministic filename.
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    # Append a "_features" suffix when feature extractors are used, otherwise use no suffix.
    suffix = "_features" if use_features else ""
    # Return the full cache file path inside the centroid cache directory.
    return CENTROID_CACHE_DIR / f"centroids_{h}{suffix}.pt"


def load_model(checkpoint_path: Optional[str] = None, device: str = "cuda"):
    # Fall back to CPU if the requested device (e.g., CUDA) is not available.
    device = device if torch.cuda.is_available() else "cpu"
    # Instantiate the StreetCLIPFusion model with unfrozen backbone for full forward pass.
    model = StreetCLIPFusion(freeze_backbone=False)
    # If an explicit checkpoint path was provided, load its state dictionary.
    if checkpoint_path:
        # Load the saved model weights from the checkpoint file (weights only for security).
        state = torch.load(checkpoint_path, weights_only=True)
    # Otherwise attempt to locate and load the default best checkpoint.
    else:
        # Construct the path to the default best model checkpoint.
        best_path = CHECKPOINT_DIR / "best_model.pt"
        # Store the string form of the checkpoint path for later return.
        checkpoint_path = str(best_path)
        # If the best model checkpoint file exists, load its state dictionary.
        if best_path.exists():
            # Load the saved model weights from the default best checkpoint file.
            state = torch.load(best_path, weights_only=True)
        # Raise an error if no checkpoint can be found in the expected directory.
        else:
            raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_DIR}")
    # Load the state dictionary into the model, ignoring any missing or unexpected keys.
    model.load_state_dict(state, strict=False)
    # Move the model to the target device (GPU or CPU).
    model = model.to(device)
    # Set the model to evaluation mode, disabling dropout and batchnorm updates.
    model.eval()
    # Return the loaded model and the checkpoint path string that was used.
    return model, checkpoint_path


# Disable gradient computation for the entire centroid-building function to save memory.
@torch.no_grad()
def compute_country_centroids(
    model,
    data_dir: Path,
    country_encoder: CountryEncoder,
    device: str = "cuda",
    max_per_country: int = 200,
    force_refresh: bool = False,
    checkpoint_path: str = "",
    use_features: bool = True,
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
):
    # Fall back to CPU if CUDA is not available on this machine.
    device = device if torch.cuda.is_available() else "cpu"
    # Determine the cache file path for these parameters (data, checkpoint, feature config).
    cache_path = _centroid_cache_path(data_dir, checkpoint_path,
                                      use_features=use_features,
                                      road_model=road_model,
                                      veg_model=veg_model)

    # If a cached centroids file exists and we are not forcing a refresh, load it.
    if cache_path.exists() and not force_refresh:
        # Load the centroid dictionary from the cached .pt file on disk.
        centroids = torch.load(cache_path, weights_only=True)
        # Squeeze any centroid tensor that is 2D with a leading dimension of 1 down to 1D.
        for k in centroids:
            # Check if the tensor is 2D with batch size 1 and needs squeezing.
            if centroids[k].ndim == 2 and centroids[k].shape[0] == 1:
                # Remove the unnecessary first dimension to obtain a flat feature vector.
                centroids[k] = centroids[k].squeeze(0)
        # Collect the set of country names that were found in the cached centroids.
        cached = {c for c in country_encoder.country_list if c in centroids}
        # Build a list of country names that still need their centroids computed.
        missing = [c for c in country_encoder.country_list if c not in centroids]
        # If no countries are missing, the cache is fully complete — return immediately.
        if not missing:
            # Notify the user that all centroids were loaded from the cache file.
            print(f"Loaded {len(centroids)} centroids from cache: {cache_path}")
            return centroids
        # Otherwise, report how many were loaded and how many still need computation.
        print(f"Loaded {len(centroids)} cached centroids, {len(missing)} new countries to compute")
    # If no cache file exists or refresh is forced, start with an empty centroids dictionary.
    else:
        centroids = {}

    # Attempt to create the feature extractor only if use_features is enabled.
    extractor = None
    if use_features:
        try:
            # Instantiate the road and vegetation feature extractors with the chosen models.
            extractor = create_feature_extractors(
                road_model=road_model, veg_model=veg_model, device=device
            )
        # If extractor creation fails (e.g., model not available), log a warning and continue without features.
        except Exception:
            print("WARNING: feature extractor creation failed, falling back to zeros")
            extractor = None

    # Gather all (image_path, json_path) sample pairs from the data directory.
    samples = gather_samples(data_dir)
    # Initialize an empty list for every country to accumulate image embeddings.
    country_embs = {c: [] for c in country_encoder.country_list}
    # For any country that was already loaded from cache, reset its embedding list to empty.
    for c in centroids:
        # Re-initialize the embedding accumulator for this country so we don't recompute cached data.
        country_embs[c] = []

    # Iterate over all gathered samples, showing a progress bar labeled "Building centroids".
    for img_path, json_path in tqdm(samples, desc="Building centroids"):
        # Parse the metadata JSON file to extract structured information (country, coordinates, etc.).
        meta = parse_metadata(json_path)
        # Extract the country name string from the parsed metadata.
        country = meta["country_name"]
        # Skip this sample if the country is not in our known list of countries.
        if country not in country_embs:
            continue
        # Skip this sample if we have already accumulated the maximum allowed images for this country.
        if len(country_embs[country]) >= max_per_country:
            continue
        # Open the image file and ensure it is in 3-channel RGB format.
        image = Image.open(img_path).convert("RGB")
        # Apply the base transform (resize, normalize, etc.), add a batch dimension, and move to the device.
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)

        # If a feature extractor is available, extract road and vegetation feature tensors.
        if extractor is not None:
            # Run the extractor on the raw PIL image to get road and vegetation feature vectors.
            features = extractor.extract(image)
            # Get the road features, add a batch dimension, and move to the target device.
            road_f = features["road_features"].unsqueeze(0).to(device)
            # Get the vegetation features, add a batch dimension, and move to the target device.
            veg_f = features["veg_features"].unsqueeze(0).to(device)
        # Otherwise, fall back to zero-filled tensors with the expected feature dimensions.
        else:
            # Create a zero-filled road feature tensor with the correct dimension on the target device.
            road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
            # Create a zero-filled vegetation feature tensor with the correct dimension on the target device.
            veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)

        # Pass pixel values and feature vectors through the model to get a fused embedding.
        emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)
        # Append the embedding (moved to CPU for memory efficiency) to the country's accumulator list.
        country_embs[country].append(emb.cpu())

    # Compute the mean embedding (centroid) for each country that has accumulated embeddings.
    for country, embs in country_embs.items():
        # Only compute a centroid if the country has at least one embedding.
        if embs:
                # Stack the list of embeddings, average across the batch dimension, and remove the extra dim.
                centroids[country] = torch.stack(embs).mean(dim=0).squeeze(0)

    # Save the computed centroids dictionary to the cache file on disk.
    torch.save(centroids, cache_path)
    # Print a confirmation message showing how many centroids were saved and the cache file path.
    print(f"Saved {len(centroids)} centroids to {cache_path}")
    # Return the dictionary mapping country names to their centroid embedding vectors.
    return centroids


# Disable gradient computation for the evaluation function to save memory and speed up inference.
@torch.no_grad()
def evaluate(
    model,
    data_dir: Path,
    country_encoder: CountryEncoder,
    centroids: dict,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    use_features: bool = True,
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
):
    # Fall back to CPU if the requested compute device (e.g., CUDA) is unavailable.
    device = device if torch.cuda.is_available() else "cpu"
    # Gather all (image_path, json_path) sample pairs from the evaluation data directory.
    samples = gather_samples(data_dir)
    # If a maximum number of samples was specified, truncate the sample list.
    if max_samples:
        # Keep only the first max_samples entries for a faster partial evaluation.
        samples = samples[:max_samples]

    # Attempt to create the feature extractor only when use_features is enabled.
    extractor = None
    if use_features:
        try:
            # Instantiate road and vegetation feature extractors with the chosen model backbones.
            extractor = create_feature_extractors(
                road_model=road_model, veg_model=veg_model, device=device
            )
        # If extractor creation fails, fall back to zero-valued feature tensors.
        except Exception:
            extractor = None

    # Initialize counter for correct top-1 predictions.
    correct_1 = 0
    # Initialize counter for correct top-5 predictions (true label in top 5).
    correct_5 = 0
    # Initialize counter for the total number of evaluated samples.
    total = 0

    # Stack all country centroids into a single matrix on the target device for batched similarity computation.
    centroid_matrix = torch.stack([centroids[c] for c in country_encoder.country_list]).to(device)

    # Iterate over evaluation samples, displaying a progress bar labeled "Evaluating".
    for img_path, json_path in tqdm(samples, desc="Evaluating"):
        # Parse the metadata JSON to extract the ground-truth country label.
        meta = parse_metadata(json_path)
        # Get the true country name for this image from the parsed metadata.
        true_country = meta["country_name"]
        # Skip samples whose country was not included in the computed centroids.
        if true_country not in centroids:
            continue

        # Open the image file and convert it to 3-channel RGB format.
        image = Image.open(img_path).convert("RGB")
        # Apply the base transform, add a batch dimension, and move the tensor to the target device.
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)

        # If a feature extractor is available, compute road and vegetation feature vectors.
        if extractor is not None:
            # Run the extractor on the raw PIL image to obtain road and vegetation feature tensors.
            features = extractor.extract(image)
            # Extract road features, add a batch dimension, and move to the target device.
            road_f = features["road_features"].unsqueeze(0).to(device)
            # Extract vegetation features, add a batch dimension, and move to the target device.
            veg_f = features["veg_features"].unsqueeze(0).to(device)
        # Otherwise, create zero-filled placeholder tensors with the expected feature dimensions.
        else:
            # Create a zero road feature tensor with the correct output dimension on the target device.
            road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
            # Create a zero vegetation feature tensor with the correct output dimension on the target device.
            veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)

        # Forward pass: fuse pixel values and feature vectors through the model to produce an embedding.
        emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)
        # Compute cosine-like similarity between the image embedding and all country centroids.
        sim = torch.matmul(emb, centroid_matrix.T).squeeze(0)
        # Find the indices of the top 5 most similar centroids in descending order.
        top5_indices = sim.argsort(descending=True)[:5].cpu().numpy()
        # Map the top-5 centroid indices back to country name strings.
        top5_countries = [country_encoder.country_list[i] for i in top5_indices]

        # Increment the top-1 accuracy counter if the best match equals the ground truth.
        if top5_countries[0] == true_country:
            correct_1 += 1
        # Increment the top-5 accuracy counter if the ground truth appears anywhere in the top 5.
        if true_country in top5_countries:
            correct_5 += 1
        # Increment the total number of evaluated samples.
        total += 1

    # Print the total number of samples that were successfully evaluated.
    print(f"Samples evaluated: {total}")
    # Print the top-1 accuracy as a percentage of correct first-guess predictions.
    print(f"Top-1 accuracy: {correct_1/total*100:.2f}%")
    # Print the top-5 accuracy as a percentage of samples where the truth was in the top 5.
    print(f"Top-5 accuracy: {correct_5/total*100:.2f}%")
    # Return the raw top-1 and top-5 accuracy ratios (floats between 0 and 1).
    return correct_1 / total, correct_5 / total


# Disable gradient computation for single-image prediction to save memory.
@torch.no_grad()
def predict_single(
    model,
    image_path: str,
    country_encoder: CountryEncoder,
    centroids: dict,
    device: str = "cuda",
    top_k: int = 5,
    use_features: bool = True,
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
):
    # Fall back to CPU if CUDA is not available on this machine.
    device = device if torch.cuda.is_available() else "cpu"
    # Open the image from disk and ensure it is in 3-channel RGB format.
    image = Image.open(image_path).convert("RGB")
    # Apply the base transform, add a batch dimension of 1, and move to the target device.
    pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(device)

    # Attempt to create the feature extractor only if use_features is enabled.
    extractor = None
    if use_features:
        try:
            # Instantiate the road and vegetation feature extractors with the chosen backbones.
            extractor = create_feature_extractors(
                road_model=road_model, veg_model=veg_model, device=device
            )
        # Fall back to zero-filled feature tensors if extractor creation fails.
        except Exception:
            extractor = None

    # If the feature extractor was successfully created, extract features from the image.
    if extractor is not None:
        # Run the extractor on the raw PIL image to get road and vegetation feature vectors.
        features = extractor.extract(image)
        # Get the road features, add a batch dimension, and move to the target device.
        road_f = features["road_features"].unsqueeze(0).to(device)
        # Get the vegetation features, add a batch dimension, and move to the target device.
        veg_f = features["veg_features"].unsqueeze(0).to(device)
    # Otherwise, create zero-filled placeholder tensors with the correct feature dimensions.
    else:
        # Create a zero road feature tensor matching the expected output dimension.
        road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(device)
        # Create a zero vegetation feature tensor matching the expected output dimension.
        veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(device)

    # Forward pass: produce a fused embedding from pixel values and feature vectors.
    emb = model(pixel_values=pixel_values, road_features=road_f, veg_features=veg_f)

    # Stack all country centroid vectors into a single matrix on the target device.
    centroid_matrix = torch.stack([centroids[c] for c in country_encoder.country_list]).to(device)
    # Compute cosine-like similarity between the image embedding and all country centroids.
    sim = torch.matmul(emb, centroid_matrix.T).squeeze(0)
    # Get the indices of the top-k highest similarity scores in descending order.
    topk = sim.argsort(descending=True)[:top_k].cpu().numpy()
    # Build a list of (country_name, similarity_score) tuples for the top-k predictions.
    results = [(country_encoder.country_list[i], sim[i].item()) for i in topk]
    # Return the ranked list of top-k country predictions with their similarity scores.
    return results


# Entry point: runs when this script is executed directly (not imported as a module).
if __name__ == "__main__":
    # Import argparse for parsing command-line arguments.
    import argparse

    # Create an ArgumentParser object to define and parse CLI options.
    parser = argparse.ArgumentParser()
    # Add an optional argument for specifying a custom model checkpoint path.
    parser.add_argument("--checkpoint", default=None)
    # Add an optional argument for specifying a single image path (triggers single-image prediction mode).
    parser.add_argument("--image", default=None)
    # Add an optional argument for specifying the data directory containing evaluation images.
    parser.add_argument("--data-dir", default=None)
    # Add an optional argument for the compute device, defaulting to CUDA if available otherwise CPU.
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Add an optional argument to limit the number of evaluation samples (default 500).
    parser.add_argument("--max-samples", type=int, default=500)
    # Add an optional flag to disable feature extraction and use zero-filled feature tensors instead.
    parser.add_argument("--no-features", action="store_true")
    # Add an optional flag to force recomputation of country centroids, ignoring any cached results.
    parser.add_argument("--refresh-centroids", action="store_true")
    # Add an optional argument to select the road detection model (grounding_dino or yolo_world).
    parser.add_argument("--road-model", default="yolo_world",
                        choices=["grounding_dino", "yolo_world"])
    # Add an optional argument to select the vegetation detection model (clip or ram++).
    parser.add_argument("--veg-model", default="clip",
                        choices=["clip", "ram++"])
    # Parse the command-line arguments into the args namespace.
    args = parser.parse_args()

    # Use the provided data directory if given, otherwise fall back to the default subset directory.
    data_dir = Path(args.data_dir) if args.data_dir else SUBSET_DIR
    # Print the data directory being used for evaluation.
    print(f"Data dir: {data_dir}")

    # Load the model from the specified checkpoint (or default) and get the checkpoint path used.
    model, ckpt_path = load_model(args.checkpoint, args.device)
    # Instantiate the CountryEncoder using the list of countries found in the data directory.
    country_encoder = CountryEncoder(data_dir)
    # Print the number of countries recognized by the encoder.
    print(f"Countries: {len(country_encoder)}")

    # Compute (or load from cache) the centroid embedding for each country.
    centroids = compute_country_centroids(
        model, data_dir, country_encoder, args.device,
        force_refresh=args.refresh_centroids,
        checkpoint_path=ckpt_path,
        use_features=not args.no_features,
        road_model=args.road_model,
        veg_model=args.veg_model,
    )

    # If an explicit image path was provided, run single-image prediction mode.
    if args.image:
        # Run the single-image prediction and get the ranked list of country predictions.
        results = predict_single(
            model, args.image, country_encoder, centroids, args.device,
            use_features=not args.no_features,
            road_model=args.road_model,
            veg_model=args.veg_model,
        )
        # Print a header for the prediction results.
        print("\nPredictions:")
        # Iterate over the ranked prediction results and print each country with its score.
        for country, score in results:
            print(f"  {country}: {score:.4f}")
    # Otherwise, run standard evaluation over the entire dataset.
    else:
        # Run the evaluation loop and print top-1 / top-5 accuracy metrics.
        evaluate(
            model, data_dir, country_encoder, centroids, args.device,
            max_samples=args.max_samples,
            use_features=not args.no_features,
            road_model=args.road_model,
            veg_model=args.veg_model,
        )
