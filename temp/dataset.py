# Import the json module for parsing JSON metadata files
import json
# Import the random module for shuffling data and setting random seeds
import random
# Import Path from pathlib for convenient file path handling
from pathlib import Path
# Import Optional for type hints that allow None values
from typing import Optional

# Import PyTorch for tensor operations and dataset infrastructure
import torch
# Import torchvision transforms for image augmentation pipelines
import torchvision.transforms as T
# Import PIL.Image for opening and manipulating image files
from PIL import Image
# Import Dataset base class from PyTorch to build custom datasets
from torch.utils.data import Dataset

# Import configuration constants for dataset creation and augmentation
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

# Set the Python random seed globally for reproducibility of shuffles
random.seed(SEED)
# Set the PyTorch random seed globally for reproducibility of tensor operations
torch.manual_seed(SEED)


# Define a function to gather valid (png, json) sample pairs from a data directory
def gather_samples(data_dir: Path, limit: Optional[int] = None):
    # Find all PNG files matching the location_*.png pattern and sort them
    pngs = sorted(data_dir.glob("location_*.png"))
    # Initialize an empty list to hold (png_path, json_path) tuples
    samples = []
    # Iterate over each matching PNG file
    for png_path in pngs:
        # Construct the corresponding JSON path by replacing the .png extension
        json_path = png_path.with_suffix(".json")
        # Only include this sample if the JSON metadata file exists
        if json_path.exists():
            # Append the valid (image_path, metadata_path) pair to the list
            samples.append((png_path, json_path))
    # If a limit is specified, truncate the list to that many samples
    if limit:
        samples = samples[:limit]
    # Return the list of valid sample pairs
    return samples


# Define a class that holds the list of geo-located sample pairs
class GeoSamples:
    # Initialize the GeoSamples container with an optional custom data directory
    def __init__(self, data_dir: Optional[Path] = None):
        # Default to SUBSET_DIR if no data directory was provided
        data_dir = data_dir or SUBSET_DIR
        # Gather all valid (png, json) sample pairs up to NUM_IMAGES_MAX
        self.samples = gather_samples(data_dir, NUM_IMAGES_MAX)

    # Return the total number of samples in this container
    def __len__(self):
        return len(self.samples)


# Define a class that builds a mapping from country names to integer indices
class CountryEncoder:
    # Initialize the CountryEncoder with an optional custom data directory
    def __init__(self, data_dir: Optional[Path] = None):
        # Default to SUBSET_DIR if no data directory was provided
        data_dir = data_dir or SUBSET_DIR
        # Build a set of unique country names found in all JSON metadata files
        countries = set()
        # Iterate over every location JSON file in the given directory (sorted)
        for json_path in sorted(data_dir.glob("location_*.json")):
            # Try to load and parse the JSON metadata
            try:
                # Open the JSON file for reading
                with open(json_path) as f:
                    # Parse the JSON content into a Python dictionary
                    meta = json.load(f)
                # Ensure the parsed result is actually a dict to safely call .get()
                if isinstance(meta, dict):
                    # Extract the country name, defaulting to "Unknown" if missing
                    countries.add(meta.get("country_name", "Unknown"))
            # If JSON parsing fails or any other error occurs, treat as "Unknown"
            except (json.JSONDecodeError, Exception):
                # Add "Unknown" as a fallback country label
                countries.add("Unknown")
        # Sort the unique country names to create a stable, ordered list
        self.country_list = sorted(countries)
        # Build a dictionary mapping each country name to its integer index
        self.country_to_idx = {c: i for i, c in enumerate(self.country_list)}

    # Encode a country name string to its integer index, or -1 if not found
    def encode(self, country_name: str) -> int:
        return self.country_to_idx.get(country_name, -1)

    # Return the total number of unique countries known to the encoder
    def __len__(self):
        return len(self.country_list)


# Define a helper to safely parse a JSON metadata file into a dict of relevant fields
def parse_metadata(json_path: Path):
    # Attempt to load and parse the JSON file
    try:
        # Open the JSON file for reading
        with open(json_path) as f:
            # Parse the JSON content into a dictionary
            meta = json.load(f)
        # If the parsed result is not a dict, fall back to an empty dict
        if not isinstance(meta, dict):
            meta = {}
    # If JSON parsing fails or any other exception occurs, use an empty dict
    except (json.JSONDecodeError, Exception):
        meta = {}
    # Return a normalized dictionary with coordinates, country name, code, and regions
    return {
        "coordinates": meta.get("coordinates", [0.0, 0.0]),
        "country_name": meta.get("country_name", "Unknown"),
        "country_code": meta.get("country_code", "XX"),
        "regions": meta.get("regions", []),
    }


# Define a PIL-based augmentation transform pipeline used before conversion to tensor
PIL_AUG_TRANSFORM = T.Compose(
    [
        # Randomly crop and resize the image with configurable scale and aspect ratio
        T.RandomResizedCrop(
            336,
            scale=AUGMENTATION_CONFIG["random_crop_scale"],
            ratio=AUGMENTATION_CONFIG["random_crop_ratio"],
        ),
        # Randomly flip the image horizontally with a configurable probability
        T.RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip_prob"]),
        # Randomly adjust brightness, contrast, saturation, and hue
        T.ColorJitter(
            brightness=AUGMENTATION_CONFIG["color_jitter_brightness"],
            contrast=AUGMENTATION_CONFIG["color_jitter_contrast"],
            saturation=AUGMENTATION_CONFIG["color_jitter_saturation"],
            hue=AUGMENTATION_CONFIG["color_jitter_hue"],
        ),
        # Randomly rotate the image by a configurable number of degrees
        T.RandomRotation(degrees=AUGMENTATION_CONFIG["rotation_degrees"]),
    ]
)

# Define a tensor-based augmentation transform applied after conversion to tensor
TENSOR_AUG_TRANSFORM = T.Compose(
    [
        # Apply Gaussian blur with configurable kernel size and sigma range
        T.GaussianBlur(
            kernel_size=AUGMENTATION_CONFIG["blur_kernel_size"],
            sigma=AUGMENTATION_CONFIG["blur_sigma"],
        ),
    ]
)

# Define the base (non-augmented) transform: resize, convert to tensor, normalize
BASE_TRANSFORM = T.Compose(
    [
        # Resize the image to a fixed 336x336 square
        T.Resize((336, 336)),
        # Convert the PIL Image to a PyTorch tensor (scales pixel values to [0, 1])
        T.ToTensor(),
        # Normalize the tensor using CLIP-style ImageNet mean and std values
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


# Define a PyTorch Dataset that loads geo-tagged images with optional augmentation
class GeoSampleDataset(Dataset):
    # Initialize the dataset with sample pairs, a country encoder, and augmentation flag
    def __init__(
        self,
        samples: list,
        country_encoder: CountryEncoder,
        augment: bool = False,
    ):
        # Store the list of (image_path, json_path) sample pairs
        self.samples = samples
        # Store the CountryEncoder instance for label encoding
        self.country_encoder = country_encoder
        # Store whether to apply data augmentation during retrieval
        self.augment = augment

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Retrieve a single sample (image, metadata, and optional augmentations) by index
    def __getitem__(self, idx):
        # Unpack the sample pair at the given index into image path and JSON path
        img_path, json_path = self.samples[idx]
        # Parse the JSON metadata file into a normalized dictionary
        meta = parse_metadata(json_path)
        # Open the image file and convert it to RGB color space
        image = Image.open(img_path).convert("RGB")
        # Encode the country name string into an integer index
        country_idx = self.country_encoder.encode(meta["country_name"])
        # Initialize a list to hold augmented image tensors if augmentation is enabled
        augmented_images = []
        # Only generate augmented copies if the augmentation flag is True
        if self.augment:
            # Generate AUGMENTATIONS_PER_IMAGE different augmented views
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                # Apply the PIL-based augmentation transform (producing a PIL Image)
                aug_img = PIL_AUG_TRANSFORM(image)
                # Convert the augmented PIL Image to a tensor
                aug_tensor = T.ToTensor()(aug_img)
                # Apply the tensor-based augmentation (e.g., Gaussian blur)
                aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
                # Normalize the augmented tensor with CLIP-style mean and std
                aug_tensor = T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )(aug_tensor)
                # Append the augmented tensor to the list
                augmented_images.append(aug_tensor)
        # Apply the base (non-augmented) transform to the original image
        image = BASE_TRANSFORM(image)
        # Build the result dictionary with all relevant data fields
        result = {
            "image": image,
            "image_path": str(img_path),
            "json_path": str(json_path),
            "coordinates": torch.tensor(meta["coordinates"], dtype=torch.float32),
            "country_name": meta["country_name"],
            "country_code": meta["country_code"],
            "country_idx": torch.tensor(country_idx, dtype=torch.long),
        }
        # If augmented images were generated, stack them into a single tensor
        if augmented_images:
            result["augmented_images"] = torch.stack(augmented_images, dim=0)
        # Return the complete sample dictionary
        return result


# Define a convenience function to create a dataset and its country encoder
def create_dataset(data_dir: Optional[Path] = None, augment: bool = False):
    # Default to SUBSET_DIR if no custom data directory is given
    data_dir = data_dir or SUBSET_DIR
    # Gather all valid (png, json) sample pairs up to NUM_IMAGES_MAX
    samples = gather_samples(data_dir, NUM_IMAGES_MAX)
    # Create a CountryEncoder that builds the index from the same data directory
    country_encoder = CountryEncoder(data_dir)
    # Return the constructed dataset object and the country encoder
    return GeoSampleDataset(samples, country_encoder, augment=augment), country_encoder


# Define a function to split a dataset into training and validation index sets
def split_dataset(
    full_dataset: GeoSampleDataset,
    train_ratio: float = 0.85,
):
    # Get the total number of samples in the full dataset
    n = len(full_dataset)
    # Create a list of all sample indices from 0 to n-1
    indices = list(range(n))
    # Shuffle the indices randomly to create a random split
    random.shuffle(indices)
    # Determine the number of samples to allocate to the training set
    split = int(n * train_ratio)
    # Take the first 'split' shuffled indices for training
    train_indices = indices[:split]
    # Take the remaining indices for validation
    val_indices = indices[split:]
    # Return the training and validation index lists
    return train_indices, val_indices
