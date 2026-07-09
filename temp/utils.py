# Import the json module for reading JSON metadata files
import json
# Import the random module for setting the Python random seed
import random
# Import the Path class from pathlib for cross-platform path handling
from pathlib import Path
# Import Optional type hint for indicating optional parameters
from typing import Optional

# Import numpy for numerical array operations
import numpy as np
# Import PyTorch for deep learning operations
import torch

# Import the SEED constant from the project configuration
from config import SEED


def set_seed(seed: int = SEED):
    # Set the Python built-in random seed for reproducibility
    random.seed(seed)
    # Set the numpy random seed for reproducibility
    np.random.seed(seed)
    # Set the PyTorch CPU random seed for reproducibility
    torch.manual_seed(seed)
    # Check if a CUDA-capable GPU is available
    if torch.cuda.is_available():
        # Set the random seed for all available CUDA devices
        torch.cuda.manual_seed_all(seed)
        # Force cuDNN to use deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        # Disable cuDNN auto-tuner to ensure deterministic behavior
        torch.backends.cudnn.benchmark = False


def get_device():
    # Check if a CUDA-capable GPU is available
    if torch.cuda.is_available():
        # Return "cuda" as the device string if GPU is available
        return "cuda"
    # Fall back to CPU if no GPU is available
    return "cpu"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True):
    # Count only parameters with requires_grad=True if trainable_only is set
    if trainable_only:
        # Return the sum of all trainable parameter element counts
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Otherwise return the total number of parameters in the model
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    # Format numbers >= 1 million with one decimal place and "M" suffix
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    # Format numbers >= 1 thousand with one decimal place and "K" suffix
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    # Return the number as a string if it's below 1000
    return str(n)


def dataset_stats(data_dir: Path):
    # Find all PNG files matching the location pattern in the data directory
    png_files = sorted(data_dir.glob("location_*.png"))
    # Initialize an empty set to collect unique country names
    countries = set()
    # Iterate over each PNG file to find its JSON metadata
    for png_path in png_files:
        # Derive the corresponding JSON path by replacing the file extension
        json_path = png_path.with_suffix(".json")
        # Only process the JSON if it exists on disk
        if json_path.exists():
            # Open the JSON metadata file for reading
            with open(json_path) as f:
                # Parse the JSON contents into a Python dictionary
                meta = json.load(f)
            # Extract the country name from the metadata, defaulting to "Unknown"
            countries.add(meta.get("country_name", "Unknown"))
    # Print the total number of images found
    print(f"Images: {len(png_files)}")
    # Print the number of unique countries found
    print(f"Countries: {len(countries)}")
    # Print the sorted list of all unique country names
    print(f"Countries list: {sorted(countries)}")
    # Return the image count and unique country count as a tuple
    return len(png_files), len(countries)
