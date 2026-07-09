# Import the os module for operating system interactions
import os
# Import the Path class from pathlib for cross-platform path handling
from pathlib import Path

# Define the project root directory as the parent of this config file
ROOT = Path(__file__).parent
# Define the dataset directory path
DATASET_DIR = ROOT / "dataset"
# Define the raw data directory path
DATA_DIR = DATASET_DIR / "data"
# Define the mapped data directory path
DATA_MAPPED_DIR = DATASET_DIR / "data_mapped"
# Define the output directory path
OUTPUT_DIR = ROOT / "output"
# Define the embeddings output directory path
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
# Define the features output directory path
FEATURE_DIR = OUTPUT_DIR / "features"
# Define the checkpoints output directory path
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# Define the logs output directory path
LOG_DIR = OUTPUT_DIR / "logs"
# Define the plonkit data directory path
PLONKIT_DIR = ROOT / "plonkit_data"

# Create all required output directories if they don't exist
for d in [OUTPUT_DIR, EMBEDDING_DIR, FEATURE_DIR, CHECKPOINT_DIR, LOG_DIR]:
    # Create the directory with parents, ignoring if it already exists
    d.mkdir(parents=True, exist_ok=True)

# Flag to control whether to use a subset of data
USE_SUBSET = False
# Select the mapped data dir if using subset, otherwise use the raw data dir
SUBSET_DIR = DATA_MAPPED_DIR if USE_SUBSET else DATA_DIR  # fallback
# Limit the number of images when using subset, otherwise no limit
NUM_IMAGES_MAX = 5000 if USE_SUBSET else None

# Model identifier for the StreetCLIP pretrained model
STREETCLIP_MODEL = "geolocal/StreetCLIP"
# Embedding dimensionality for the StreetCLIP model
STREETCLIP_EMBED_DIM = 768
# Model identifier for the Grounding DINO object detection model
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
# Path to the YOLO-World model checkpoint file
YOLO_WORLD_MODEL = str(ROOT / "road_model" / "yolov8m-worldv2.pt")
# Model identifier for the RAM++ (Recognize Anything Plus) model
RAM_PLUS_MODEL = "xinyu1205/recognize-anything-plus-model"

# List of road-related object categories for feature extraction
ROAD_CATEGORIES = [
    "traffic light", "traffic sign", "street sign", "road signal",
    "bollard", "street pole", "utility pole", "electric pole",
    "telephone pole", "wooden pole", "concrete pole", "ladder pole",
    "round pole", "pole insulator", "pole crossbar", "pole support",
    "fire hydrant", "fence", "guard rail", "guardrail", "barrier",
    "curb", "sidewalk", "crosswalk", "road marking", "lane marking",
    "chevron sign", "warning sign", "speed limit sign", "stop sign",
    "yield sign", "directional sign", "highway sign",
    "manhole cover", "drainage grate", "bus stop", "bench",
    "trash can", "mailbox", "parking meter", "bike rack",
    "street lamp", "lamp post", "billboard", "overhead sign gantry",
    "tunnel", "bridge", "road line paint", "license plate",
    "road reflector", "delineator post", "concrete barrier",
    "crash barrier", "mile marker", "kilometer marker",
    "cobblestone pavement", "hexagonal tile road",
]

# List of vegetation-related categories for feature extraction
VEGETATION_CATEGORIES = [
    "tree", "palm tree", "pine tree", "oak tree", "birch tree",
    "eucalyptus", "araucaria", "parana pine", "coconut palm",
    "bush", "shrub", "hedge", "grass", "lawn", "meadow",
    "tall grass", "dry grass", "savanna",
    "flower", "wildflower", "crop field", "farmland", "vineyard",
    "sugarcane", "coffee plant", "soybean field", "banana plant",
    "forest", "jungle", "rainforest", "woodland", "orchard",
    "moss", "fern", "ivy", "cactus", "succulent",
    "bamboo", "reed", "mangrove", "swamp vegetation",
    "deciduous forest", "coniferous forest", "mixed forest",
    "tropical vegetation", "subtropical vegetation",
    "mediterranean vegetation", "alpine vegetation",
    "arid vegetation", "tundra vegetation",
    "cultivated land", "plantation", "greenhouse crops",
    "dead tree", "fallen leaves", "autumn foliage",
    "red soil", "dark soil", "sandy soil", "rocky terrain",
]

# Number of augmented copies to generate per original image
AUGMENTATIONS_PER_IMAGE = 2
# Configuration parameters for data augmentation transforms
AUGMENTATION_CONFIG = {
    "random_crop_scale": (0.7, 1.0),
    "random_crop_ratio": (0.9, 1.1),
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 10,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1,
    "blur_kernel_size": (3, 7),
    "blur_sigma": (0.1, 2.0),
}

# Proportion of data to use for training (85%)
TRAIN_SPLIT = 0.85
# Proportion of data to use for validation (15%)
VAL_SPLIT = 0.15

# Number of samples per training batch
BATCH_SIZE = 32
# Number of gradient accumulation steps before an optimizer step
GRADIENT_ACCUMULATION_STEPS = 4
# Total number of training epochs
NUM_EPOCHS = 10
# Initial learning rate for the optimizer
LEARNING_RATE = 1e-4
# Number of warmup steps for learning rate scheduling
WARMUP_STEPS = 500
# Weight decay coefficient for L2 regularization
WEIGHT_DECAY = 0.01
# Maximum gradient norm for gradient clipping
MAX_GRAD_NORM = 1.0

# Hidden dimension size for the projection head
PROJECTION_HIDDEN_DIM = 512
# Output dimension size for the fusion layer
FUSION_OUTPUT_DIM = 512
# Temperature parameter for contrastive loss scaling
TEMPERATURE = 0.07

# Dimensionality of road object feature vectors (number of road categories)
OBJ_FEATURE_DIM = len(ROAD_CATEGORIES)
# Dimensionality of vegetation feature vectors (number of vegetation categories)
VEG_FEATURE_DIM = len(VEGETATION_CATEGORIES)
# Output dimension for the road object projection layer
OBJ_PROJECTION_DIM = 256
# Output dimension for the vegetation projection layer
VEG_PROJECTION_DIM = 256

# Number of worker processes for data loading
NUM_WORKERS = 4
# Batch size for precomputation pipeline
PRECOMPUTE_BATCH_SIZE = 16
# Batch size for embedding precomputation
PRECOMPUTE_EMBED_BATCH = 16
# Batch size for feature precomputation
PRECOMPUTE_FEATURE_BATCH = 4
# Number of worker processes for precomputation
PRECOMPUTE_NUM_WORKERS = 4
# Number of samples per shard file for precomputed data
SHARD_SIZE = 10000
# Random seed for reproducibility
SEED = 42

# Flag to enable Automatic Mixed Precision training
USE_AMP = True
# Data type to use for Automatic Mixed Precision
AMP_DTYPE = "float16"

# Target label column for contrastive learning
CONTRASTIVE_TARGET = "country"
