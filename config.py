import os
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
DATA_DIR = DATASET_DIR / "data"
DATA_MAPPED_DIR = DATASET_DIR / "data_mapped"
OUTPUT_DIR = ROOT / "output"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
FEATURE_DIR = OUTPUT_DIR / "features"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLONKIT_DIR = ROOT / "plonkit_data"

for d in [OUTPUT_DIR, EMBEDDING_DIR, FEATURE_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

USE_SUBSET = False
SUBSET_DIR = DATA_MAPPED_DIR if USE_SUBSET else DATA_DIR  # fallback
NUM_IMAGES_MAX = 5000 if USE_SUBSET else None

STREETCLIP_MODEL = "geolocal/StreetCLIP"
STREETCLIP_EMBED_DIM = 768
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
YOLO_WORLD_MODEL = str(ROOT / "models" / "yolov8m-worldv2.pt")
RAM_PLUS_MODEL = "xinyu1205/recognize-anything-plus-model"

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

AUGMENTATIONS_PER_IMAGE = 2
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

TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15

BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

PROJECTION_HIDDEN_DIM = 512
FUSION_OUTPUT_DIM = 512
TEMPERATURE = 0.07

OBJ_FEATURE_DIM = len(ROAD_CATEGORIES)
VEG_FEATURE_DIM = len(VEGETATION_CATEGORIES)
OBJ_PROJECTION_DIM = 256
VEG_PROJECTION_DIM = 256

NUM_WORKERS = 4
PRECOMPUTE_BATCH_SIZE = 16
PRECOMPUTE_EMBED_BATCH = 16
PRECOMPUTE_FEATURE_BATCH = 4
PRECOMPUTE_NUM_WORKERS = 4
SHARD_SIZE = 10000
SEED = 42

USE_AMP = True
AMP_DTYPE = "float16"

CONTRASTIVE_TARGET = "country"
