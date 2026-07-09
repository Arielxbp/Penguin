#!/usr/bin/env python3
"""
OpenGuessr AI Player — plays openguessr.com using browser automation and the
Penguin geolocation model.

Usage:
    python play_openguessr.py                       # 5 rounds, AI mode, headed
    python play_openguessr.py --mode perfect        # exact coords from iframe
    python play_openguessr.py --rounds 10           # 10 rounds
    python play_openguessr.py --headless            # headless browser
    python play_openguessr.py --device cpu          # CPU inference
    python play_openguessr.py --benchmark           # benchmark vs raw StreetCLIP
"""

# Import the argparse module for parsing command-line arguments.
import argparse
# Import the asyncio module for writing asynchronous code using async/await.
import asyncio
# Import the json module for reading and writing JSON files.
import json
# Import the math module for mathematical functions (e.g., sin, cos, radians, log).
import math
# Import the random module for generating random numbers and choices.
import random
# Import the sys module for system-specific functions (manipulating sys.path).
import sys
# Import the time module for time-related functions (not heavily used here, but available).
import time
# Import datetime for generating timestamps in ISO format.
from datetime import datetime
# Import BytesIO from io for reading screenshots in memory as file-like objects.
from io import BytesIO
# Import Path from pathlib for convenient filesystem path handling.
from pathlib import Path
# Import parse_qs and urlparse from urllib.parse for parsing URL query parameters.
from urllib.parse import parse_qs, urlparse

# Import numpy for numerical operations, especially on coordinates.
import numpy as np
# Import torch (PyTorch) for model inference and tensor operations.
import torch
# Import Image from PIL (Pillow) for loading and converting image data.
from PIL import Image

# Compute the root directory as the parent directory of this script.
ROOT = Path(__file__).parent
# Insert the root directory at the front of sys.path so local modules can be imported.
sys.path.insert(0, str(ROOT))

# Import configuration constants from the project's config module.
from config import (
    CHECKPOINT_DIR,
    FUSION_OUTPUT_DIM,
    OBJ_FEATURE_DIM,
    VEG_FEATURE_DIM,
    OUTPUT_DIR,
    SUBSET_DIR,
)
# Import the StreetCLIPFusion model class from the model module.
from model import StreetCLIPFusion
# Import CountryEncoder (country label mapping) and BASE_TRANSFORM (image transforms).
from dataset import CountryEncoder, BASE_TRANSFORM
# Import centroid cache utility functions and directory from the eval module.
from eval import _centroid_cache_path, CENTROID_CACHE_DIR
# Import the create_feature_extractors factory function from features module.
from features import create_feature_extractors
# Import the set_seed utility to ensure reproducible random/pytorch results.
from utils import set_seed

# Seed all random number generators for reproducibility.
set_seed(42)
# Ensure the output directory exists, creating parent directories if needed.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Define subdirectories for play-related output.
PLAY_DIR = OUTPUT_DIR / "play"
# Directory to store per-session run logs.
RUNS_DIR = PLAY_DIR / "runs"
# Directory to store per-round screenshots and data.
ROUNDS_DIR = PLAY_DIR / "rounds"

# Base URL of the OpenGuessr game website.
GAME_URL = "https://openguessr.com"

# JavaScript snippet injected into the page to hide automation fingerprints.
ANTI_DETECT_SCRIPT = """
delete Object.getPrototypeOf(navigator).webdriver;
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
"""

# JavaScript snippet injected into the page to hook into Leaflet map instances.
LEAFLET_HOOK_SCRIPT = """
(() => {
    if (window.__ogLeafletHookInstalled) return;
    window.__ogLeafletHookInstalled = true;
    window.__ogMaps__ = [];
    const hook = (L) => {
        try {
            if (L && L.Map && L.Map.prototype && !L.Map.__ogHooked) {
                const orig = L.Map.prototype.initialize;
                L.Map.prototype.initialize = function() {
                    try { window.__ogMaps__.push(this); } catch (e) {}
                    return orig.apply(this, arguments);
                };
                L.Map.__ogHooked = true;
            }
        } catch (e) {}
    };
    let _L = window.L;
    if (_L) hook(_L);
    try {
        Object.defineProperty(window, 'L', {
            configurable: true,
            get() { return _L; },
            set(v) { _L = v; hook(v); },
        });
    } catch (e) {}
})();
"""


# Define a function to compute the great-circle distance between two lat/lng points.
def haversine_km(lat1, lng1, lat2, lng2):
    # Earth's mean radius in kilometers.
    R = 6371.0
    # Convert latitude difference from degrees to radians.
    dlat = math.radians(lat2 - lat1)
    # Convert longitude difference from degrees to radians.
    dlng = math.radians(lng2 - lng1)
    # Compute the haversine formula's 'a' term.
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    # Return the distance in kilometers using the haversine formula.
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Define a function to map a distance in km to an OpenGuessr-style score.
def distance_score(km):
    # If the distance is zero or negative, return the maximum score.
    if km <= 0:
        return 5000
    # Otherwise, score decays exponentially with distance; rounded to integer.
    return round(5000 * math.exp(-km / 2000))


# Define a helper to format a distance in km into a human-readable string.
def format_distance(km):
    # If less than 1 km, display in meters.
    if km < 1:
        return f"{km * 1000:.0f} m"
    # If less than 1000 km, display with one decimal in km.
    if km < 1000:
        return f"{km:.1f} km"
    # Otherwise, display in thousands of km (K km).
    return f"{km / 1000:.1f}K km"


# Return the current time as an ISO 8601 formatted string.
def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Penguin model wrapper
# ---------------------------------------------------------------------------

# Define the PenguinAI class that wraps the Penguin geolocation model.
class PenguinAI:
    # Initialize the AI with device, road model type, and vegetation model type.
    def __init__(self, device="cuda", road_model="yolo_world", veg_model="clip"):
        # Set device to "cuda" if available, otherwise fall back to "cpu".
        self.device = device if torch.cuda.is_available() else "cpu"
        # Placeholder for the StreetCLIPFusion model instance.
        self.model = None
        # Placeholder for the feature extractors (road + vegetation).
        self.extractor = None
        # Store the road feature extractor model choice.
        self.road_model = road_model
        # Store the vegetation feature extractor model choice.
        self.veg_model = veg_model
        # Dictionary mapping country name to cached centroid embedding.
        self.centroids = {}
        # List of country names from the dataset.
        self.country_list = []
        # Dictionary mapping country name to list of all its coordinate tuples.
        self._all_coords = {}
        # Dictionary mapping country name to its median coordinate (lat, lng).
        self.country_coords = {}
        # Flag to track whether the model has been loaded.
        self._loaded = False

    # Load the model, centroids, coordinate maps, and feature extractors.
    def load(self):
        # If already loaded, do nothing.
        if self._loaded:
            return
        # Print which device is being used for inference.
        print(f"Loading Penguin model on {self.device}...")
        # Instantiate the StreetCLIPFusion model with a frozen backbone.
        self.model = StreetCLIPFusion(
            freeze_backbone=True, fusion_output_dim=FUSION_OUTPUT_DIM,
        )
        # Build the path to the checkpoint file.
        ckpt = CHECKPOINT_DIR / "best_model.pt"
        # If the checkpoint file exists, load the model weights.
        if ckpt.exists():
            # Load the checkpoint state dict safely (weights_only) on CPU.
            state = torch.load(ckpt, weights_only=True, map_location="cpu")
            # Load the state dict into the model, ignoring missing/excess keys.
            self.model.load_state_dict(state, strict=False)
            # Print the checkpoint filename.
            print(f"  checkpoint  : {ckpt.name}")
        else:
            # Warn if no checkpoint file was found.
            print("  WARNING: no checkpoint found")

        # Move the model to the target device (GPU/CPU).
        self.model = self.model.to(self.device)
        # Set the model to evaluation mode (disables dropout, etc.).
        self.model.eval()

        # Determine the data directory for country labels and coordinates.
        data_dir = SUBSET_DIR
        # Load the list of country names from the dataset.
        self.country_list = CountryEncoder(data_dir).country_list

        # Build the path to the checkpoint file (as string for cache naming).
        ckpt_path = str(CHECKPOINT_DIR / "best_model.pt")
        # Get the path to the cached centroid embeddings for this model config.
        cache_path = _centroid_cache_path(
            data_dir, ckpt_path, use_features=True,
            road_model=self.road_model, veg_model=self.veg_model,
        )
        # If the centroid cache file exists, load it.
        if cache_path.exists():
            # Load the centroids dictionary safely on CPU.
            self.centroids = torch.load(cache_path, weights_only=True,
                                        map_location="cpu")
            # If any centroid tensor has shape (1, D), squeeze the batch dim.
            for k in self.centroids:
                if (self.centroids[k].ndim == 2
                        and self.centroids[k].shape[0] == 1):
                    self.centroids[k] = self.centroids[k].squeeze(0)
            # Print the number of country centroids loaded and the cache file name.
            print(f"  centroids   : {len(self.centroids)} countries "
                  f"({cache_path.name})")
        else:
            # Warn if the centroid cache file was not found.
            print(f"  centroids   : not found ({cache_path.name})")

        # Build the coordinate lookup map from the dataset's location JSONs.
        self._build_coord_map(data_dir)
        # Print the number of countries with known coordinates.
        print(f"  coordinates : {len(self.country_coords)} countries")

        # Attempt to create the road and vegetation feature extractors.
        try:
            # Create feature extractors for the chosen road and veg models.
            self.extractor = create_feature_extractors(
                road_model=self.road_model,
                veg_model=self.veg_model,
                device=self.device,
            )
            # Print the feature extractor model names.
            print(f"  features    : {self.road_model} + {self.veg_model}")
        except Exception as e:
            # If feature extractors fail to load, warn and leave as None.
            print(f"  features    : unavailable ({e})")
            self.extractor = None

        # Mark the model as loaded.
        self._loaded = True

    # Build a mapping from country name to its coordinates from JSON files.
    def _build_coord_map(self, data_dir):
        # Dictionary to hold country name -> list of (lat, lng) tuples.
        coords = {}
        # Iterate over all location JSON files in sorted order.
        for jf in sorted(data_dir.glob("location_*.json")):
            try:
                # Open and parse the JSON file.
                with open(jf) as f:
                    d = json.load(f)
                # Skip entries that are not dictionaries.
                if not isinstance(d, dict):
                    continue
                # Extract country name, defaulting to "Unknown".
                c = d.get("country_name", "Unknown")
                # Extract coordinates, defaulting to [0.0, 0.0].
                ll = d.get("coordinates", [0.0, 0.0])
                # Skip entries without exactly two coordinate values.
                if len(ll) != 2:
                    continue
                # Convert coordinate values to floats.
                lat, lng = float(ll[0]), float(ll[1])
                # Initialize the list for this country if not present.
                if c not in coords:
                    coords[c] = []
                # Append the coordinate tuple to the country's list.
                coords[c].append((lat, lng))
            except (json.JSONDecodeError, Exception):
                # Skip any file that cannot be parsed.
                continue
        # Store the raw coordinate map.
        self._all_coords = coords
        # Compute the median coordinate for each country and store it.
        self.country_coords = {
            c: (np.median([p[0] for p in pts]),
                np.median([p[1] for p in pts]))
            for c, pts in coords.items()
        }

    # Run inference on an image, returning top-k country predictions.
    @torch.inference_mode()
    def predict(self, image, top_k=5):
        # Apply base transforms, add batch dim, and move to device.
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(self.device)

        # If feature extractors are available, extract road and veg features.
        if self.extractor is not None:
            # Extract per-modality features from the PIL image.
            features = self.extractor.extract(image)
            # Extract road features, add batch dim, and move to device.
            road_f = features["road_features"].unsqueeze(0).to(self.device)
            # Extract vegetation features, add batch dim, and move to device.
            veg_f = features["veg_features"].unsqueeze(0).to(self.device)
        else:
            # If no extractor, use zero tensors as dummy features.
            road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(self.device)
            veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(self.device)

        # Run the model and move the output embedding to CPU.
        emb = self.model(
            pixel_values=pixel_values, road_features=road_f, veg_features=veg_f,
        ).cpu()

        # Build the list of countries that have a valid centroid.
        valid = [c for c in self.country_list if c in self.centroids]
        # If no valid centroids exist, return an "Unknown" placeholder.
        if not valid:
            return [("Unknown", 0.0, (0.0, 0.0))]

        # Stack all valid centroids into a matrix of shape (N, D).
        matrix = torch.stack([self.centroids[c] for c in valid])
        # Compute cosine similarity between image embedding and all centroids.
        sim = torch.matmul(emb, matrix.T).squeeze(0)
        # Get the indices of the top-k highest similarity scores.
        topk = sim.argsort(descending=True)[:top_k].numpy()
        # Return list of (country, similarity_score, representative_coordinates).
        return [(valid[i], float(sim[i]), self._coords_for(valid[i]))
                for i in topk]

    # Return the representative coordinates for a given country name.
    def _coords_for(self, country):
        # If an exact match exists in the coordinate map, return it.
        if country in self.country_coords:
            return self.country_coords[country]
        # Try fuzzy matching by lowercasing names.
        nl = country.lower()
        # Iterate over all known country-coordinate pairs.
        for k, v in self.country_coords.items():
            # Match if names are equal ignoring case, or one contains the other.
            if k.lower() == nl or nl in k.lower() or k.lower() in nl:
                return v
        # Return a default coordinate of (0.0, 0.0) if no match found.
        return (0.0, 0.0)

    # Return a random coordinate from the known country coordinates.
    def random_coords(self):
        # If the coordinate map is empty, return (0, 0).
        if not self.country_coords:
            return (0.0, 0.0)
        # Pick a random country's median coordinate.
        return random.choice(list(self.country_coords.values()))


# ---------------------------------------------------------------------------
# Raw StreetCLIP baseline for benchmarking — uses CLIPProcessor + model(**inputs)
# like the official Hugging Face example, no manual embedding computation.
# ---------------------------------------------------------------------------

# Define a baseline class using the raw StreetCLIP model from HuggingFace.
class StreetCLIPBaseline:
    # Initialize with device selection.
    def __init__(self, device="cuda"):
        # Set device to "cuda" if available, otherwise "cpu".
        self.device = device if torch.cuda.is_available() else "cpu"
        # Placeholder for the CLIPModel.
        self.model = None
        # Placeholder for the CLIPProcessor.
        self.processor = None
        # List of country names used as text prompts.
        self.country_list = []
        # Dictionary mapping country name to all its coordinates.
        self._all_coords = {}
        # Dictionary mapping country name to median coordinate.
        self.country_coords = {}
        # Flag indicating whether the model has been loaded.
        self._loaded = False

    # Load the HuggingFace CLIP model and processor.
    def load(self):
        # If already loaded, do nothing.
        if self._loaded:
            return
        # Print the device being used.
        print(f"Loading StreetCLIP baseline on {self.device}...")
        # Import CLIPModel and CLIPProcessor from transformers (lazy import).
        from transformers import CLIPModel, CLIPProcessor
        # Download and instantiate the StreetCLIP model, moving it to device.
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(
            self.device)
        # Download and instantiate the matching processor.
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        # Set model to evaluation mode.
        self.model.eval()

        # Determine the data directory for country labels.
        data_dir = SUBSET_DIR
        # Load the country list from the dataset.
        self.country_list = CountryEncoder(data_dir).country_list
        # Print the number of countries.
        print(f"  countries   : {len(self.country_list)}")

        # Build the coordinate lookup map from dataset JSON files.
        self._build_coord_map(data_dir)
        # Print the number of countries with known coordinates.
        print(f"  coordinates : {len(self.country_coords)} countries")
        # Mark the model as loaded.
        self._loaded = True

    # Build a mapping from country name to its coordinates from JSON files.
    def _build_coord_map(self, data_dir):
        # Dictionary for country -> list of (lat, lng) tuples.
        coords = {}
        # Iterate over all location JSON files.
        for jf in sorted(data_dir.glob("location_*.json")):
            try:
                # Open and parse the JSON file.
                with open(jf) as f:
                    d = json.load(f)
                # Skip non-dict entries.
                if not isinstance(d, dict):
                    continue
                # Extract country name, default "Unknown".
                c = d.get("country_name", "Unknown")
                # Extract coordinates, default [0.0, 0.0].
                ll = d.get("coordinates", [0.0, 0.0])
                # Skip entries without exactly two values.
                if len(ll) != 2:
                    continue
                # Convert to floats.
                lat, lng = float(ll[0]), float(ll[1])
                # Create list for this country if not present.
                if c not in coords:
                    coords[c] = []
                # Add coordinate tuple.
                coords[c].append((lat, lng))
            except (json.JSONDecodeError, Exception):
                # Skip unparseable files.
                continue
        # Store raw coordinate map.
        self._all_coords = coords
        # Compute median coordinate per country.
        self.country_coords = {
            c: (np.median([p[0] for p in pts]),
                np.median([p[1] for p in pts]))
            for c, pts in coords.items()
        }

    # Run inference with the StreetCLIP baseline, returning top-k predictions.
    @torch.inference_mode()
    def predict(self, image, top_k=5):
        # Preprocess the image and country text prompts into model inputs.
        inputs = self.processor(
            text=self.country_list, images=image, return_tensors="pt",
            padding=True)
        # Move all input tensors to the target device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Run the model forward pass.
        outputs = self.model(**inputs)
        # Extract image-to-text logits and softmax to get probabilities.
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)
        # Get indices of top-k highest probability countries.
        topk = probs.argsort(descending=True)[:top_k].cpu().numpy()
        # Return list of (country, probability, representative_coordinates).
        return [(self.country_list[i], float(probs[i]),
                 self._coords_for(self.country_list[i]))
                for i in topk]

    # Return the representative coordinates for a given country (fuzzy match).
    def _coords_for(self, country):
        # Direct lookup in the coordinate map.
        if country in self.country_coords:
            return self.country_coords[country]
        # Lowercase for fuzzy comparison.
        nl = country.lower()
        # Iterate over all known country entries.
        for k, v in self.country_coords.items():
            # Match if names are equal ignoring case, or one is a substring.
            if k.lower() == nl or nl in k.lower() or k.lower() in nl:
                return v
        # Default to (0.0, 0.0).
        return (0.0, 0.0)


# Determine which country a given lat/lng point belongs to, by nearest neighbor.
def find_country_for_coords(lat, lng, all_coords):
    # Track the best (closest) country name.
    best = None
    # Track the smallest distance found so far.
    best_dist = float("inf")
    # Iterate over all countries and their coordinate lists.
    for country, pts in all_coords.items():
        # For each coordinate point in the country.
        for clat, clng in pts:
            # Compute the haversine distance from the query point.
            d = haversine_km(lat, lng, clat, clng)
            # If this point is closer, update best.
            if d < best_dist:
                best_dist = d
                best = country
    # Return the name of the closest country.
    return best


# Compute top-1 and top-5 accuracy given model predictions and the true country.
def _compute_accuracy(top_k_countries, true_country):
    # If either list is missing or empty, return False for both.
    if not true_country or not top_k_countries:
        return False, False
    # Top-1 is correct if the first prediction matches the true country.
    top1 = top_k_countries[0] == true_country
    # Top-5 is correct if the true country appears anywhere in the predictions.
    top5 = true_country in top_k_countries
    # Return both bools.
    return top1, top5


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

# Take a screenshot of the panorama view with UI elements hidden.
async def _screenshot_panorama(page):
    # Hide UI overlay elements before taking the screenshot.
    await page.evaluate("""
    () => {
        const ids = [
            '.logo', '.menu-button-area', '#bottom-bar',
            '.end-bottom-area', '.gameplay-ad-area',
            '#map-holder',
        ];
        ids.forEach(sel => {
            const el = document.querySelector(sel);
            if (el) { el.__ogPrevDisplay = el.style.display; el.style.display = 'none'; }
        });
    }""")
    # Capture the screenshot as raw PNG bytes.
    data = await page.screenshot(type="png")
    # Restore the original display styles of the hidden elements.
    await page.evaluate("""
    () => {
        const ids = [
            '.logo', '.menu-button-area', '#bottom-bar',
            '.end-bottom-area', '.gameplay-ad-area',
            '#map-holder',
        ];
        ids.forEach(sel => {
            const el = document.querySelector(sel);
            if (el) el.style.display = el.__ogPrevDisplay || '';
        });
    }""")
    # Return the screenshot bytes.
    return data


# Read the true location (lat, lng) from the panorama iframe's URL.
async def _read_true_location(page):
    try:
        # Evaluate JS to get the src attribute of the panorama iframe.
        src = await page.evaluate(
            "() => document.querySelector('#panorama-iframe')?.src || null")
        # If no iframe source was found, return None.
        if not src:
            return None
        # Parse the URL query string to get the 'location' parameter.
        loc = parse_qs(urlparse(src).query).get("location", [None])[0]
        # If no location parameter was found, return None.
        if not loc:
            return None
        # Split the location string by comma into latitude and longitude parts.
        parts = loc.split(",")
        # If there are at least two parts, return them as floats.
        if len(parts) >= 2:
            return (float(parts[0].strip()), float(parts[1].strip()))
    except Exception:
        # Silently ignore any parsing errors.
        pass
    # Return None if location could not be determined.
    return None


# Attempt to dismiss cookie consent dialogs on the page.
async def _dismiss_cookies(page):
    # Try up to 5 times to dismiss the cookie consent popup.
    for _ in range(5):
        # Click any visible consent button using JavaScript selectors.
        await page.evaluate("""
        () => {
            for (const sel of [
                '.fc-primary-button', '.fc-consent-root .fc-primary-button',
                '[class*="fc-button"]', '.fc-button'
            ]) {
                try {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) { el.click(); return; }
                } catch(e) {}
            }
        }""")
        # Wait a short random interval before trying again.
        await asyncio.sleep(random.uniform(0.3, 0.6))
        # Try pressing the Escape key to dismiss overlays.
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

        # Check whether the consent dialog is still visible.
        gone = await page.evaluate("""
        () => {
            const roots = document.querySelectorAll(
                '.fc-consent-root, [class*="fc-dialog-overlay"]');
            if (roots.length === 0) return true;
            for (const el of roots) {
                const s = getComputedStyle(el);
                if (s.display !== 'none' && s.visibility !== 'hidden'
                    && parseFloat(s.opacity) > 0.1) return false;
            }
            return true;
        }""")
        # If the dialog is gone, print confirmation and exit the loop.
        if gone:
            print("  cookie consent dismissed")
            return
        # Otherwise, wait 0.5 seconds and try again.
        await asyncio.sleep(0.5)


# Hover the mouse over the minimap to trigger any lazy-load effects.
async def _hover_minimap(page):
    # Try several selectors to find the minimap container element.
    for sel in [".leaflet-container", '[class*="guess-map"]', "#map"]:
        try:
            # Look for an element matching the selector.
            el = await page.query_selector(sel)
            # If no element found, try the next selector.
            if not el:
                continue
            # Get the bounding rectangle of the element.
            box = await page.evaluate(f"""() => {{
                const el = document.querySelector('{sel}');
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {{left:r.left,top:r.top,w:r.width,h:r.height}};
            }}""")
            # If no box or the element has zero dimensions, try next selector.
            if not box or box["w"] <= 0 or box["h"] <= 0:
                continue
            # Found a valid minimap; stop searching.
            break
        except Exception:
            # If query fails, try next selector.
            continue
    else:
        # If no minimap element was found after exhausting selectors, return.
        return

    # Compute the horizontal center of the minimap.
    cx = box["left"] + box["w"] / 2
    # Compute the vertical center of the minimap.
    cy = box["top"] + box["h"] / 2
    # Move the mouse to a random position first (simulates human behavior).
    await page.mouse.move(random.randint(100, 800), random.randint(100, 600))
    # Small delay before starting the hover motion.
    await asyncio.sleep(0.1)
    # Perform a progressive mouse move toward the minimap center.
    for i in range(random.randint(6, 12)):
        # Compute interpolation factor t (0 to ~1) with slight randomness.
        t = (i + 1) / random.randint(8, 14)
        # Move the mouse progressively toward the center with noise.
        await page.mouse.move(
            int(box["left"] + box["w"] / 2 * t + random.uniform(-3, 3)),
            int(box["top"] + box["h"] / 2 * t + random.uniform(-3, 3)),
        )
        # Short delay between incremental moves.
        await asyncio.sleep(random.uniform(0.01, 0.03))
    # Wait 1 second for the map tiles to load after hovering.
    await asyncio.sleep(1.0)


# Reset the minimap to its default view (world view, minimum zoom).
async def _reset_minimap(page):
    # Try to reset the Leaflet map via its JavaScript API.
    ok = await page.evaluate("""
    () => {
        const el = document.querySelector('.leaflet-container, #map');
        const maps = window.__ogMaps__ || [];
        let map = null;
        for (const m of maps) {
            try { if (m && m._container === el) { map = m; break; } } catch(e) {}
        }
        if (!map && maps.length === 1) map = maps[0];
        if (!map && el && el._leaflet_map) map = el._leaflet_map;
        if (map && typeof map.setView === 'function') {
            try {
                const z = (typeof map.getMinZoom === 'function')
                    ? (map.getMinZoom() || 0) : 0;
                map.setView([20, 0], z, { animate: false });
                if (typeof map.invalidateSize === 'function') map.invalidateSize();
                return true;
            } catch(e) {}
        }
        return false;
    }""")
    # If the JS reset succeeded, wait a short time and return.
    if ok:
        await asyncio.sleep(0.35)
        return
    # Fallback: click the zoom-out button 8 times to reset the view.
    try:
        btn = await page.query_selector(
            ".leaflet-control-zoom-out, a.leaflet-control-zoom-out")
        if btn:
            for _ in range(8):
                try:
                    # Click the zoom-out button.
                    await btn.click(timeout=500)
                    # Small delay between clicks.
                    await asyncio.sleep(0.12)
                except Exception:
                    # Stop if clicking fails.
                    break
            # Wait for the map to settle after zoom operations.
            await asyncio.sleep(0.3)
    except Exception:
        # Silently ignore fallback errors.
        pass


# Click on the minimap at the given geographic (lat, lng) coordinates.
async def _click_on_map(page, lat, lng):
    # Use the Leaflet map instance to convert lat/lng to pixel coordinates.
    c = await page.evaluate("""([lat, lng]) => {
        const el = document.querySelector('.leaflet-container, #map, [class*="map"]');
        if (!el) return null;
        const maps = window.__ogMaps__ || [];
        let map = null;
        for (const m of maps) {
            try { if (m && m._container === el) { map = m; break; } } catch(e) {}
        }
        if (!map && maps.length === 1) map = maps[0];
        if (!map && el._leaflet_map) map = el._leaflet_map;
        if (!map || typeof map.setView !== 'function') return null;
        try {
            if (typeof map.invalidateSize === 'function') map.invalidateSize();
            const minZ = (typeof map.getMinZoom === 'function')
                ? (map.getMinZoom() || 0) : 0;
            map.setView([lat, lng], minZ, { animate: false });
            const rect = el.getBoundingClientRect();
            const pt = map.latLngToContainerPoint([lat, lng]);
            const x = rect.left + pt.x;
            const y = rect.top + pt.y;
            if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom)
                return null;
            return {x, y, left: rect.left, top: rect.top, w: rect.width, h: rect.height};
        } catch(e) { return null; }
    }""", [lat, lng])

    # If Leaflet successfully returned a pixel position, click it directly.
    if c:
        # Short delay before clicking.
        await asyncio.sleep(0.25)
        # Clamp the click coordinate within the map element bounds.
        x = max(c["left"] + 4, min(c["x"], c["left"] + c["w"] - 4))
        y = max(c["top"] + 4, min(c["y"], c["top"] + c["h"] - 4))
        # Log the click coordinates for debugging.
        print(f"    click [center] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")
        # Move the mouse to the target position.
        await page.mouse.move(x, y)
        # Short delay before clicking.
        await asyncio.sleep(0.1)
        # Perform the click.
        await page.mouse.click(x, y)
        # Wait a random interval after clicking.
        await asyncio.sleep(random.uniform(0.3, 0.6))
        # Done; exit early.
        return

    # Fallback approach: reset the minimap and then compute pixel location.
    await _reset_minimap(page)
    # Determine the pixel position using either Leaflet, tile calculation, or Mercator.
    info = await page.evaluate("""([lat, lng]) => {
        const el = document.querySelector('.leaflet-container, #map, [class*="map"]');
        if (!el) return {m:'none'};
        const rect = el.getBoundingClientRect();
        const ri = {left:rect.left,top:rect.top,w:rect.width,h:rect.height};
        const cx = rect.left + rect.width / 2;

        let map = null;
        for (const m of (window.__ogMaps__ || [])) {
            try { if (m && m._container === el) { map = m; break; } } catch(e) {}
        }
        if (!map && (window.__ogMaps__ || []).length === 1) map = window.__ogMaps__[0];
        if (!map && el._leaflet_map) map = el._leaflet_map;
        if (map && typeof map.latLngToContainerPoint === 'function') {
            const pt = map.latLngToContainerPoint([lat, lng]);
            const x = rect.left + pt.x, y = rect.top + pt.y;
            return {m:'leaflet', x, y, on: x >= rect.left && x <= rect.right
                    && y >= rect.top && y <= rect.bottom, rect: ri};
        }

        const imgs = Array.from(el.querySelectorAll('img'));
        let tile = null, bestD = Infinity;
        for (const img of imgs) {
            const src = img.currentSrc || img.src || '';
            let z, tx, ty;
            let m = src.match(/\\/(\\d{1,2})\\/(\\d{1,7})\\/(\\d{1,7})(?:[.?&\\/]|$)/);
            if (m) { z = +m[1]; tx = +m[2]; ty = +m[3]; }
            else {
                const mz = src.match(/[?&](?:z|zoom)=(\\d{1,2})/);
                const mx = src.match(/[?&]x=(\\d{1,7})/);
                const my = src.match(/[?&]y=(\\d{1,7})/);
                if (mz && mx && my) { z = +mz[1]; tx = +mx[1]; ty = +my[1]; }
            }
            if (z === undefined || isNaN(z)) continue;
            const r = img.getBoundingClientRect();
            if (r.width < 64 || r.height < 64) continue;
            if (r.right < rect.left || r.left > rect.right
                || r.bottom < rect.top || r.top > rect.bottom) continue;
            const d = (r.left+r.width/2-cx)**2 + (r.top+r.height/2-(rect.top+rect.height/2))**2;
            if (d < bestD) { bestD = d;
                tile = {z, tx, ty, left:r.left, top:r.top, w:r.width, h:r.height}; }
        }
        if (tile) {
            const span = tile.w * Math.pow(2, tile.z);
            let x = tile.left + ((lng+180)/360)*span - tile.tx*tile.w;
            const s = Math.max(-0.9999, Math.min(0.9999, Math.sin(lat*Math.PI/180)));
            const yNorm = 0.5 - Math.log((1+s)/(1-s))/(4*Math.PI);
            const y = tile.top + yNorm*(tile.h*Math.pow(2,tile.z)) - tile.ty*tile.h;
            x = x - span * Math.round((x - cx) / span);
            return {m:'tiles', x, y, z:tile.z,
                    on: x >= rect.left && x <= rect.right
                        && y >= rect.top && y <= rect.bottom, rect: ri};
        }
        return {m:'rect', rect: ri};
    }""", [lat, lng])

    # Extract the method type used (leaflet, tiles, or rect).
    m = info.get("m", "none") if info else "none"
    # If Leaflet or tile-based position was computed, use it.
    if m in ("leaflet", "tiles") and info.get("x") is not None:
        x, y = info["x"], info["y"]
    # If only rectangle info is available, use Mercator projection to estimate.
    elif info and info.get("rect"):
        # Extract the bounding rectangle of the map element.
        r = info["rect"]
        # Compute x using simple linear longitude-to-pixel mapping.
        x = r["left"] + (lng + 180) / 360 * r["w"]
        # Convert latitude to radians for Mercator projection.
        lat_rad = math.radians(lat)
        # Compute y using the inverse Mercator projection formula.
        y = r["top"] + (0.5 - math.log(
            math.tan(math.pi / 4 + lat_rad / 2)) / (2 * math.pi)) * r["h"]
        # Mark the method as mercator.
        m = "mercator"
    else:
        # If all methods fail, click the center of a 1920x1080 viewport.
        print("    WARNING: fallback centre click")
        x, y = 960.0, 540.0

    # Clamp the computed coordinates within the map element boundaries.
    if info and info.get("rect"):
        # Get the map's bounding rectangle.
        r = info["rect"]
        # Clamp x to stay within the horizontal bounds with a 4px margin.
        x = max(r["left"] + 4, min(x, r["left"] + r["w"] - 4))
        # Clamp y to stay within the vertical bounds with a 4px margin.
        y = max(r["top"] + 4, min(y, r["top"] + r["h"] - 4))

    # Log the click position and method used.
    print(f"    click [{m}] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")
    # Move the mouse to the computed pixel coordinates.
    await page.mouse.move(x, y)
    # Short delay before clicking.
    await asyncio.sleep(0.1)
    # Perform the click.
    await page.mouse.click(x, y)
    # Wait a random interval after clicking.
    await asyncio.sleep(random.uniform(0.3, 0.6))


# Submit the current guess by clicking the confirm/guess button.
async def _submit(page):
    # Try several selectors for the submit button.
    for sel in [
        "#confirm-button", "[class*='confirm-button']",
        "button:has-text('Guess')", "button:has-text('Submit')",
        "div:has-text('Guess')",
    ]:
        try:
            # Wait up to 2 seconds for the element to appear.
            btn = await page.wait_for_selector(sel, timeout=2000)
            # If the button was found, click it.
            if btn:
                await btn.click()
                # Wait a random interval after clicking.
                await asyncio.sleep(random.uniform(0.3, 0.6))
                # Done; exit early.
                return
        except Exception:
            # If the element was not found or timed out, try the next selector.
            continue
    # Fallback: press the Enter key as a last-resort submission method.
    try:
        await page.keyboard.press("Enter")
    except Exception:
        pass


# Advance to the next round by clicking the appropriate button.
async def _next_round(page):
    # Try several button text labels to advance.
    for text in ["Play Again", "Next Round", "Next", "Continue", "Return"]:
        try:
            # Wait up to 2 seconds for a button with the given text.
            btn = await page.wait_for_selector(
                f"button:has-text('{text}')", timeout=2000)
            # If found, click it.
            if btn:
                await btn.click()
                # Wait a random interval after clicking.
                await asyncio.sleep(random.uniform(0.3, 0.6))
                # Done; exit early.
                return
        except Exception:
            # If not found, try the next label.
            continue
    # Fallback: reload the game page entirely.
    await page.goto(GAME_URL, wait_until="networkidle", timeout=30000)


# ---------------------------------------------------------------------------
# Round execution + logging
# ---------------------------------------------------------------------------

# Execute a single round of the game: screenshot, predict, click, submit.
async def _play_round(page, ai, mode, round_num, log_entries, run_id,
                      save_images, baselines=None):
    # Initialize a result dictionary for this round with default values.
    result = {
        "round": round_num,
        "time": now_iso(),
        "mode": mode,
        "true_lat": None, "true_lng": None,
        "true_country": None,
        "guess_lat": None, "guess_lng": None,
        "guess_country": None,
        "distance_km": None,
        "score": None,
        "predictions": [],
        "baseline_predictions": {},
    }

    # If saving images is enabled, create a directory for this round.
    if save_images:
        # Build the path for the round's directory.
        round_dir = ROUNDS_DIR / f"round{round_num}"
        # Create the directory (and parents) if they don't exist.
        round_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Otherwise, leave round_dir as None.
        round_dir = None

    # Read the true geographic location from the panorama iframe URL.
    true_loc = await _read_true_location(page)

    # Handle "perfect" mode: use the exact coordinates without model prediction.
    if mode == "perfect":
        # If coords are missing or zero, skip this round.
        if not true_loc or true_loc == (0.0, 0.0):
            print("  FAILED: no coords available")
            return
        # Unpack the true coordinates.
        lat, lng = true_loc
        # Store true coordinates in the result.
        result["true_lat"], result["true_lng"] = lat, lng
        # The guess is the same as the true location (perfect mode).
        result["guess_lat"], result["guess_lng"] = lat, lng
        # If the AI is loaded with coordinate data, estimate the country.
        if ai and ai.country_coords:
            # Find the country matching these coordinates.
            result["true_country"] = find_country_for_coords(
                lat, lng, ai._all_coords)
            # Print the estimated true country.
            print(f"  true country (est): {result['true_country']}")
        # Print the true coordinates.
        print(f"  true : {lat:.4f}, {lng:.4f}")
    else:
        # "ai" mode: take a screenshot and use the model to predict.
        # Capture a screenshot of the panorama without UI overlays.
        screenshot_bytes = await _screenshot_panorama(page)
        # If saving images, write the screenshot to disk.
        if round_dir:
            (round_dir / "streetview.png").write_bytes(screenshot_bytes)
        # Open the screenshot bytes as a PIL Image and convert to RGB.
        img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        # Set the label for the main model's predictions.
        main_model_label = "Penguin"
        # Get the top-5 model predictions from the AI.
        preds = ai.predict(img, top_k=5)
        # Extract only the country names from the predictions.
        pred_countries = [c for c, _, _ in preds]
        # Format predictions as a list of dicts for logging.
        result["predictions"] = [
            {"country": c, "score": s, "lat": cl, "lng": cg}
            for c, s, (cl, cg) in preds
        ]
        # Print a header for the main model output.
        print(f"  --- {main_model_label} ---")
        # Print each prediction with rank, country, score, and coordinates.
        for j, p in enumerate(result["predictions"], 1):
            # Mark the top (first) prediction with an arrow indicator.
            mark = " <<<" if j == 1 else ""
            # Print the formatted prediction line.
            print(f"  {j:4d} {p['country']:35s} {p['score']:.4f}  "
                  f"({p['lat']:.4f}, {p['lng']:.4f}){mark}")

        # Define a set of countries to skip as likely wrong (microstates).
        skip = {"Unknown", "San Marino", "Holy See (Vatican City State)"}
        # Filter predictions to exclude skipped countries.
        chosen = [p for p in preds if p[0] not in skip]
        # If all predictions are skipped, fall back to the original first prediction.
        if not chosen:
            chosen = preds[:1]
        # Use the top remaining prediction's coordinates as the guess.
        lat, lng = chosen[0][2]
        # If the guess is exactly (0, 0), pick a random coordinate instead.
        if lat == 0.0 and lng == 0.0:
            lat, lng = ai.random_coords()
        # Store the guess coordinates in the result.
        result["guess_lat"], result["guess_lng"] = lat, lng
        # Store the guessed country name.
        result["guess_country"] = chosen[0][0]

        # If the true location is known, compute accuracy metrics.
        if true_loc:
            # Store the true coordinates in the result.
            result["true_lat"], result["true_lng"] = true_loc
            # Compute the haversine distance between guess and truth.
            dist = haversine_km(lat, lng, true_loc[0], true_loc[1])
            # Record the distance in km (rounded to 1 decimal).
            result["distance_km"] = round(dist, 1)
            # Convert distance to an OpenGuessr-style score.
            result["score"] = distance_score(dist)
            # Print the true coordinates, distance, and score.
            print(f"  true : {true_loc[0]:.4f}, {true_loc[1]:.4f}  "
                  f"|  {format_distance(dist)}  |  {result['score']:,} pts")

        # If true location and coordinate data are available, estimate the country.
        if true_loc and ai.country_coords:
            # Find the country for the true location.
            result["true_country"] = find_country_for_coords(
                true_loc[0], true_loc[1], ai._all_coords)
            # Print the identified true country.
            print(f"  true country (est): {result['true_country']}")

        # If baseline models exist, also run predictions with them.
        if baselines and img:
            # Iterate over each baseline model.
            for name, bl_model in baselines.items():
                # Get top-5 predictions from the baseline model.
                bl_preds = bl_model.predict(img, top_k=5)
                # Extract country names from baseline predictions.
                bl_countries = [c for c, _, _ in bl_preds]
                # Store formatted baseline predictions in the result.
                result["baseline_predictions"][name] = [
                    {"country": c, "score": s, "lat": cl, "lng": cg}
                    for c, s, (cl, cg) in bl_preds
                ]
                # Print header for baseline model output.
                print(f"  --- {name} ---")
                # Print each baseline prediction.
                for j, p in enumerate(result["baseline_predictions"][name], 1):
                    # Mark the top prediction.
                    mark = " <<<" if j == 1 else ""
                    # Print the formatted prediction.
                    print(f"  {j:4d} {p['country']:35s} {p['score']:.4f}  "
                          f"({p['lat']:.4f}, {p['lng']:.4f}){mark}")

        # If a true country was identified, compute model accuracy.
        if result.get("true_country"):
            # Alias for the true country.
            tc = result["true_country"]
            # Extract the chosen prediction country names for top-1 check.
            chosen_countries = [c for c, _, _ in chosen]
            # Extract all predicted country names for top-5 check.
            pred_countries = [c for c, _, _ in preds]
            # Check if the top chosen country matches the true country.
            top1_ok = chosen_countries[0] == tc if chosen_countries else False
            # Check if the true country is anywhere in the top-5 predictions.
            top5_ok = tc in pred_countries
            # Store Penguin top-1 accuracy for this round.
            result["penguin_top1"] = top1_ok
            # Store Penguin top-5 accuracy for this round.
            result["penguin_top5"] = top5_ok
            # Print the Penguin accuracy results.
            print(f"  {main_model_label} top-1: {'OK' if top1_ok else 'miss'}  "
                  f"top-5: {'OK' if top5_ok else 'miss'}")
            # For each baseline model, compute its accuracy as well.
            for name in result.get("baseline_predictions", {}):
                # Extract country names from baseline predictions.
                bl_countries = [p["country"]
                                for p in result["baseline_predictions"][name]]
                # Compute top-1 and top-5 accuracy for this baseline.
                b_top1, b_top5 = _compute_accuracy(bl_countries, tc)
                # Initialize the baseline_top1 dict if not present.
                result.setdefault("baseline_top1", {})
                # Initialize the baseline_top5 dict if not present.
                result.setdefault("baseline_top5", {})
                # Store this baseline's top-1 result.
                result["baseline_top1"][name] = b_top1
                # Store this baseline's top-5 result.
                result["baseline_top5"][name] = b_top5
                # Print the baseline accuracy results.
                print(f"  {name} top-1: {'OK' if b_top1 else 'miss'}  "
                      f"top-5: {'OK' if b_top5 else 'miss'}")

    # Hover over the minimap to trigger tile loading.
    await _hover_minimap(page)
    # Reset the minimap to the default world view.
    await _reset_minimap(page)
    # Click on the minimap at the guessed coordinates.
    await _click_on_map(page, lat, lng)
    # Wait a random short interval.
    await asyncio.sleep(random.uniform(0.3, 0.6))
    # Submit the guess by clicking the confirm button.
    await _submit(page)
    # Wait a random interval before the next round.
    await asyncio.sleep(random.uniform(0.5, 1.0))

    # If saving images, capture a screenshot of the result screen.
    if round_dir:
        await page.screenshot(path=str(round_dir / "result.png"))

    # Append this round's result to the log entries list.
    log_entries.append(result)


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

# Run a full game session: launch browser, load models, play all rounds, log.
async def _run_session(args):
    # Import async_playwright from the playwright library (lazy import).
    from playwright.async_api import async_playwright

    # Instantiate the Penguin AI with the chosen device and feature models.
    ai = PenguinAI(device=args.device,
                   road_model=args.road_model,
                   veg_model=args.veg_model)
    # Load the AI model and all associated data.
    ai.load()

    # Dictionary to hold any baseline models for benchmarking.
    baselines = {}
    # If benchmarking is enabled, instantiate and load the StreetCLIP baseline.
    if args.benchmark:
        # Create the StreetCLIP baseline instance.
        sc = StreetCLIPBaseline(device=args.device)
        # Load its model and data.
        sc.load()
        # Add to the baselines dictionary under the key "StreetCLIP".
        baselines["StreetCLIP"] = sc

    # Use the async_playwright context manager for browser lifecycle.
    async with async_playwright() as pw:
        # Launch a Chromium browser instance with stealth flags.
        browser = await pw.chromium.launch(
            headless=args.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        # Create a new browser context with a desktop-like viewport and agent.
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            bypass_csp=True,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        # Create a new page (tab) in the context.
        page = await ctx.new_page()
        # Inject the anti-detection script before page navigation.
        await page.add_init_script(ANTI_DETECT_SCRIPT)
        # Inject the Leaflet map hooking script before page navigation.
        await page.add_init_script(LEAFLET_HOOK_SCRIPT)

        # Print the session header with settings.
        print(f"\n{'=' * 60}")
        print(f"  OpenGuessr Player")
        print(f"  Mode  : {args.mode.upper()}{' + BENCHMARK' if args.benchmark else ''}")
        print(f"  Rounds: {args.rounds}")
        print(f"  Device: {ai.device}")
        # Print baseline model names if active.
        if baselines:
            print(f"  Baselines: {', '.join(baselines.keys())}")
        print(f"{'=' * 60}")
        # Print the game URL before navigating.
        print(f"\nLoading {GAME_URL} ...")
        # Navigate to the game page and wait for network idle.
        await page.goto(GAME_URL, wait_until="networkidle", timeout=30000)
        # Dismiss any cookie consent dialogs on the page.
        await _dismiss_cookies(page)

        # Generate a unique run ID from the current datetime.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Initialize an empty list to collect per-round log entries.
        log_entries = []

        # Play the specified number of rounds.
        for r in range(1, args.rounds + 1):
            # Print a round separator with the current round number.
            print(f"\n--- Round {r}/{args.rounds} ---")
            try:
                # Execute one round of the game.
                await _play_round(page, ai, args.mode, r, log_entries, run_id,
                                  not args.no_images,
                                  baselines=baselines if baselines else None)
            except Exception as e:
                # If an error occurs, print it and continue with next round.
                print(f"  ERROR: {e}")
            # If there are more rounds to play, advance to the next round.
            if r < args.rounds:
                # Click the next-round button.
                await _next_round(page)
                # Wait a random interval before the next round starts.
                await asyncio.sleep(random.uniform(0.5, 1.5))

        # Close the browser after all rounds are done.
        await browser.close()

    # Write the session log to disk.
    _write_log(log_entries, run_id)


# Write the accumulated round entries and summary statistics to a JSON file.
def _write_log(entries, run_id):
    # If there are no entries, do nothing.
    if not entries:
        return
    # Sum the scores of all rounds (treating None as 0).
    total = sum(e.get("score", 0) or 0 for e in entries)
    # Collect non-None distance values from all rounds.
    distances = [e["distance_km"] for e in entries if e.get("distance_km")]

    # Collect Penguin top-1 correctness per round.
    penguin_top1 = [e["penguin_top1"] for e in entries
                    if "penguin_top1" in e]
    # Collect Penguin top-5 correctness per round.
    penguin_top5 = [e["penguin_top5"] for e in entries
                    if "penguin_top5" in e]
    # Compute Penguin top-1 accuracy as a percentage (or None if no data).
    penguin_top1_acc = sum(penguin_top1) / len(penguin_top1) * 100 \
        if penguin_top1 else None
    # Compute Penguin top-5 accuracy as a percentage (or None if no data).
    penguin_top5_acc = sum(penguin_top5) / len(penguin_top5) * 100 \
        if penguin_top5 else None

    # Initialize a dict to accumulate per-baseline top-1 and top-5 results.
    baseline_acc = {}
    # Iterate over all round entries.
    for e in entries:
        # Skip rounds that don't have baseline_top1 data.
        if "baseline_top1" not in e:
            continue
        # For each baseline model tracked in this round.
        for name in e["baseline_top1"]:
            # Initialize tracking lists for this baseline if not present.
            if name not in baseline_acc:
                baseline_acc[name] = {"top1": [], "top5": []}
            # Append this round's top-1 correctness for this baseline.
            baseline_acc[name]["top1"].append(e["baseline_top1"][name])
            # Append this round's top-5 correctness for this baseline.
            baseline_acc[name]["top5"].append(e["baseline_top5"][name])

    # Build the summary dictionary with overall statistics.
    summary = {
        "run_id": run_id,
        "rounds": len(entries),
        "total_score": total,
        "avg_distance_km": round(sum(distances) / len(distances), 1)
        if distances else None,
        "median_distance_km": round(sorted(distances)[len(distances) // 2], 1)
        if distances else None,
    }
    # Add Penguin accuracy percentages to the summary if available.
    if penguin_top1_acc is not None:
        summary["penguin_top1_pct"] = round(penguin_top1_acc, 1)
        summary["penguin_top5_pct"] = round(penguin_top5_acc, 1)
    # For each baseline, compute and add its accuracy percentages.
    for name, accs in baseline_acc.items():
        # Only include if there is top-1 data.
        if accs["top1"]:
            # Compute baseline top-1 accuracy as percentage.
            summary[f"{name}_top1_pct"] = round(
                sum(accs["top1"]) / len(accs["top1"]) * 100, 1)
            # Compute baseline top-5 accuracy as percentage.
            summary[f"{name}_top5_pct"] = round(
                sum(accs["top5"]) / len(accs["top5"]) * 100, 1)

    # Ensure the runs directory exists.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Build the log file path using the run ID.
    log_path = RUNS_DIR / f"run_{run_id}.json"
    # Write the summary and round entries to the JSON file.
    with open(log_path, "w") as f:
        json.dump({"summary": summary, "rounds": entries}, f, indent=2)

    # Print a summary report to the console.
    print(f"\n{'=' * 60}")
    print(f"  Session complete")
    print(f"  Rounds       : {summary['rounds']}")
    print(f"  Total score  : {summary['total_score']:,}")
    # Print average and median distance if available.
    if summary["avg_distance_km"] is not None:
        print(f"  Avg distance : {format_distance(summary['avg_distance_km'])}")
        print(f"  Med distance : {format_distance(summary['median_distance_km'])}")
    # Print Penguin accuracy if available.
    if penguin_top1_acc is not None:
        print(f"  Penguin top-1: {penguin_top1_acc:.1f}%  "
              f"top-5: {penguin_top5_acc:.1f}%")
    # Print baseline accuracy for each baseline.
    for name, accs in baseline_acc.items():
        if accs["top1"]:
            # Calculate top-1 accuracy as a percentage.
            t1 = sum(accs["top1"]) / len(accs["top1"]) * 100
            # Calculate top-5 accuracy as a percentage.
            t5 = sum(accs["top5"]) / len(accs["top5"]) * 100
            # Print the baseline accuracy.
            print(f"  {name} top-1: {t1:.1f}%  top-5: {t5:.1f}%")
    # Print the path where the log was saved.
    print(f"  Log saved to : {log_path}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Define the main entry-point function for command-line usage.
def main():
    # Create an argument parser with a description.
    parser = argparse.ArgumentParser(description="OpenGuessr AI Player")
    # Add an argument for the game mode: "ai" (model) or "perfect" (iframe coords).
    parser.add_argument("--mode", choices=["ai", "perfect"], default="ai",
                        help="ai=model prediction | perfect=iframe coords")
    # Add an argument for the number of rounds to play (default 5).
    parser.add_argument("--rounds", type=int, default=5)
    # Add an argument to select the device for inference (default cuda if available).
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Add a flag to run the browser in headless mode.
    parser.add_argument("--headless", action="store_true",
                        help="Run browser headless")
    # Add a flag to skip saving round screenshots.
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving round screenshots")
    # Add a flag to enable benchmarking against the raw StreetCLIP baseline.
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark Penguin vs raw StreetCLIP baseline")
    # Add an argument to choose the road feature detection model.
    parser.add_argument("--road-model", default="yolo_world",
                        choices=["grounding_dino", "yolo_world"])
    # Add an argument to choose the vegetation feature detection model.
    parser.add_argument("--veg-model", default="clip",
                        choices=["clip", "ram++"])
    # Parse the command-line arguments.
    args = parser.parse_args()

    # Run the async session function in the asyncio event loop.
    asyncio.run(_run_session(args))


# If this script is executed directly (not imported), run the main function.
if __name__ == "__main__":
    main()
