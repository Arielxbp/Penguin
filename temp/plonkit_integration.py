# Import json for loading and saving JSON files (country prompts, etc.)
import json
# Import re for cleaning text strings with regular expression substitutions
import re
# Import Path from pathlib for file system path operations
from pathlib import Path
# Import Optional for type hints that allow None values
from typing import Optional

# Import PyTorch for tensor operations and GPU/CPU device management
import torch
# Import CLIPModel and CLIPProcessor from HuggingFace transformers for text encoding
from transformers import CLIPModel, CLIPProcessor
# Import tqdm for progress bars during long-running loops
from tqdm import tqdm

# Import configuration constants for paths and model identifiers
from config import (
    ROOT,
    STREETCLIP_MODEL,
    EMBEDDING_DIR,
)
# Import Plonkit utility functions and constants for building the country database
from plonkit import (
    build_database,
    build_country_text_prompt,
    get_country_by_name,
    PLONKIT_DIR,
)

# Define the directory path for cached country text embeddings
COUNTRY_TEXT_DIR = EMBEDDING_DIR / "country_texts"
# Create the directory (and any missing parents) if it doesn't already exist
COUNTRY_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Map long/unusual country names to shorter common-name equivalents for Plonkit lookup
COUNTRY_NAME_MAP = {
    "Bolivia, Plurinational State of": "bolivia",
    "Brunei Darussalam": "brunei",
    "Iran, Islamic Republic of": "iran",
    "Korea, Republic of": "south korea",
    "Lao People's Democratic Republic": "laos",
    "Micronesia, Federated States of": "micronesia",
    "Moldova, Republic of": "moldova",
    "Russian Federation": "russia",
    "Syrian Arab Republic": "syria",
    "Tanzania, United Republic of": "tanzania",
    "United States": "united states",
    "Venezuela, Bolivarian Republic of": "venezuela",
    "Viet Nam": "vietnam",
    "Czechia": "czech republic",
    "Eswatini": "eswatini",
    "Côte d'Ivoire": "ivory coast",
    "Holy See (Vatican City State)": "vatican city",
    "Kosovo": "kosovo",
    "Macao": "macau",
    "Taiwan, Province of China": "taiwan",
    "Hong Kong": "hong kong",
    "Congo, The Democratic Republic of the": "democratic republic of the congo",
}


# Attempt to find a country's Plonkit database entry by resolving name aliases
def resolve_country_in_plonkit(db: dict, country_name: str) -> Optional[dict]:
    # Try a direct lookup by the given country name
    entry = get_country_by_name(db, country_name)
    # If found, return the entry immediately
    if entry:
        return entry
    # Check if the name has a predefined mapping in COUNTRY_NAME_MAP
    mapped = COUNTRY_NAME_MAP.get(country_name)
    # If a mapped name exists, attempt a lookup with that name
    if mapped:
        entry = get_country_by_name(db, mapped)
        # If the mapped lookup succeeds, return the entry
        if entry:
            return entry
    # Fallback: lowercase and strip the name for case-insensitive comparison
    name_lower = country_name.lower().strip()
    # Iterate through all entries in the database to find a case-insensitive match
    for key, value in db.items():
        # Check if the value is a dict and its "name" field matches the lowercased input
        if isinstance(value, dict) and value.get("name", "").lower() == name_lower:
            return value
    # If no match is found by any method, return None
    return None


# Define a class that encodes country names into text embeddings using Plonkit prompts + CLIP
class PlonkitCountryEncoder:
    # Initialize the encoder with a target device (defaults to CUDA if available)
    def __init__(self, device: str = "cuda"):
        # Use the requested device if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        # Build the Plonkit country feature database
        self.db = build_database()
        # Placeholder for the CLIP text model (lazily loaded)
        self._text_model = None
        # Placeholder for the CLIP tokenizer (lazily loaded)
        self._tokenizer = None
        # Dictionary to cache precomputed country text embeddings
        self.country_embeddings: dict[str, torch.Tensor] = {}
        # Dictionary to cache the text prompts used for each country
        self.country_prompts: dict[str, str] = {}

    # Lazily load the CLIP text encoder and tokenizer (only on first use)
    def _load_text_encoder(self):
        # If the text model is already loaded, do nothing
        if self._text_model is not None:
            return
        # Load the pretrained CLIP model from the configured StreetCLIP checkpoint
        model = CLIPModel.from_pretrained(STREETCLIP_MODEL).to(self.device)
        # Extract the text-only sub-model from the full CLIP model
        self._text_model = model.text_model
        # Extract the final linear projection layer for text embeddings
        self._text_projection = model.text_projection
        # Load the tokenizer from the same pretrained model
        self._tokenizer = CLIPProcessor.from_pretrained(STREETCLIP_MODEL).tokenizer

    # Encode a single text string into a text embedding vector (runs without gradients)
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        # Ensure the text encoder is loaded before encoding
        self._load_text_encoder()
        # Tokenize the input text with padding, truncation, and a max length of 77
        tokens = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        # Pass the tokenized input through the CLIP text model
        text_outputs = self._text_model(**tokens)
        # Extract the pooled output representation from the text model
        pooled = text_outputs.pooler_output
        # Project the pooled output into the shared embedding space
        embedding = self._text_projection(pooled)
        # Remove the batch dimension, move to CPU, and return the embedding
        return embedding.squeeze(0).cpu()

    # Build and cache Plonkit-powered text prompts for a list of country names
    def build_prompts(self, country_names: list[str]):
        # Iterate over each country name with a tqdm progress bar
        for name in tqdm(country_names, desc="Building Plonkit prompts"):
            # Build a descriptive text prompt for this specific country
            prompt = self._build_prompt_for_country(name)
            # Store the generated prompt in the prompts cache
            self.country_prompts[name] = prompt

    # Build a natural-language prompt describing visual features of a given country
    def _build_prompt_for_country(self, country_name: str) -> str:
        # Resolve the country name to a Plonkit database entry (with alias mapping)
        entry = resolve_country_in_plonkit(self.db, country_name)
        # If a valid entry was found, generate a prompt from Plonkit feature data
        if entry:
            return build_country_text_prompt(self.db, entry["code"])
        # Otherwise, fall back to a generic prompt with the country name
        return f"A street view photo from {country_name}."

    # Precompute and cache text embeddings for all given country names
    @torch.no_grad()
    def precompute_embeddings(self, country_names: list[str]):
        # First, build all country prompts (enriching with Plonkit feature data)
        self.build_prompts(country_names)
        # Define the file path for cached embeddings
        cache_file = COUNTRY_TEXT_DIR / "country_embeddings.pt"
        # Define the file path for cached prompts
        prompt_file = COUNTRY_TEXT_DIR / "country_prompts.json"

        # If both cache files already exist, load from disk instead of recomputing
        if cache_file.exists() and prompt_file.exists():
            # Load the embeddings tensor dictionary from the cache file
            self.country_embeddings = torch.load(cache_file, weights_only=True)
            # Open and load the prompts JSON cache
            with open(prompt_file) as f:
                self.country_prompts = json.load(f)
            # Print a status message showing how many embeddings were loaded
            print(f"Loaded {len(self.country_embeddings)} cached country embeddings")
            # Exit early since everything was loaded from cache
            return

        # Ensure the text encoder is loaded before computing embeddings
        self._load_text_encoder()
        # Iterate over each country name with a progress bar
        for name in tqdm(country_names, desc="Encoding Plonkit prompts"):
            # Retrieve the prompt from the cache, or build it on-the-fly if missing
            prompt = self.country_prompts.get(name, self._build_prompt_for_country(name))
            # Encode the text prompt into a CLIP-compatible embedding vector
            emb = self.encode_text(prompt)
            # Store the embedding in the cache dictionary
            self.country_embeddings[name] = emb

        # Persist the computed embeddings to disk for future runs
        torch.save(self.country_embeddings, cache_file)
        # Persist the prompts dictionary to disk as a JSON file
        with open(prompt_file, "w") as f:
            json.dump(self.country_prompts, f, indent=2, ensure_ascii=False)
        # Print a status message showing how many embeddings were saved
        print(f"Saved {len(self.country_embeddings)} country text embeddings")

    # Retrieve the text embedding for a given country (computes on-the-fly if missing)
    def get_embedding(self, country_name: str) -> torch.Tensor:
        # If the embedding is not cached, compute it now
        if country_name not in self.country_embeddings:
            # Build the prompt for this country
            prompt = self._build_prompt_for_country(country_name)
            # Encode the prompt into an embedding vector
            emb = self.encode_text(prompt)
            # Cache the embedding for future lookups
            self.country_embeddings[country_name] = emb
            # Cache the prompt string as well
            self.country_prompts[country_name] = prompt
        # Return the cached embedding tensor
        return self.country_embeddings[country_name]


# Extract and categorize all tagged visual features from the entire Plonkit database
def get_plonkit_feature_categories() -> dict[str, list[str]]:
    # Build the full Plonkit database
    db = build_database()
    # Dictionary mapping category tags to sets of unique feature text descriptions
    tag_texts: dict[str, set[str]] = {}

    # Iterate over every country entry in the database
    for entry in db.values():
        # Skip entries that aren't dictionaries or lack tagged_features
        if not isinstance(entry, dict) or "tagged_features" not in entry:
            continue
        # Iterate over each tag and its associated text descriptions
        for tag, texts in entry.get("tagged_features", {}).items():
            # Initialize a set for this tag if it hasn't been seen yet
            if tag not in tag_texts:
                tag_texts[tag] = set()
            # Process each raw text description
            for text in texts:
                # Remove wiki-style markup, footnotes, and parenthetical notes
                clean = re.sub(r"\*\*|\[.*?\]|\(.*?\)|NOTE:.*", "", text)
                # Strip leading/trailing whitespace, periods, and whitespace again
                clean = clean.strip().strip(".").strip()
                # Only keep text descriptions of a reasonable length (4-79 chars)
                if 3 < len(clean) < 80:
                    tag_texts[tag].add(clean)

    # Return categories with sorted text lists for stable output
    return {tag: sorted(texts) for tag, texts in tag_texts.items()}


# Retrieve country-specific tagged feature queries from the Plonkit database
def get_country_specific_queries(db: dict, country_code: str) -> dict[str, list[str]]:
    # Look up the country entry by its code; return empty dict if not found
    entry = db.get(country_code)
    # If the entry is missing or not a valid dict, return an empty result
    if entry is None or not isinstance(entry, dict):
        return {}
    # Get the tagged_features sub-dictionary (defaults to empty dict)
    tagged = entry.get("tagged_features", {})
    # Initialize a dictionary to hold cleaned queries per category
    queries = {}
    # Iterate over each tag category and its associated text descriptions
    for tag, texts in tagged.items():
        # Start with an empty list for this tag's cleaned queries
        queries[tag] = []
        # Take at most the first 5 text descriptions
        for text in texts[:5]:
            # Remove wiki-style markup and notes from the text
            clean = re.sub(r"\*\*|\[.*?\]|\(.*?\)|NOTE:.*", "", text)
            # Strip surrounding whitespace and common trailing punctuation
            clean = clean.strip().strip(".").strip().strip(".,;")
            # Only include texts with a reasonable character length (6-119)
            if 5 < len(clean) < 120:
                queries[tag].append(clean)
    # Return the cleaned, truncated queries dictionary
    return queries


# Build a semicolon-joined string of visual queries for Grounding DINO object detection
def build_grounding_dino_queries(db: dict, country_code: str) -> str:
    # Get the country-specific tagged feature queries
    queries = get_country_specific_queries(db, country_code)
    # List to accumulate unique text parts in priority order
    parts = []
    # Define a priority order for feature categories (most relevant first)
    priority_order = ["pole", "bollard", "chevron/sign", "guardrail", "roadline", "license plates"]
    # First, add texts from priority categories in the defined order
    for tag in priority_order:
        # Skip tags that have no queries for this country
        if tag in queries:
            # Add each text from this tag, avoiding duplicates
            for text in queries[tag]:
                if text not in parts:
                    parts.append(text)
    # Then, add texts from all remaining non-priority categories
    for tag, texts in queries.items():
        # Skip tags already handled by the priority pass
        if tag not in priority_order:
            # Add each text, avoiding duplicates
            for text in texts:
                if text not in parts:
                    parts.append(text)
    # Truncate to at most 20 parts to keep the query concise
    parts = parts[:20]
    # Join all parts with periods and a trailing period to form a single query string
    return ". ".join(parts) + "."


# Script entry point: provides CLI for showing categories or building embeddings
if __name__ == "__main__":
    # Import argparse for command-line argument parsing
    import argparse

    # Create an argument parser for the CLI script
    parser = argparse.ArgumentParser()
    # Add a flag to trigger building and caching country text embeddings
    parser.add_argument("--build-embeddings", action="store_true")
    # Add a flag to display all Plonkit feature categories
    parser.add_argument("--show-categories", action="store_true")
    # Add an option to specify the compute device (defaults to CUDA if available)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Parse the command-line arguments
    args = parser.parse_args()

    # If --show-categories flag was set, display all feature categories and samples
    if args.show_categories:
        # Retrieve all Plonkit feature categories grouped by tag
        cats = get_plonkit_feature_categories()
        # Iterate over categories sorted alphabetically by tag name
        for tag, texts in sorted(cats.items()):
            # Print the tag name and the number of features it contains
            print(f"\n[{tag}] ({len(texts)} features):")
            # Print up to 10 example text descriptions for this category
            for t in texts[:10]:
                print(f"  - {t}")

    # If --build-embeddings flag was set, precompute and cache country text embeddings
    if args.build_embeddings:
        # Import the CountryEncoder and SUBSET_DIR for building the country list
        from dataset import CountryEncoder, SUBSET_DIR

        # Create a CountryEncoder from the subset data directory
        ce = CountryEncoder(SUBSET_DIR)
        # Instantiate the PlonkitCountryEncoder on the specified device
        encoder = PlonkitCountryEncoder(device=args.device)
        # Precompute text embeddings for all countries known to the encoder
        encoder.precompute_embeddings(ce.country_list)
        # Print a completion message
        print("Done!")
