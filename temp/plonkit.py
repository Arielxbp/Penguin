# Import json module for parsing and writing JSON data files
import json
# Import re module for regular expression pattern matching on HTML content
import re
# Import time module for sleeping between API requests to respect rate limits
import time
# Import Path from pathlib for convenient filesystem path manipulation
from pathlib import Path
# Import Optional type hint for optional function parameters and return values
from typing import Optional

# Import PyTorch for tensor creation and numerical operations
import torch
# Import requests library for making HTTP GET requests to the Plonkit API
import requests
# Import tqdm for displaying progress bars during iterative operations
from tqdm import tqdm

# Import root directory and dataset directory paths from the project configuration
from config import ROOT, DATASET_DIR

# Define the base directory for storing Plonkit cached data and the database
PLONKIT_DIR = ROOT / "plonkit_data"
# Create the Plonkit data directory (and any missing parent directories) if it doesn't exist
PLONKIT_DIR.mkdir(parents=True, exist_ok=True)
# Define the subdirectory for caching individual country guide JSON files
CACHE_DIR = PLONKIT_DIR / "countries"
# Create the country cache directory (and any missing parent directories) if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Define the file path for the compiled Plonkit database JSON file
DB_PATH = PLONKIT_DIR / "plonkit_db.json"

# Base URL for the Plonkit API endpoint that returns the list of country guides
PLONKIT_API = "https://www.plonkit.net/api/guides"
# Template URL for individual country guide pages, with slug as a format placeholder
PLONKIT_GUIDE_URL = "https://www.plonkit.net/{slug}"

# Set of leaf-node tag names representing distinct GeoGuessr clue categories
LEAF_TAGS = {
    "license plates",
    "language",
    "landscape",
    "vegetation",
    "pole",
    "chevron/sign",
    "roadline",
    "architecture",
    "coverage",
    "moving info",
}


# Convert a URL slug (e.g. "united-states") into a readable country name ("United States")
def slug_to_country_name(slug: str) -> str:
    # Split the slug on hyphens, capitalize each word, and join with spaces
    return " ".join(part.capitalize() for part in slug.split("-"))


# Fetch and parse the preloaded JSON data embedded in a Plonkit guide page's HTML
def fetch_json(url: str, retries: int = 3) -> Optional[dict]:
    # Retry the HTTP request up to the specified number of attempts
    for attempt in range(retries):
        # Attempt to make the request and catch any exceptions
        try:
            # Send an HTTP GET request to the URL with a 30-second timeout
            resp = requests.get(url, timeout=30)
            # Raise an HTTPError if the response status code indicates a client or server error
            resp.raise_for_status()
            # Get the raw HTML text content of the response
            html = resp.text
            # Search for the __PRELOADED_DATA__ script tag containing embedded JSON data
            match = re.search(
                r'<script[^>]*id="__PRELOADED_DATA__"[^>]*type="application/json"[^>]*>\s*({.*?})\s*</script>',
                html,
                re.DOTALL,
            )
            # If the regex pattern found a match in the HTML
            if match:
                # Parse the captured JSON string from group 1 and return the resulting dict
                return json.loads(match.group(1))
            # Return None if the preloaded data script tag was not found in the page
            return None
        # Catch any exception that occurred during the request or parsing
        except Exception as e:
            # On the last retry attempt, give up and return None
            if attempt == retries - 1:
                return None
            # Wait with exponential backoff (1s, 2s, 4s, ...) before the next retry
            time.sleep(2**attempt)
    # Return None if all retries were exhausted without success
    return None


# Fetch the list of available country guides from the Plonkit API
def fetch_country_list() -> list[dict]:
    # Attempt to make the API request and catch any exceptions gracefully
    try:
        # Send an HTTP GET request to the Plonkit API endpoint with a 30-second timeout
        resp = requests.get(PLONKIT_API, timeout=30)
        # Raise an HTTPError if the response status code indicates a failure
        resp.raise_for_status()
        # Parse the JSON response body into a Python dictionary
        data = resp.json()
        # Check if the API response indicates success
        if data.get("success"):
            # Return a filtered list of guides that have a country code and exclude XX-prefixed test entries
            return [
                g
                for g in data["data"]
                if "code" in g and not g.get("code", "").startswith("XX")
            ]
        # Return an empty list if the API response does not indicate success
        return []
    # Catch any exception (network error, JSON decode error, etc.) and return an empty list
    except Exception:
        # Return an empty list to gracefully handle API failures
        return []


# Parse a raw Plonkit country guide JSON into a structured dictionary of features and metadata
def parse_country_guide(data: dict) -> dict:
    # Navigate to the public-facing data section of the guide JSON
    public = data.get("data", {}).get("public", {})
    # Extract the human-readable country or region name
    country_name = public.get("title", "")
    # Extract the two-letter country code (or region code)
    country_code = public.get("code", "")
    # Extract the URL slug used to identify the guide
    slug = public.get("slug", "")
    # Extract the category tags associated with this guide
    category = public.get("cat", [])

    # Dictionary mapping lowercase tag names to lists of associated feature text strings
    tagged_features: dict[str, list[str]] = {}
    # List of all individual feature entries with their tags, text, and image URLs
    all_feature_texts: list[dict] = []

    # Extract the list of instructional steps from the guide
    steps = public.get("steps", [])

    # Recursive inner function to walk through nested tip items and extract feature data
    def walk_items(items):
        # Iterate over each item in the current list of guide items
        for item in items:
            # Only process items that are of kind "tip" (feature clues)
            if item.get("kind") == "tip":
                # Extract the list of tag strings assigned to this tip
                tags = item.get("tags", [])
                # Extract the data block containing text and image information
                data_block = item.get("data", {})
                # Extract the list of text content entries from the data block
                texts = data_block.get("text", [])
                # Extract the optional image metadata dictionary
                image = data_block.get("image", {})
                # Get the image URL if an image entry exists, otherwise empty string
                image_url = image.get("imageUrl", "") if image else ""

                # Initialize an empty list for each new tag encountered
                for tag in tags:
                    # Normalize the tag to lowercase and strip surrounding whitespace
                    tag_lower = tag.lower().strip()
                    # Create an empty list entry for this tag if it hasn't been seen before
                    if tag_lower not in tagged_features:
                        tagged_features[tag_lower] = []

                # Build a feature entry dictionary with normalized tags, joined text, and image URL
                feature_entry = {
                    # Normalize all tags to lowercase with whitespace stripped
                    "tags": [t.lower().strip() for t in tags],
                    # Join all string-type text entries with a space separator
                    "text": " ".join(t for t in texts if isinstance(t, str)),
                    # Store the image URL for this feature
                    "image": image_url,
                }
                # Append this feature entry to the flat list of all features
                all_feature_texts.append(feature_entry)

                # Associate each tag with the joined text for quick lookup by tag
                for tag in tags:
                    # Normalize the tag to lowercase and strip surrounding whitespace
                    tag_lower = tag.lower().strip()
                    # Join all string-type text entries with a space separator
                    text = " ".join(t for t in texts if isinstance(t, str))
                    # Only store the text if it is non-empty
                    if text:
                        # Append this text to the list of feature texts for this tag
                        tagged_features[tag_lower].append(text)

            # If this item is a tip and contains nested items, recursively walk them
            if item.get("kind") == "tip" and "items" in item:
                # Recursively process the nested items inside this tip
                walk_items(item["items"])
            # If this item contains nested items (regardless of kind), recursively walk them
            if "items" in item:
                # Recursively process the nested items
                walk_items(item["items"])

    # Iterate over each top-level step in the guide
    for step in steps:
        # If this step contains nested items, recursively walk through them
        if "items" in step:
            # Begin recursive traversal of items within this step
            walk_items(step["items"])

    # Return a structured dictionary with all parsed country guide information
    return {
        # Human-readable country or region name
        "name": country_name,
        # Two-letter country or region code
        "code": country_code,
        # URL slug identifier for the guide
        "slug": slug,
        # List of category tags for this guide
        "category": category,
        # Dictionary mapping lowercase tag names to their feature text lists
        "tagged_features": tagged_features,
        # Flat list of all feature entries with tags, text, and image URLs
        "features": all_feature_texts,
        # Total count of feature entries found in this guide
        "num_features": len(all_feature_texts),
    }


# Build or load the complete Plonkit database, fetching guides from the web if needed
def build_database(force_refresh: bool = False) -> dict:
    # If the database file already exists locally and a forced refresh is not requested
    if DB_PATH.exists() and not force_refresh:
        # Open the cached database JSON file for reading
        with open(DB_PATH) as f:
            # Load and parse the JSON content into a Python dictionary
            db = json.load(f)
        # Print a status message indicating the database was loaded from cache
        print(f"Loaded Plonkit DB from cache ({len(db)} countries)")
        # Return the loaded database dictionary
        return db

    # Print a status message indicating that country guides are being fetched
    print("Fetching Plonkit country list...")
    # Fetch the list of available country guides from the Plonkit API
    country_list = fetch_country_list()
    # Print the number of countries found with available guides
    print(f"Found {len(country_list)} countries with guides")

    # Initialize an empty dictionary for the database
    db = {}

    # Iterate over each country guide with a tqdm progress bar
    for guide in tqdm(country_list, desc="Fetching country guides"):
        # Extract the URL slug for this country from the guide entry
        slug = guide["slug"]
        # Build the filesystem path for the cached JSON file for this country
        cache_file = CACHE_DIR / f"{slug}.json"

        # If the cached file already exists and a forced refresh is not requested
        if cache_file.exists() and not force_refresh:
            # Open the cached JSON file for reading
            with open(cache_file) as f:
                # Load the raw guide data from the cache file
                raw_data = json.load(f)
        else:
            # Format the full guide URL using the country's slug
            url = PLONKIT_GUIDE_URL.format(slug=slug)
            # Fetch and parse the raw guide data from the web
            raw_data = fetch_json(url)
            # If the fetch failed (returned None), skip this country
            if raw_data is None:
                # Print a skip message without disturbing the tqdm progress bar
                tqdm.write(f"  SKIP {slug}: no preloaded data")
                # Continue to the next country guide in the loop
                continue
            # Open the cache file for writing to save the fetched data locally
            with open(cache_file, "w") as f:
                # Write the raw guide data as formatted JSON to the cache file
                json.dump(raw_data, f)
            # Sleep for 300ms to avoid hitting the server with too many rapid requests
            time.sleep(0.3)

        # Parse the raw guide data into a structured dictionary
        parsed = parse_country_guide(raw_data)
        # Store the parsed guide under its country code as the primary key
        db[parsed["code"]] = parsed
        # Also store the parsed guide under its URL slug as an alternative lookup key
        db[slug] = parsed

    # Build a name-based lookup index for finding countries by human-readable name
    # Also index by normalized name
    name_index = {}
    # Iterate over all database entries to build the name index
    for entry in db.values():
        # Only index dictionary entries that contain a "name" field
        if isinstance(entry, dict) and "name" in entry:
            # Normalize the country name to lowercase for case-insensitive lookup
            name_lower = entry["name"].lower()
            # Map the lowercase name to the country code for reverse lookup
            name_index[name_lower] = entry["code"]
    # Store the name index in the database under a special metadata key
    db["_name_index"] = name_index

    # Open the database file for writing
    with open(DB_PATH, "w") as f:
        # Write the complete database as formatted JSON, preserving non-ASCII characters
        json.dump(db, f, indent=2, ensure_ascii=False)

    # Print a status message with the number of real entries (excluding metadata keys starting with '_')
    print(f"Plonkit DB saved: {len([k for k in db if not k.startswith('_')])} entries")
    # Return the complete database dictionary
    return db


# Retrieve the tagged features for a specific country, optionally filtered by desired tags
def get_country_features(db: dict, country_code: str, tags: Optional[list[str]] = None):
    # Look up the country entry in the database by its code
    entry = db.get(country_code)
    # If no entry exists or the value is not a dictionary, return an empty dict
    if entry is None or not isinstance(entry, dict):
        return {}
    # Extract the dictionary of tagged features from the country entry
    tagged = entry.get("tagged_features", {})
    # If no specific tags are requested, return all tagged features
    if tags is None:
        return tagged
    # Return a filtered dictionary with only the requested tags (empty list for missing tags)
    return {tag: tagged.get(tag, []) for tag in tags}


# Look up a country entry by its human-readable name using the name index
def get_country_by_name(db: dict, country_name: str) -> Optional[dict]:
    # Retrieve the name-to-code lookup index from the database metadata
    name_index = db.get("_name_index", {})
    # Normalize the query name to lowercase for case-insensitive matching
    name_lower = country_name.lower()
    # Look up the country code by its lowercase name in the index
    code = name_index.get(name_lower)
    # If a matching country code was found, return the full country entry
    if code:
        return db.get(code)
    # Return None if no matching country was found by name
    return None


# Build a natural language text prompt describing distinctive features of a country for GeoGuessr
def build_country_text_prompt(db: dict, country_code: str, max_features: int = 8) -> str:
    # Look up the country entry in the database by its code
    entry = db.get(country_code)
    # If no entry exists or the value is not a dictionary, return a generic fallback prompt
    if entry is None or not isinstance(entry, dict):
        return f"A street view photo from {country_code}."

    # Extract the dictionary of tagged features from the country entry
    tagged = entry.get("tagged_features", {})
    # Define a prioritized list of tag categories to include in the prompt
    priority_tags = ["pole", "chevron/sign", "roadline", "vegetation", "landscape", "architecture", "license plates", "language"]
    # Start building the prompt lines with a basic description of the location
    lines = [f"A street view photo from {entry['name']}."]

    # Collect feature clue sentences from the prioritized tags
    all_clues = []
    # Iterate over each tag in priority order
    for tag in priority_tags:
        # Get the list of feature text strings for this tag (empty list if tag not present)
        texts = tagged.get(tag, [])
        # Take up to the first 3 feature texts for each priority tag
        for text in texts[:3]:
            # Extract the first sentence before a period, and strip any "NOTE:" prefix
            sentence = text.split(".")[0].split("NOTE:")[0].strip()
            # Only include sentences that are between 10 and 200 characters long (meaningful clues)
            if len(sentence) > 10 and len(sentence) < 200:
                # Append this clue sentence to the collected list
                all_clues.append(sentence)

    # If the number of collected clues exceeds the maximum allowed
    if len(all_clues) > max_features:
        # Truncate the clues list to the maximum allowed number
        all_clues = all_clues[:max_features]

    # If there are any clue sentences to include
    if all_clues:
        # Append a line describing distinctive features with semicolon-joined clues
        lines.append("Distinctive features: " + "; ".join(all_clues) + ".")

    # Join all prompt lines with a space and return the complete prompt string
    return " ".join(lines)


# Build a feature vocabulary mapping feature text to the set of countries that share it
def build_feature_vocabulary(db: dict, min_countries: int = 3) -> dict:
    # Dictionary mapping feature text keys to the set of country codes where the feature appears
    feature_to_countries = {}
    # Iterate over all entries in the database
    for entry in db.values():
        # Skip non-dictionary entries and entries without a country code
        if not isinstance(entry, dict) or "code" not in entry:
            continue
        # Extract the country code for this entry
        code = entry["code"]
        # Get the tagged features dictionary for this country
        tagged = entry.get("tagged_features", {})
        # Iterate over each tag and its associated list of feature text strings
        for tag, texts in tagged.items():
            # Process each individual feature text string
            for text in texts:
                # Truncate the text to 120 characters, strip whitespace, and normalize to lowercase as the key
                key = text[:120].strip().lower()
                # If this feature key hasn't been seen before, initialize an empty set of countries
                if key not in feature_to_countries:
                    feature_to_countries[key] = set()
                # Add this country's code to the set of countries that share this feature
                feature_to_countries[key].add(code)

    # Build the final vocabulary: keep only features shared by at least min_countries countries
    vocab = {
        feat: sorted(countries)
        for feat, countries in feature_to_countries.items()
        if len(countries) >= min_countries
    }
    # Return the filtered feature vocabulary
    return vocab


# Encode a country's features as a binary vector using a pre-built feature vocabulary
def encode_country_vector(db: dict, country_code: str, feature_vocab: list[str]) -> torch.Tensor:
    # Look up the country entry in the database by its code
    entry = db.get(country_code)
    # If no entry exists or the value is not a dictionary, return an all-zeros vector
    if entry is None or not isinstance(entry, dict):
        return torch.zeros(len(feature_vocab))
    # Initialize a zero tensor with one element per vocabulary feature
    vec = torch.zeros(len(feature_vocab))
    # Get the tagged features dictionary for this country
    tagged = entry.get("tagged_features", {})
    # Iterate over each tag and its associated list of feature text strings
    for tag, texts in tagged.items():
        # Process each individual feature text string
        for text in texts:
            # Truncate the text to 120 characters, strip whitespace, and normalize to lowercase as the key
            key = text[:120].strip().lower()
            # If this feature key exists in the vocabulary
            if key in feature_vocab:
                # Set the corresponding vector element to 1.0 to indicate presence of this feature
                vec[feature_vocab.index(key)] = 1.0
    # Return the binary feature vector
    return vec


# Execute the script directly (not imported) to provide CLI functionality for the Plonkit database
if __name__ == "__main__":
    # Import argparse for parsing command-line arguments
    import argparse

    # Create an argument parser with the script's name as the program identifier
    parser = argparse.ArgumentParser()
    # Add a boolean flag to force refreshing all data from the web
    parser.add_argument("--refresh", action="store_true")
    # Add a boolean flag to print database statistics
    parser.add_argument("--stats", action="store_true")
    # Add an optional argument to query a specific country by name or code
    parser.add_argument("--country", default=None)
    # Add an optional argument to generate a text prompt for a country
    parser.add_argument("--prompt", default=None)
    # Parse the command-line arguments into the args namespace
    args = parser.parse_args()

    # Build or load the Plonkit database, optionally forcing a refresh from the web
    db = build_database(force_refresh=args.refresh)

    # If the --stats flag was provided, print database statistics
    if args.stats:
        # Collect all keys that are real country entries (not metadata keys starting with '_')
        codes = [k for k in db if not k.startswith("_") and isinstance(db[k], dict)]
        # Print the total number of countries in the database
        print(f"\nCountries: {len(codes)}")
        # Count how many countries have at least one feature entry
        non_empty = sum(
            1 for c in codes
            if isinstance(db[c], dict) and db[c].get("num_features", 0) > 0
        )
        # Print the number of countries that have feature data
        print(f"With features: {non_empty}")
        # Collect all unique tag names across all countries
        all_tags = set()
        # Iterate over all country codes
        for c in codes:
            # Only process dictionary entries
            if isinstance(db[c], dict):
                # Add all tag keys from this country to the set of all tags
                all_tags.update(db[c].get("tagged_features", {}).keys())
        # Print the sorted list of all unique tag names
        print(f"Unique tags: {sorted(all_tags)}")
        # Print the count of countries for each tag
        for tag in sorted(all_tags):
            # Count how many countries have this specific tag
            count = sum(
                1 for c in codes
                if isinstance(db[c], dict) and tag in db[c].get("tagged_features", {})
            )
            # Print the tag name and the number of countries that have it
            print(f"  {tag}: {count} countries")

    # If the --country flag was provided, look up and display country information
    if args.country:
        # Try to find the country by its human-readable name first
        entry = get_country_by_name(db, args.country)
        # If a matching entry was found by name
        if entry:
            # Print the country name and code header
            print(f"\n{entry['name']} ({entry['code']}):")
            # Iterate over each tag and its associated feature texts
            for tag, texts in entry.get("tagged_features", {}).items():
                # Print the tag name and the number of clue texts for that tag
                print(f"  [{tag}] ({len(texts)} clues)")
        else:
            # Try by code
            # Attempt to look up the country entry directly by uppercase country code
            entry = db.get(args.country.upper())
            # If a valid dictionary entry was found by code
            if entry and isinstance(entry, dict):
                # Print the country name and code header
                print(f"\n{entry['name']} ({entry['code']}):")
                # Iterate over each tag and its associated feature texts
                for tag, texts in entry.get("tagged_features", {}).items():
                    # Print the tag name and the number of clue texts for that tag
                    print(f"  [{tag}] ({len(texts)} clues)")

    # If the --prompt flag was provided, generate and print a text prompt for the country
    if args.prompt:
        # Build a natural language prompt for the requested country code (uppercased)
        prompt = build_country_text_prompt(db, args.prompt.upper())
        # Print the generated prompt string
        print(f"\nPROMPT: {prompt}")
