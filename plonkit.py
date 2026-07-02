import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
import requests
from tqdm import tqdm

from config import ROOT, DATASET_DIR

PLONKIT_DIR = ROOT / "plonkit_data"
PLONKIT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PLONKIT_DIR / "countries"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PLONKIT_DIR / "plonkit_db.json"

PLONKIT_API = "https://www.plonkit.net/api/guides"
PLONKIT_GUIDE_URL = "https://www.plonkit.net/{slug}"

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


def slug_to_country_name(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-"))


def fetch_json(url: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
            match = re.search(
                r'<script[^>]*id="__PRELOADED_DATA__"[^>]*type="application/json"[^>]*>\s*({.*?})\s*</script>',
                html,
                re.DOTALL,
            )
            if match:
                return json.loads(match.group(1))
            return None
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(2**attempt)
    return None


def fetch_country_list() -> list[dict]:
    try:
        resp = requests.get(PLONKIT_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return [
                g
                for g in data["data"]
                if "code" in g and not g.get("code", "").startswith("XX")
            ]
        return []
    except Exception:
        return []


def parse_country_guide(data: dict) -> dict:
    public = data.get("data", {}).get("public", {})
    country_name = public.get("title", "")
    country_code = public.get("code", "")
    slug = public.get("slug", "")
    category = public.get("cat", [])

    tagged_features: dict[str, list[str]] = {}
    all_feature_texts: list[dict] = []

    steps = public.get("steps", [])

    def walk_items(items):
        for item in items:
            if item.get("kind") == "tip":
                tags = item.get("tags", [])
                data_block = item.get("data", {})
                texts = data_block.get("text", [])
                image = data_block.get("image", {})
                image_url = image.get("imageUrl", "") if image else ""

                for tag in tags:
                    tag_lower = tag.lower().strip()
                    if tag_lower not in tagged_features:
                        tagged_features[tag_lower] = []

                feature_entry = {
                    "tags": [t.lower().strip() for t in tags],
                    "text": " ".join(t for t in texts if isinstance(t, str)),
                    "image": image_url,
                }
                all_feature_texts.append(feature_entry)

                for tag in tags:
                    tag_lower = tag.lower().strip()
                    text = " ".join(t for t in texts if isinstance(t, str))
                    if text:
                        tagged_features[tag_lower].append(text)

            if item.get("kind") == "tip" and "items" in item:
                walk_items(item["items"])
            if "items" in item:
                walk_items(item["items"])

    for step in steps:
        if "items" in step:
            walk_items(step["items"])

    return {
        "name": country_name,
        "code": country_code,
        "slug": slug,
        "category": category,
        "tagged_features": tagged_features,
        "features": all_feature_texts,
        "num_features": len(all_feature_texts),
    }


def build_database(force_refresh: bool = False) -> dict:
    if DB_PATH.exists() and not force_refresh:
        with open(DB_PATH) as f:
            db = json.load(f)
        print(f"Loaded Plonkit DB from cache ({len(db)} countries)")
        return db

    print("Fetching Plonkit country list...")
    country_list = fetch_country_list()
    print(f"Found {len(country_list)} countries with guides")

    db = {}

    for guide in tqdm(country_list, desc="Fetching country guides"):
        slug = guide["slug"]
        cache_file = CACHE_DIR / f"{slug}.json"

        if cache_file.exists() and not force_refresh:
            with open(cache_file) as f:
                raw_data = json.load(f)
        else:
            url = PLONKIT_GUIDE_URL.format(slug=slug)
            raw_data = fetch_json(url)
            if raw_data is None:
                tqdm.write(f"  SKIP {slug}: no preloaded data")
                continue
            with open(cache_file, "w") as f:
                json.dump(raw_data, f)
            time.sleep(0.3)

        parsed = parse_country_guide(raw_data)
        db[parsed["code"]] = parsed
        db[slug] = parsed

    # Also index by normalized name
    name_index = {}
    for entry in db.values():
        if isinstance(entry, dict) and "name" in entry:
            name_lower = entry["name"].lower()
            name_index[name_lower] = entry["code"]
    db["_name_index"] = name_index

    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"Plonkit DB saved: {len([k for k in db if not k.startswith('_')])} entries")
    return db


def get_country_features(db: dict, country_code: str, tags: Optional[list[str]] = None):
    entry = db.get(country_code)
    if entry is None or not isinstance(entry, dict):
        return {}
    tagged = entry.get("tagged_features", {})
    if tags is None:
        return tagged
    return {tag: tagged.get(tag, []) for tag in tags}


def get_country_by_name(db: dict, country_name: str) -> Optional[dict]:
    name_index = db.get("_name_index", {})
    name_lower = country_name.lower()
    code = name_index.get(name_lower)
    if code:
        return db.get(code)
    return None


def build_country_text_prompt(db: dict, country_code: str, max_features: int = 8) -> str:
    entry = db.get(country_code)
    if entry is None or not isinstance(entry, dict):
        return f"A street view photo from {country_code}."

    tagged = entry.get("tagged_features", {})
    priority_tags = ["pole", "chevron/sign", "roadline", "vegetation", "landscape", "architecture", "license plates", "language"]
    lines = [f"A street view photo from {entry['name']}."]

    all_clues = []
    for tag in priority_tags:
        texts = tagged.get(tag, [])
        for text in texts[:3]:
            sentence = text.split(".")[0].split("NOTE:")[0].strip()
            if len(sentence) > 10 and len(sentence) < 200:
                all_clues.append(sentence)

    if len(all_clues) > max_features:
        all_clues = all_clues[:max_features]

    if all_clues:
        lines.append("Distinctive features: " + "; ".join(all_clues) + ".")

    return " ".join(lines)


def build_feature_vocabulary(db: dict, min_countries: int = 3) -> dict:
    feature_to_countries = {}
    for entry in db.values():
        if not isinstance(entry, dict) or "code" not in entry:
            continue
        code = entry["code"]
        tagged = entry.get("tagged_features", {})
        for tag, texts in tagged.items():
            for text in texts:
                key = text[:120].strip().lower()
                if key not in feature_to_countries:
                    feature_to_countries[key] = set()
                feature_to_countries[key].add(code)

    vocab = {
        feat: sorted(countries)
        for feat, countries in feature_to_countries.items()
        if len(countries) >= min_countries
    }
    return vocab


def encode_country_vector(db: dict, country_code: str, feature_vocab: list[str]) -> torch.Tensor:
    entry = db.get(country_code)
    if entry is None or not isinstance(entry, dict):
        return torch.zeros(len(feature_vocab))
    vec = torch.zeros(len(feature_vocab))
    tagged = entry.get("tagged_features", {})
    for tag, texts in tagged.items():
        for text in texts:
            key = text[:120].strip().lower()
            if key in feature_vocab:
                vec[feature_vocab.index(key)] = 1.0
    return vec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--country", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    db = build_database(force_refresh=args.refresh)

    if args.stats:
        codes = [k for k in db if not k.startswith("_") and isinstance(db[k], dict)]
        print(f"\nCountries: {len(codes)}")
        non_empty = sum(
            1 for c in codes
            if isinstance(db[c], dict) and db[c].get("num_features", 0) > 0
        )
        print(f"With features: {non_empty}")
        all_tags = set()
        for c in codes:
            if isinstance(db[c], dict):
                all_tags.update(db[c].get("tagged_features", {}).keys())
        print(f"Unique tags: {sorted(all_tags)}")
        for tag in sorted(all_tags):
            count = sum(
                1 for c in codes
                if isinstance(db[c], dict) and tag in db[c].get("tagged_features", {})
            )
            print(f"  {tag}: {count} countries")

    if args.country:
        entry = get_country_by_name(db, args.country)
        if entry:
            print(f"\n{entry['name']} ({entry['code']}):")
            for tag, texts in entry.get("tagged_features", {}).items():
                print(f"  [{tag}] ({len(texts)} clues)")
        else:
            # Try by code
            entry = db.get(args.country.upper())
            if entry and isinstance(entry, dict):
                print(f"\n{entry['name']} ({entry['code']}):")
                for tag, texts in entry.get("tagged_features", {}).items():
                    print(f"  [{tag}] ({len(texts)} clues)")

    if args.prompt:
        prompt = build_country_text_prompt(db, args.prompt.upper())
        print(f"\nPROMPT: {prompt}")
