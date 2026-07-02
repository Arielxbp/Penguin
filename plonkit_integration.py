import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

from config import (
    ROOT,
    STREETCLIP_MODEL,
    EMBEDDING_DIR,
)
from plonkit import (
    build_database,
    build_country_text_prompt,
    get_country_by_name,
    PLONKIT_DIR,
)

COUNTRY_TEXT_DIR = EMBEDDING_DIR / "country_texts"
COUNTRY_TEXT_DIR.mkdir(parents=True, exist_ok=True)

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


def resolve_country_in_plonkit(db: dict, country_name: str) -> Optional[dict]:
    entry = get_country_by_name(db, country_name)
    if entry:
        return entry
    mapped = COUNTRY_NAME_MAP.get(country_name)
    if mapped:
        entry = get_country_by_name(db, mapped)
        if entry:
            return entry
    name_lower = country_name.lower().strip()
    for key, value in db.items():
        if isinstance(value, dict) and value.get("name", "").lower() == name_lower:
            return value
    return None


class PlonkitCountryEncoder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.db = build_database()
        self._text_model = None
        self._tokenizer = None
        self.country_embeddings: dict[str, torch.Tensor] = {}
        self.country_prompts: dict[str, str] = {}

    def _load_text_encoder(self):
        if self._text_model is not None:
            return
        model = CLIPModel.from_pretrained(STREETCLIP_MODEL).to(self.device)
        self._text_model = model.text_model
        self._text_projection = model.text_projection
        self._tokenizer = CLIPProcessor.from_pretrained(STREETCLIP_MODEL).tokenizer

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        self._load_text_encoder()
        tokens = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        text_outputs = self._text_model(**tokens)
        pooled = text_outputs.pooler_output
        embedding = self._text_projection(pooled)
        return embedding.squeeze(0).cpu()

    def build_prompts(self, country_names: list[str]):
        for name in tqdm(country_names, desc="Building Plonkit prompts"):
            prompt = self._build_prompt_for_country(name)
            self.country_prompts[name] = prompt

    def _build_prompt_for_country(self, country_name: str) -> str:
        entry = resolve_country_in_plonkit(self.db, country_name)
        if entry:
            return build_country_text_prompt(self.db, entry["code"])
        return f"A street view photo from {country_name}."

    @torch.no_grad()
    def precompute_embeddings(self, country_names: list[str]):
        self.build_prompts(country_names)
        cache_file = COUNTRY_TEXT_DIR / "country_embeddings.pt"
        prompt_file = COUNTRY_TEXT_DIR / "country_prompts.json"

        if cache_file.exists() and prompt_file.exists():
            self.country_embeddings = torch.load(cache_file, weights_only=True)
            with open(prompt_file) as f:
                self.country_prompts = json.load(f)
            print(f"Loaded {len(self.country_embeddings)} cached country embeddings")
            return

        self._load_text_encoder()
        for name in tqdm(country_names, desc="Encoding Plonkit prompts"):
            prompt = self.country_prompts.get(name, self._build_prompt_for_country(name))
            emb = self.encode_text(prompt)
            self.country_embeddings[name] = emb

        torch.save(self.country_embeddings, cache_file)
        with open(prompt_file, "w") as f:
            json.dump(self.country_prompts, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.country_embeddings)} country text embeddings")

    def get_embedding(self, country_name: str) -> torch.Tensor:
        if country_name not in self.country_embeddings:
            prompt = self._build_prompt_for_country(country_name)
            emb = self.encode_text(prompt)
            self.country_embeddings[country_name] = emb
            self.country_prompts[country_name] = prompt
        return self.country_embeddings[country_name]


def get_plonkit_feature_categories() -> dict[str, list[str]]:
    db = build_database()
    tag_texts: dict[str, set[str]] = {}

    for entry in db.values():
        if not isinstance(entry, dict) or "tagged_features" not in entry:
            continue
        for tag, texts in entry.get("tagged_features", {}).items():
            if tag not in tag_texts:
                tag_texts[tag] = set()
            for text in texts:
                clean = re.sub(r"\*\*|\[.*?\]|\(.*?\)|NOTE:.*", "", text)
                clean = clean.strip().strip(".").strip()
                if 3 < len(clean) < 80:
                    tag_texts[tag].add(clean)

    return {tag: sorted(texts) for tag, texts in tag_texts.items()}


def get_country_specific_queries(db: dict, country_code: str) -> dict[str, list[str]]:
    entry = db.get(country_code)
    if entry is None or not isinstance(entry, dict):
        return {}
    tagged = entry.get("tagged_features", {})
    queries = {}
    for tag, texts in tagged.items():
        queries[tag] = []
        for text in texts[:5]:
            clean = re.sub(r"\*\*|\[.*?\]|\(.*?\)|NOTE:.*", "", text)
            clean = clean.strip().strip(".").strip().strip(".,;")
            if 5 < len(clean) < 120:
                queries[tag].append(clean)
    return queries


def build_grounding_dino_queries(db: dict, country_code: str) -> str:
    queries = get_country_specific_queries(db, country_code)
    parts = []
    priority_order = ["pole", "bollard", "chevron/sign", "guardrail", "roadline", "license plates"]
    for tag in priority_order:
        if tag in queries:
            for text in queries[tag]:
                if text not in parts:
                    parts.append(text)
    for tag, texts in queries.items():
        if tag not in priority_order:
            for text in texts:
                if text not in parts:
                    parts.append(text)
    parts = parts[:20]
    return ". ".join(parts) + "."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-embeddings", action="store_true")
    parser.add_argument("--show-categories", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.show_categories:
        cats = get_plonkit_feature_categories()
        for tag, texts in sorted(cats.items()):
            print(f"\n[{tag}] ({len(texts)} features):")
            for t in texts[:10]:
                print(f"  - {t}")

    if args.build_embeddings:
        from dataset import CountryEncoder, SUBSET_DIR

        ce = CountryEncoder(SUBSET_DIR)
        encoder = PlonkitCountryEncoder(device=args.device)
        encoder.precompute_embeddings(ce.country_list)
        print("Done!")
