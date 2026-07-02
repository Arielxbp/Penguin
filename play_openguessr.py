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

import argparse
import asyncio
import json
import math
import random
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    CHECKPOINT_DIR,
    FUSION_OUTPUT_DIM,
    OBJ_FEATURE_DIM,
    VEG_FEATURE_DIM,
    OUTPUT_DIR,
    SUBSET_DIR,
)
from model import StreetCLIPFusion
from dataset import CountryEncoder, BASE_TRANSFORM
from utils import set_seed

set_seed(42)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLAY_DIR = OUTPUT_DIR / "play"
RUNS_DIR = PLAY_DIR / "runs"
ROUNDS_DIR = PLAY_DIR / "rounds"

GAME_URL = "https://openguessr.com"

ANTI_DETECT_SCRIPT = """
delete Object.getPrototypeOf(navigator).webdriver;
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
"""

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


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_score(km):
    if km <= 0:
        return 5000
    return round(5000 * math.exp(-km / 2000))


def format_distance(km):
    if km < 1:
        return f"{km * 1000:.0f} m"
    if km < 1000:
        return f"{km:.1f} km"
    return f"{km / 1000:.1f}K km"


def now_iso():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Penguin model wrapper
# ---------------------------------------------------------------------------

class PenguinAI:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.centroids = {}
        self.country_list = []
        self._all_coords = {}
        self.country_coords = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"Loading Penguin model on {self.device}...")
        self.model = StreetCLIPFusion(
            freeze_backbone=True, fusion_output_dim=FUSION_OUTPUT_DIM,
        )
        ckpt = CHECKPOINT_DIR / "best_model.pt"
        if ckpt.exists():
            state = torch.load(ckpt, weights_only=True, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
            print(f"  checkpoint  : {ckpt.name}")
        else:
            print("  WARNING: no checkpoint found")

        self.model = self.model.to(self.device)
        self.model.eval()

        data_dir = SUBSET_DIR
        self.country_list = CountryEncoder(data_dir).country_list

        centroid_dir = OUTPUT_DIR / "centroids"
        centroid_path = None
        if centroid_dir.exists():
            for p in sorted(centroid_dir.glob("centroids_*.pt")):
                centroid_path = p
                break
        if centroid_path:
            self.centroids = torch.load(centroid_path, weights_only=True,
                                        map_location="cpu")
            for k in self.centroids:
                if (self.centroids[k].ndim == 2
                        and self.centroids[k].shape[0] == 1):
                    self.centroids[k] = self.centroids[k].squeeze(0)
            print(f"  centroids   : {len(self.centroids)} countries")

        self._build_coord_map(data_dir)
        print(f"  coordinates : {len(self.country_coords)} countries")
        self._loaded = True

    def _build_coord_map(self, data_dir):
        coords = {}
        for jf in sorted(data_dir.glob("location_*.json")):
            try:
                with open(jf) as f:
                    d = json.load(f)
                if not isinstance(d, dict):
                    continue
                c = d.get("country_name", "Unknown")
                ll = d.get("coordinates", [0.0, 0.0])
                if len(ll) != 2:
                    continue
                lat, lng = float(ll[0]), float(ll[1])
                if c not in coords:
                    coords[c] = []
                coords[c].append((lat, lng))
            except (json.JSONDecodeError, Exception):
                continue
        self._all_coords = coords
        self.country_coords = {
            c: (np.median([p[0] for p in pts]),
                np.median([p[1] for p in pts]))
            for c, pts in coords.items()
        }

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        pixel_values = BASE_TRANSFORM(image).unsqueeze(0).to(self.device)
        road_f = torch.zeros(1, OBJ_FEATURE_DIM).to(self.device)
        veg_f = torch.zeros(1, VEG_FEATURE_DIM).to(self.device)
        emb = self.model(
            pixel_values=pixel_values, road_features=road_f, veg_features=veg_f,
        ).cpu()

        valid = [c for c in self.country_list if c in self.centroids]
        if not valid:
            return [("Unknown", 0.0, (0.0, 0.0))]

        matrix = torch.stack([self.centroids[c] for c in valid])
        sim = torch.matmul(emb, matrix.T).squeeze(0)
        topk = sim.argsort(descending=True)[:top_k].numpy()
        return [(valid[i], float(sim[i]), self._coords_for(valid[i]))
                for i in topk]

    def _coords_for(self, country):
        if country in self.country_coords:
            return self.country_coords[country]
        nl = country.lower()
        for k, v in self.country_coords.items():
            if k.lower() == nl or nl in k.lower() or k.lower() in nl:
                return v
        return (0.0, 0.0)

    def random_coords(self):
        if not self.country_coords:
            return (0.0, 0.0)
        return random.choice(list(self.country_coords.values()))


# ---------------------------------------------------------------------------
# Raw StreetCLIP baseline for benchmarking — uses CLIPProcessor + model(**inputs)
# like the official Hugging Face example, no manual embedding computation.
# ---------------------------------------------------------------------------

class StreetCLIPBaseline:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.country_list = []
        self._all_coords = {}
        self.country_coords = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"Loading StreetCLIP baseline on {self.device}...")
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(
            self.device)
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        self.model.eval()

        data_dir = SUBSET_DIR
        self.country_list = CountryEncoder(data_dir).country_list
        print(f"  countries   : {len(self.country_list)}")

        self._build_coord_map(data_dir)
        print(f"  coordinates : {len(self.country_coords)} countries")
        self._loaded = True

    def _build_coord_map(self, data_dir):
        coords = {}
        for jf in sorted(data_dir.glob("location_*.json")):
            try:
                with open(jf) as f:
                    d = json.load(f)
                if not isinstance(d, dict):
                    continue
                c = d.get("country_name", "Unknown")
                ll = d.get("coordinates", [0.0, 0.0])
                if len(ll) != 2:
                    continue
                lat, lng = float(ll[0]), float(ll[1])
                if c not in coords:
                    coords[c] = []
                coords[c].append((lat, lng))
            except (json.JSONDecodeError, Exception):
                continue
        self._all_coords = coords
        self.country_coords = {
            c: (np.median([p[0] for p in pts]),
                np.median([p[1] for p in pts]))
            for c, pts in coords.items()
        }

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        inputs = self.processor(
            text=self.country_list, images=image, return_tensors="pt",
            padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)
        topk = probs.argsort(descending=True)[:top_k].cpu().numpy()
        return [(self.country_list[i], float(probs[i]),
                 self._coords_for(self.country_list[i]))
                for i in topk]

    def _coords_for(self, country):
        if country in self.country_coords:
            return self.country_coords[country]
        nl = country.lower()
        for k, v in self.country_coords.items():
            if k.lower() == nl or nl in k.lower() or k.lower() in nl:
                return v
        return (0.0, 0.0)


def find_country_for_coords(lat, lng, all_coords):
    best = None
    best_dist = float("inf")
    for country, pts in all_coords.items():
        for clat, clng in pts:
            d = haversine_km(lat, lng, clat, clng)
            if d < best_dist:
                best_dist = d
                best = country
    return best


def _compute_accuracy(top_k_countries, true_country):
    if not true_country or not top_k_countries:
        return False, False
    top1 = top_k_countries[0] == true_country
    top5 = true_country in top_k_countries
    return top1, top5


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

async def _screenshot_panorama(page):
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
    data = await page.screenshot(type="png")
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
    return data


async def _read_true_location(page):
    try:
        src = await page.evaluate(
            "() => document.querySelector('#panorama-iframe')?.src || null")
        if not src:
            return None
        loc = parse_qs(urlparse(src).query).get("location", [None])[0]
        if not loc:
            return None
        parts = loc.split(",")
        if len(parts) >= 2:
            return (float(parts[0].strip()), float(parts[1].strip()))
    except Exception:
        pass
    return None


async def _dismiss_cookies(page):
    for _ in range(5):
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
        await asyncio.sleep(random.uniform(0.3, 0.6))
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

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
        if gone:
            print("  cookie consent dismissed")
            return
        await asyncio.sleep(0.5)


async def _hover_minimap(page):
    for sel in [".leaflet-container", '[class*="guess-map"]', "#map"]:
        try:
            el = await page.query_selector(sel)
            if not el:
                continue
            box = await page.evaluate(f"""() => {{
                const el = document.querySelector('{sel}');
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {{left:r.left,top:r.top,w:r.width,h:r.height}};
            }}""")
            if not box or box["w"] <= 0 or box["h"] <= 0:
                continue
            break
        except Exception:
            continue
    else:
        return

    cx = box["left"] + box["w"] / 2
    cy = box["top"] + box["h"] / 2
    await page.mouse.move(random.randint(100, 800), random.randint(100, 600))
    await asyncio.sleep(0.1)
    for i in range(random.randint(6, 12)):
        t = (i + 1) / random.randint(8, 14)
        await page.mouse.move(
            int(box["left"] + box["w"] / 2 * t + random.uniform(-3, 3)),
            int(box["top"] + box["h"] / 2 * t + random.uniform(-3, 3)),
        )
        await asyncio.sleep(random.uniform(0.01, 0.03))
    await asyncio.sleep(1.0)


async def _reset_minimap(page):
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
    if ok:
        await asyncio.sleep(0.35)
        return
    try:
        btn = await page.query_selector(
            ".leaflet-control-zoom-out, a.leaflet-control-zoom-out")
        if btn:
            for _ in range(8):
                try:
                    await btn.click(timeout=500)
                    await asyncio.sleep(0.12)
                except Exception:
                    break
            await asyncio.sleep(0.3)
    except Exception:
        pass


async def _click_on_map(page, lat, lng):
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

    if c:
        await asyncio.sleep(0.25)
        x = max(c["left"] + 4, min(c["x"], c["left"] + c["w"] - 4))
        y = max(c["top"] + 4, min(c["y"], c["top"] + c["h"] - 4))
        print(f"    click [center] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")
        await page.mouse.move(x, y)
        await asyncio.sleep(0.1)
        await page.mouse.click(x, y)
        await asyncio.sleep(random.uniform(0.3, 0.6))
        return

    await _reset_minimap(page)
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

    m = info.get("m", "none") if info else "none"
    if m in ("leaflet", "tiles") and info.get("x") is not None:
        x, y = info["x"], info["y"]
    elif info and info.get("rect"):
        r = info["rect"]
        x = r["left"] + (lng + 180) / 360 * r["w"]
        lat_rad = math.radians(lat)
        y = r["top"] + (0.5 - math.log(
            math.tan(math.pi / 4 + lat_rad / 2)) / (2 * math.pi)) * r["h"]
        m = "mercator"
    else:
        print("    WARNING: fallback centre click")
        x, y = 960.0, 540.0

    if info and info.get("rect"):
        r = info["rect"]
        x = max(r["left"] + 4, min(x, r["left"] + r["w"] - 4))
        y = max(r["top"] + 4, min(y, r["top"] + r["h"] - 4))

    print(f"    click [{m}] ({lat:.4f},{lng:.4f}) -> ({x:.0f},{y:.0f})")
    await page.mouse.move(x, y)
    await asyncio.sleep(0.1)
    await page.mouse.click(x, y)
    await asyncio.sleep(random.uniform(0.3, 0.6))


async def _submit(page):
    for sel in [
        "#confirm-button", "[class*='confirm-button']",
        "button:has-text('Guess')", "button:has-text('Submit')",
        "div:has-text('Guess')",
    ]:
        try:
            btn = await page.wait_for_selector(sel, timeout=2000)
            if btn:
                await btn.click()
                await asyncio.sleep(random.uniform(0.3, 0.6))
                return
        except Exception:
            continue
    try:
        await page.keyboard.press("Enter")
    except Exception:
        pass


async def _next_round(page):
    for text in ["Play Again", "Next Round", "Next", "Continue", "Return"]:
        try:
            btn = await page.wait_for_selector(
                f"button:has-text('{text}')", timeout=2000)
            if btn:
                await btn.click()
                await asyncio.sleep(random.uniform(0.3, 0.6))
                return
        except Exception:
            continue
    await page.goto(GAME_URL, wait_until="networkidle", timeout=30000)


# ---------------------------------------------------------------------------
# Round execution + logging
# ---------------------------------------------------------------------------

async def _play_round(page, ai, mode, round_num, log_entries, run_id,
                      save_images, baselines=None):
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

    if save_images:
        round_dir = ROUNDS_DIR / f"round{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
    else:
        round_dir = None

    true_loc = await _read_true_location(page)

    if mode == "perfect":
        if not true_loc or true_loc == (0.0, 0.0):
            print("  FAILED: no coords available")
            return
        lat, lng = true_loc
        result["true_lat"], result["true_lng"] = lat, lng
        result["guess_lat"], result["guess_lng"] = lat, lng
        if ai and ai.country_coords:
            result["true_country"] = find_country_for_coords(
                lat, lng, ai._all_coords)
            print(f"  true country (est): {result['true_country']}")
        print(f"  true : {lat:.4f}, {lng:.4f}")
    else:
        screenshot_bytes = await _screenshot_panorama(page)
        if round_dir:
            (round_dir / "streetview.png").write_bytes(screenshot_bytes)
        img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")

        main_model_label = "Penguin"
        preds = ai.predict(img, top_k=5)
        pred_countries = [c for c, _, _ in preds]
        result["predictions"] = [
            {"country": c, "score": s, "lat": cl, "lng": cg}
            for c, s, (cl, cg) in preds
        ]
        print(f"  --- {main_model_label} ---")
        for j, p in enumerate(result["predictions"], 1):
            mark = " <<<" if j == 1 else ""
            print(f"  {j:4d} {p['country']:35s} {p['score']:.4f}  "
                  f"({p['lat']:.4f}, {p['lng']:.4f}){mark}")

        skip = {"Unknown", "San Marino", "Holy See (Vatican City State)"}
        chosen = [p for p in preds if p[0] not in skip]
        if not chosen:
            chosen = preds[:1]
        lat, lng = chosen[0][2]
        if lat == 0.0 and lng == 0.0:
            lat, lng = ai.random_coords()
        result["guess_lat"], result["guess_lng"] = lat, lng
        result["guess_country"] = chosen[0][0]

        if true_loc:
            result["true_lat"], result["true_lng"] = true_loc
            dist = haversine_km(lat, lng, true_loc[0], true_loc[1])
            result["distance_km"] = round(dist, 1)
            result["score"] = distance_score(dist)
            print(f"  true : {true_loc[0]:.4f}, {true_loc[1]:.4f}  "
                  f"|  {format_distance(dist)}  |  {result['score']:,} pts")

        if true_loc and ai.country_coords:
            result["true_country"] = find_country_for_coords(
                true_loc[0], true_loc[1], ai._all_coords)
            print(f"  true country (est): {result['true_country']}")

        if baselines and img:
            for name, bl_model in baselines.items():
                bl_preds = bl_model.predict(img, top_k=5)
                bl_countries = [c for c, _, _ in bl_preds]
                result["baseline_predictions"][name] = [
                    {"country": c, "score": s, "lat": cl, "lng": cg}
                    for c, s, (cl, cg) in bl_preds
                ]
                print(f"  --- {name} ---")
                for j, p in enumerate(result["baseline_predictions"][name], 1):
                    mark = " <<<" if j == 1 else ""
                    print(f"  {j:4d} {p['country']:35s} {p['score']:.4f}  "
                          f"({p['lat']:.4f}, {p['lng']:.4f}){mark}")

        if result.get("true_country"):
            tc = result["true_country"]
            chosen_countries = [c for c, _, _ in chosen]
            pred_countries = [c for c, _, _ in preds]
            top1_ok = chosen_countries[0] == tc if chosen_countries else False
            top5_ok = tc in pred_countries
            result["penguin_top1"] = top1_ok
            result["penguin_top5"] = top5_ok
            print(f"  {main_model_label} top-1: {'OK' if top1_ok else 'miss'}  "
                  f"top-5: {'OK' if top5_ok else 'miss'}")
            for name in result.get("baseline_predictions", {}):
                bl_countries = [p["country"]
                                for p in result["baseline_predictions"][name]]
                b_top1, b_top5 = _compute_accuracy(bl_countries, tc)
                result.setdefault("baseline_top1", {})
                result.setdefault("baseline_top5", {})
                result["baseline_top1"][name] = b_top1
                result["baseline_top5"][name] = b_top5
                print(f"  {name} top-1: {'OK' if b_top1 else 'miss'}  "
                      f"top-5: {'OK' if b_top5 else 'miss'}")

    await _hover_minimap(page)
    await _reset_minimap(page)
    await _click_on_map(page, lat, lng)
    await asyncio.sleep(random.uniform(0.3, 0.6))
    await _submit(page)
    await asyncio.sleep(random.uniform(0.5, 1.0))

    if round_dir:
        await page.screenshot(path=str(round_dir / "result.png"))

    log_entries.append(result)


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def _run_session(args):
    from playwright.async_api import async_playwright

    ai = PenguinAI(device=args.device)
    ai.load()

    baselines = {}
    if args.benchmark:
        sc = StreetCLIPBaseline(device=args.device)
        sc.load()
        baselines["StreetCLIP"] = sc

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=args.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
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
        page = await ctx.new_page()
        await page.add_init_script(ANTI_DETECT_SCRIPT)
        await page.add_init_script(LEAFLET_HOOK_SCRIPT)

        print(f"\n{'=' * 60}")
        print(f"  OpenGuessr Player")
        print(f"  Mode  : {args.mode.upper()}{' + BENCHMARK' if args.benchmark else ''}")
        print(f"  Rounds: {args.rounds}")
        print(f"  Device: {ai.device}")
        if baselines:
            print(f"  Baselines: {', '.join(baselines.keys())}")
        print(f"{'=' * 60}")
        print(f"\nLoading {GAME_URL} ...")
        await page.goto(GAME_URL, wait_until="networkidle", timeout=30000)
        await _dismiss_cookies(page)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entries = []

        for r in range(1, args.rounds + 1):
            print(f"\n--- Round {r}/{args.rounds} ---")
            try:
                await _play_round(page, ai, args.mode, r, log_entries, run_id,
                                  not args.no_images,
                                  baselines=baselines if baselines else None)
            except Exception as e:
                print(f"  ERROR: {e}")
            if r < args.rounds:
                await _next_round(page)
                await asyncio.sleep(random.uniform(0.5, 1.5))

        await browser.close()

    _write_log(log_entries, run_id)


def _write_log(entries, run_id):
    if not entries:
        return
    total = sum(e.get("score", 0) or 0 for e in entries)
    distances = [e["distance_km"] for e in entries if e.get("distance_km")]

    penguin_top1 = [e["penguin_top1"] for e in entries
                    if "penguin_top1" in e]
    penguin_top5 = [e["penguin_top5"] for e in entries
                    if "penguin_top5" in e]
    penguin_top1_acc = sum(penguin_top1) / len(penguin_top1) * 100 \
        if penguin_top1 else None
    penguin_top5_acc = sum(penguin_top5) / len(penguin_top5) * 100 \
        if penguin_top5 else None

    baseline_acc = {}
    for e in entries:
        if "baseline_top1" not in e:
            continue
        for name in e["baseline_top1"]:
            if name not in baseline_acc:
                baseline_acc[name] = {"top1": [], "top5": []}
            baseline_acc[name]["top1"].append(e["baseline_top1"][name])
            baseline_acc[name]["top5"].append(e["baseline_top5"][name])

    summary = {
        "run_id": run_id,
        "rounds": len(entries),
        "total_score": total,
        "avg_distance_km": round(sum(distances) / len(distances), 1)
        if distances else None,
        "median_distance_km": round(sorted(distances)[len(distances) // 2], 1)
        if distances else None,
    }
    if penguin_top1_acc is not None:
        summary["penguin_top1_pct"] = round(penguin_top1_acc, 1)
        summary["penguin_top5_pct"] = round(penguin_top5_acc, 1)
    for name, accs in baseline_acc.items():
        if accs["top1"]:
            summary[f"{name}_top1_pct"] = round(
                sum(accs["top1"]) / len(accs["top1"]) * 100, 1)
            summary[f"{name}_top5_pct"] = round(
                sum(accs["top5"]) / len(accs["top5"]) * 100, 1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNS_DIR / f"run_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump({"summary": summary, "rounds": entries}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Session complete")
    print(f"  Rounds       : {summary['rounds']}")
    print(f"  Total score  : {summary['total_score']:,}")
    if summary["avg_distance_km"] is not None:
        print(f"  Avg distance : {format_distance(summary['avg_distance_km'])}")
        print(f"  Med distance : {format_distance(summary['median_distance_km'])}")
    if penguin_top1_acc is not None:
        print(f"  Penguin top-1: {penguin_top1_acc:.1f}%  "
              f"top-5: {penguin_top5_acc:.1f}%")
    for name, accs in baseline_acc.items():
        if accs["top1"]:
            t1 = sum(accs["top1"]) / len(accs["top1"]) * 100
            t5 = sum(accs["top5"]) / len(accs["top5"]) * 100
            print(f"  {name} top-1: {t1:.1f}%  top-5: {t5:.1f}%")
    print(f"  Log saved to : {log_path}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenGuessr AI Player")
    parser.add_argument("--mode", choices=["ai", "perfect"], default="ai",
                        help="ai=model prediction | perfect=iframe coords")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser headless")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving round screenshots")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark Penguin vs raw StreetCLIP baseline")
    args = parser.parse_args()

    asyncio.run(_run_session(args))


if __name__ == "__main__":
    main()
