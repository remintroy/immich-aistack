import os
import gc
import math
import logging
from datetime import datetime, timedelta
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

import requests
import numpy as np
import torch
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from imagehash import phash

# ===================== CONFIG =====================

IMMICH_URL = os.getenv("IMMICH_URL")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY")

if not IMMICH_URL or not IMMICH_API_KEY:
    raise RuntimeError("IMMICH_URL and IMMICH_API_KEY must be set")

HEADERS = {
    "x-api-key": IMMICH_API_KEY,
    "Content-Type": "application/json",
}

PAGE_SIZE = 40
EMBED_BATCH = 4
MAX_IMAGE_SIZE = 384

# --- Similarity tuning ---
HIGH_CLIP = 0.917          # always accept
LOW_CLIP = 0.88           # accept only with pHash
PHASH_DISTANCE = 4
TOP_K = 5                # increased for better chaining

DRY_RUN = False

# =================================================

torch.set_num_threads(1)

# ===================== LOGGING =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("immich-auto-stacker")

# ===================== MODEL =====================

log.info("Loading CLIP model (CPU)")
model = SentenceTransformer("clip-ViT-B-32")

# ===================== IMMICH API =====================

def get_asset_count(date):
    payload = {
        "takenAfter": f"{date}T00:00:00.000Z",
        "takenBefore": f"{date}T23:59:59.999Z",
        "type": "IMAGE",
    }
    r = requests.post(
        f"{IMMICH_URL}/api/search/statistics",
        json=payload,
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["total"]


def fetch_metadata_page(date, page):
    payload = {
        "takenAfter": f"{date}T00:00:00.000Z",
        "takenBefore": f"{date}T23:59:59.999Z",
        "type": "IMAGE",
        "page": page,
        "size": PAGE_SIZE,
    }
    r = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        json=payload,
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["assets"]["items"]


def load_thumbnail(asset_id):
    url = f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail"
    with requests.get(url, headers=HEADERS, timeout=20) as r:
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        return img


def create_stack(primary_id, asset_ids):
    if DRY_RUN:
        log.info("[DRY-RUN] Stack → %s (%d assets)", primary_id, len(asset_ids))
        return True

    payload = {
        "primaryAssetId": primary_id,
        "assetIds": asset_ids,
    }

    try:
        r = requests.post(
            f"{IMMICH_URL}/api/stacks",
            json=payload,
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        return True

    except requests.HTTPError as e:
        status = e.response.status_code
        if status in (400, 409):
            log.warning("Stack skipped (HTTP %d) — conflict", status)
            return False
        raise

# ===================== GROUPING =====================

def accept_pair(score, phash_a, phash_b):
    if score >= HIGH_CLIP:
        return True
    if score >= LOW_CLIP and abs(phash_a - phash_b) <= PHASH_DISTANCE:
        return True
    return False


def group_with_faiss(asset_ids, embeddings, phashes):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    visited = set()
    groups = []

    for i in range(len(asset_ids)):
        if i in visited:
            continue

        stack = [i]
        group = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue

            visited.add(cur)
            group.append(asset_ids[cur])

            scores, idxs = index.search(
                embeddings[cur:cur + 1], TOP_K
            )

            for score, j in zip(scores[0], idxs[0]):
                if j == cur or j in visited:
                    continue

                if accept_pair(
                    score,
                    phashes[asset_ids[cur]],
                    phashes[asset_ids[j]],
                ):
                    stack.append(j)

        groups.append(group)

    return groups

# ===================== DAILY PIPELINE =====================

def process_day(date):
    log.info("Processing day %s", date)

    total = get_asset_count(date)
    log.info("Total images (statistics): %d", total)

    if total < 2:
        return

    pages = math.ceil(total / PAGE_SIZE)

    all_embeddings = []
    all_asset_ids = []
    asset_lookup = {}
    phash_lookup = {}

    for page in range(1, pages + 1):
        items = fetch_metadata_page(date, page)
        log.info("Page %d/%d → %d assets", page, pages, len(items))

        images = []
        ids = []

        for asset in items:
            asset_lookup[asset["id"]] = asset
            try:
                img = load_thumbnail(asset["id"])
                images.append(img)
                ids.append(asset["id"])
                phash_lookup[asset["id"]] = phash(img)
            except Exception as e:
                log.warning("Thumbnail failed %s (%s)", asset["id"], e)

        for i in range(0, len(images), EMBED_BATCH):
            emb = model.encode(
                images[i:i + EMBED_BATCH],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.append(emb)
            all_asset_ids.extend(ids[i:i + EMBED_BATCH])

        del images
        gc.collect()

    embeddings = np.vstack(all_embeddings).astype("float32")

    log.info("Embedding complete → %d vectors", len(all_asset_ids))

    groups = group_with_faiss(all_asset_ids, embeddings, phash_lookup)

    created = 0
    for group in groups:
        unstacked = [
            aid for aid in group
            if asset_lookup.get(aid, {}).get("duplicateId") is None
        ]

        if len(unstacked) < 2:
            continue

        if create_stack(unstacked[0], unstacked):
            created += 1

    log.info("Stacks created for %s → %d", date, created)

# ===================== RUN RANGE =====================

def run(start_date, end_date):
    cur = start_date
    while cur <= end_date:
        process_day(cur.strftime("%Y-%m-%d"))
        gc.collect()
        cur += timedelta(days=1)

# ===================== ENTRY =====================

if __name__ == "__main__":
    START = "2025-09-27"
    END = "2025-09-27"

    log.info("==== IMMICH AUTO-STACKER START ====")
    log.info("DRY-RUN MODE: %s", DRY_RUN)

    run(
        datetime.fromisoformat(START),
        datetime.fromisoformat(END),
    )

    log.info("==== DONE ====")
