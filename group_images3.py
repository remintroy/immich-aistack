import os
import hashlib
import math
import time
import logging
from datetime import datetime
from PIL import Image
import piexif
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
import shutil
from pathlib import Path


# ================== LOGGING ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("image-stacker")

# ================== CONFIG ==================

IMAGE_DIR = "images"

OUTPUT_DIR = "output"
COPY_SINGLE_IMAGES = False

SIMILARITY_THRESHOLD = 0.90
TOP_K = 10

MODEL_NAME = "clip-ViT-B-32"
BATCH_SIZE = 8
MAX_IMAGE_SIZE = 512

TIME_WINDOW_SECONDS = 10
GPS_GRID_SIZE_METERS = 100  # fuzzy GPS

# ============================================

torch.set_num_threads(1)
log.info("Loading CLIP model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# ================== UTIL ==================

def sha256_of_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def resize_image(img):
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


def dms_to_deg(dms):
    return dms[0][0]/dms[0][1] + dms[1][0]/dms[1][1]/60 + dms[2][0]/dms[2][1]/3600


def extract_metadata(path):
    try:
        exif = piexif.load(path)
    except Exception:
        return None, None, None

    dt = None
    if piexif.ExifIFD.DateTimeOriginal in exif["Exif"]:
        raw = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")

    gps = exif.get("GPS", {})
    lat = lon = None
    if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
        lat = dms_to_deg(gps[piexif.GPSIFD.GPSLatitude])
        lon = dms_to_deg(gps[piexif.GPSIFD.GPSLongitude])

    return dt, lat, lon


def gps_bucket(lat, lon):
    if lat is None or lon is None:
        return None
    grid_deg = GPS_GRID_SIZE_METERS / 111_000
    return (
        round(lat / grid_deg),
        round(lon / grid_deg),
    )

# ================== LOAD ==================

def load_records(image_dir):
    log.info("Scanning directory: %s", image_dir)

    records = []
    skipped = 0

    for file in sorted(os.listdir(image_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        path = os.path.join(image_dir, file)

        try:
            img = resize_image(Image.open(path).convert("RGB"))
            dt, lat, lon = extract_metadata(path)

            records.append({
                "filename": file,
                "hash": sha256_of_file(path),
                "datetime": dt,
                "gps": gps_bucket(lat, lon),
                "image": img,
            })

        except Exception as e:
            skipped += 1
            log.warning("Skipping %s (%s)", file, e)

    log.info("Loaded %d images (skipped %d)", len(records), skipped)
    return records

# ================== HASH ==================

def group_by_hash(records):
    log.info("Grouping exact duplicates by SHA-256")

    m = {}
    for r in records:
        m.setdefault(r["hash"], []).append(r)

    groups = list(m.values())

    dup_groups = sum(1 for g in groups if len(g) > 1)
    log.info(
        "Hash groups: %d total | %d duplicate groups",
        len(groups),
        dup_groups,
    )

    return groups

# ================== METADATA ==================

def build_metadata_buckets(hash_groups):
    log.info("Bucketing by time (±%ss) and fuzzy GPS (%sm)",
             TIME_WINDOW_SECONDS, GPS_GRID_SIZE_METERS)

    buckets = {}

    for group in hash_groups:
        rep = group[0]
        dt = rep["datetime"]

        if dt:
            time_key = (
                dt.date(),
                dt.hour,
                dt.minute,
                dt.second // TIME_WINDOW_SECONDS,
            )
        else:
            time_key = None

        key = (time_key, rep["gps"])
        buckets.setdefault(key, []).append(group)

    log.info(
        "Metadata buckets created: %d (avg %.1f images/bucket)",
        len(buckets),
        sum(len(b) for b in buckets.values()) / len(buckets),
    )

    return list(buckets.values())

# ================== CLIP ==================

def compute_embeddings(records):
    log.info("Computing CLIP embeddings for %d images", len(records))
    start = time.time()

    imgs = [r["image"] for r in records]
    embs = []

    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i:i + BATCH_SIZE]
        e = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embs.append(e)

    log.info("Embeddings computed in %.2fs", time.time() - start)
    return np.vstack(embs).astype("float32")

# ================== FAISS ==================

def group_with_faiss(records, embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    visited = set()
    groups = []

    for i in range(len(records)):
        if i in visited:
            continue

        stack = [i]
        group = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue

            visited.add(cur)
            group.append(records[cur])

            scores, idxs = index.search(
                embeddings[cur:cur+1], TOP_K
            )

            for score, j in zip(scores[0], idxs[0]):
                if j != cur and j not in visited and score >= SIMILARITY_THRESHOLD:
                    stack.append(j)

        groups.append(group)

    return groups

def copy_groups_to_output(groups):
    log.info("Copying grouped images to output directory: %s", OUTPUT_DIR)

    out_base = Path(OUTPUT_DIR)
    out_base.mkdir(parents=True, exist_ok=True)

    copied_groups = 0
    copied_files = 0

    for idx, group in enumerate(groups, start=1):
        if len(group) < 2 and not COPY_SINGLE_IMAGES:
            continue

        group_dir = out_base / f"group_{idx:03d}"
        group_dir.mkdir(exist_ok=True)

        for record in group:
            src = Path(IMAGE_DIR) / record["filename"]
            dst = group_dir / record["filename"]

            try:
                shutil.copy2(src, dst)
                copied_files += 1
            except Exception as e:
                log.warning(
                    "Failed to copy %s → %s (%s)",
                    src, dst, e
                )

        copied_groups += 1

    log.info(
        "Copy complete: %d groups, %d files copied",
        copied_groups, copied_files
    )


# ================== MAIN ==================

def main():
    log.info("==== IMAGE GROUPING STARTED ====")

    records = load_records(IMAGE_DIR)
    hash_groups = group_by_hash(records)
    buckets = build_metadata_buckets(hash_groups)

    final_groups = []
    processed = 0

    for idx, bucket in enumerate(buckets, 1):
        reps = [g[0] for g in bucket]

        if len(reps) < 2:
            final_groups.append(bucket[0])
            continue

        log.info(
            "Bucket %d/%d → %d candidate images",
            idx, len(buckets), len(reps)
        )

        embeddings = compute_embeddings(reps)
        clip_groups = group_with_faiss(reps, embeddings)

        for cg in clip_groups:
            merged = []
            for r in cg:
                for hg in bucket:
                    if hg[0]["hash"] == r["hash"]:
                        merged.extend(hg)
            final_groups.append(merged)
            processed += 1

    log.info("==== GROUPING COMPLETE ====")
    log.info("Final groups created: %d", processed)

    print("\n========== FINAL GROUPS ==========\n")
    gid = 1
    for g in final_groups:
        if len(g) < 2:
            continue
        print(f"Group {gid} ({len(g)} images):")
        for r in g:
            print(f"  - {r['filename']}")
        print()
        gid += 1

    copy_groups_to_output(final_groups)


if __name__ == "__main__":
    main()
