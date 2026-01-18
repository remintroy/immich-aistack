import os
import hashlib
from PIL import Image
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------

IMAGE_DIR = "images02"
SIMILARITY_THRESHOLD = 0.90
MODEL_NAME = "clip-ViT-B-32"
BATCH_SIZE = 8
MAX_IMAGE_SIZE = 512

# --------------------------------------

torch.set_num_threads(1)

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# ---------- UTIL ----------

def sha256_of_file(path, chunk_size=8192):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def resize_image(img, max_size):
    img.thumbnail((max_size, max_size))
    return img


# ---------- LOAD FILES ----------

def load_records(image_dir):
    records = []

    for file in sorted(os.listdir(image_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        path = os.path.join(image_dir, file)

        try:
            file_hash = sha256_of_file(path)
            img = Image.open(path).convert("RGB")
            img = resize_image(img, MAX_IMAGE_SIZE)

            records.append({
                "filename": file,
                "path": path,
                "hash": file_hash,
                "image": img,
            })

        except Exception as e:
            print(f"Skipping {file}: {e}")

    return records


# ---------- HASH GROUPING ----------

def build_hash_groups(records):
    hash_map = {}
    for r in records:
        hash_map.setdefault(r["hash"], []).append(r)

    return list(hash_map.values())


# ---------- CLIP EMBEDDINGS ----------

def compute_embeddings(records):
    images = [r["image"] for r in records]
    all_embeddings = []

    print("Computing CLIP embeddings...")

    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]

        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        all_embeddings.append(emb)
        del batch

    return np.vstack(all_embeddings)


# ---------- CLIP GROUPING (TRANSITIVE) ----------

def group_clip(records, embeddings, threshold):
    similarity = cosine_similarity(embeddings)

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
            group.append(cur)

            for j in range(len(records)):
                if j not in visited and similarity[cur, j] >= threshold:
                    stack.append(j)

        groups.append(group)

    return groups


# ---------- FINAL COMBINE ----------

def combine_hash_and_clip(hash_groups):
    """
    For each hash group:
    - pick one representative
    - run CLIP on representatives
    - expand CLIP groups with full hash groups
    """

    # Pick representatives
    reps = [group[0] for group in hash_groups]

    embeddings = compute_embeddings(reps)
    clip_groups_idx = group_clip(reps, embeddings, SIMILARITY_THRESHOLD)

    final_groups = []

    for group_idx in clip_groups_idx:
        merged = []
        for idx in group_idx:
            merged.extend(hash_groups[idx])
        final_groups.append(merged)

    return final_groups


# ---------- MAIN ----------

def main():
    records = load_records(IMAGE_DIR)
    print(f"Loaded {len(records)} images")

    hash_groups = build_hash_groups(records)
    print(f"Hash groups: {len(hash_groups)}")

    final_groups = combine_hash_and_clip(hash_groups)

    print("\n========== FINAL GROUPS ==========\n")

    for i, group in enumerate(final_groups, 1):
        if len(group) < 2:
            continue

        print(f"Group {i} ({len(group)} images):")
        for r in group:
            print(f"  - {r['filename']}")
        print()

    print("Done 🎉")


if __name__ == "__main__":
    main()
