import os
import hashlib
from PIL import Image
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

IMAGE_DIR = "images"
SIMILARITY_THRESHOLD = 0.95
MODEL_NAME = "clip-ViT-B-32"
BATCH_SIZE = 8
MAX_IMAGE_SIZE = 512
TOP_K = 10            # number of neighbors to search

# --------------------------------------

torch.set_num_threads(1)

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# ---------- UTIL ----------

def sha256_of_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def resize_image(img):
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
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
            img = resize_image(img)

            records.append({
                "filename": file,
                "hash": file_hash,
                "image": img
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

    return np.vstack(all_embeddings).astype("float32")


# ---------- FAISS GROUPING ----------

def group_with_faiss(records, embeddings, threshold, top_k):
    dim = embeddings.shape[1]

    # FAISS cosine similarity = inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
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
            group.append(cur)

            scores, neighbors = index.search(
                embeddings[cur:cur + 1], top_k
            )

            for score, j in zip(scores[0], neighbors[0]):
                if j == cur:
                    continue
                if j not in visited and score >= threshold:
                    stack.append(j)

        groups.append(group)

    return groups


# ---------- FINAL COMBINE ----------

def combine_hash_and_clip(hash_groups):
    reps = [group[0] for group in hash_groups]

    embeddings = compute_embeddings(reps)

    clip_groups_idx = group_with_faiss(
        reps, embeddings, SIMILARITY_THRESHOLD, TOP_K
    )

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
