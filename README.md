# Immich AI Stacker

**Automatically group similar images and bursts into Stacks using AI.**

This tool leverages CLIP embeddings and perceptual hashing to semantically organize your Immich library, tidying up bursts and near-duplicates with high precision.

🔗 **[github.com/remintroy/immich-aistack](https://github.com/remintroy/immich-aistack)**

## Features

- **Semantic Grouping**: Uses `CLIP ViT-L-14` to stack images with matching content.
- **Burst Detection**: Uses Perceptual Hashing (pHash) to catch near-identical shots.
- **High Performance**: Optimized with FAISS for fast clustering.
- **Safe**: Configurable dry-run mode to test before applying changes.

## Quick Start

### 1. Install

```bash
git clone https://github.com/remintroy/immich-aistack.git
cd immich-aistack

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root:

```env
IMMICH_URL=https://your-immich-domain.com
IMMICH_API_KEY=your_api_key_here
DRY_RUN=false
```

> **API Permissions Required:**
> Ensure your API Key includes:
>
> - `asset.read`, `asset.view`, `asset.download`, `asset.statistics`
> - All `Stack` permissions (create, delete, etc.)

### 3. Run

```bash
python stack.py
```

## How It Works

1. **Scan**: Fetches images day-by-day from your library.
2. **Analyze**: Generates vector embeddings and visual hashes for each photo.
3. **Cluster**: Groups images that are visually or semantically similar.
4. **Stack**: Sends commands to Immich to stack the grouped assets.

---

_Tip: Set `DRY_RUN=true` in your `.env` to preview stacks without modifying your library._
