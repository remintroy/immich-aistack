# Immich Auto Stacker 📸

> **Automatically group similar images, bursts, and duplicates into Stacks in Immich.**

This tool uses advanced AI (CLIP models) and perceptual hashing to identify semantically similar images and near-duplicates, automatically organizing them into tidy "Stacks" within your Immich library.

## ✨ Features

- **Semantic Matching**: Uses `sentence-transformers/clip-ViT-L-14` to understand the _content_ of your images.
- **Burst Detection**: Uses **Perceptual Hashing (pHash)** to find near-identical images (like burst mode shots).
- **Efficient Clustering**: Powered by Facebook's **FAISS** for high-speed similarity search.
- **Date-Aware**: Processes images day-by-day to ensure relevant grouping.
- **Configurable**: Adjustable thresholds for strictness and semantic similarity.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- An [Immich](https://immich.app/) instance
- An Immich API Key

### Installation

1. **Clone the repository** (or download the files):

   ```bash
   git clone <your-repo-url>
   cd image-stacker
   ```

2. **Set up a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Environment Variables**:
   Create a `.env` file in the root directory:

   ```env
   IMMICH_URL=http://your-immich-ip:2283
   IMMICH_API_KEY=your-api-key-here
   ```

2. **Script Settings** (Optional):
   Open `stack.py` to adjust processing parameters:
   - `START_DATE` / `END_DATE`: The range of dates to process.
   - `DRY_RUN`: Set to `True` to test without making changes.
   - `HIGH_CLIP` / `LOW_CLIP`: Adjust semantic similarity sensitivity.

## 🛠 Usage

Run the script directly:

```bash
python stack.py
```

The script will:

1. Load the AI model (approx 1-2GB download on first run).
2. Fetch image metadata for the specified date range.
3. Download thumbnails to compute embeddings and hashes.
4. Cluster images and send "Stack" commands to your Immich server.

## ⚙️ How It Works

1. **Fetch**: Retrieves all images for a specific day.
2. **Embed**: Generates a CLIP embedding (vector) and a perceptual hash (pHash) for each image.
3. **Cluster**:
   - Assets are grouped if they share a high **Semantic Score** (look similar conceptually).
   - Assets are grouped if they have a low **Hamming Distance** (look nearly identical visually).
4. **Stack**: Calls the Immich API to stack the grouped assets, picking the first one as the primary asset.

## ⚠️ Notes

- **Performance**: The first run will download models. GPU acceleration (`cuda`) is used if available; otherwise, it runs on CPU.
- **Safety**: Use `DRY_RUN = True` in `stack.py` first to see what _would_ happen without modifying your library.
- **Backup**: While this only "stacks" images, it's always good practice to backup your Immich database before running bulk automated tools.
