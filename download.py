"""
download_samples.py

Downloads 10 sample PDFs from pdf_links.jsonl to a local folder.
Run this on your Mac before uploading to Colab.

Usage:
    python download_samples.py

Output:
    sample_pdfs/          <- folder with 10 PDFs
    sample_pdfs/links.jsonl  <- just the 10 links used (for Colab)
"""

import json
import os
import random
import requests
from tqdm import tqdm

# ===================== CONFIG =====================
LINKS_FILE   = "pdf_links.jsonl"
OUTPUT_DIR   = "sample_pdfs"
SAMPLE_SIZE  = 10
RANDOM_SEED  = 42
TIMEOUT      = 30
# ==================================================

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all links
with open(LINKS_FILE) as f:
    all_links = [json.loads(l) for l in f if l.strip()]

sample = random.sample(all_links, min(SAMPLE_SIZE, len(all_links)))
print(f"Downloading {len(sample)} PDFs to {OUTPUT_DIR}/\n")

downloaded = []

for item in tqdm(sample):
    url = item["pdf_url"]
    filename = url.split("/")[-1]
    # Clean filename (spaces etc)
    filename = filename.replace(" ", "_")
    dest = os.path.join(OUTPUT_DIR, filename)

    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  OK  {filename}  ({size_kb:.0f} KB)")
        downloaded.append({"pdf_url": url, "local_file": filename})
    except Exception as e:
        print(f"  FAIL {filename}: {e}")

# Save the links used so Colab knows which URL maps to which file
with open(os.path.join(OUTPUT_DIR, "links.jsonl"), "w") as f:
    for d in downloaded:
        f.write(json.dumps(d) + "\n")

print(f"\nDone. {len(downloaded)}/{len(sample)} PDFs saved to {OUTPUT_DIR}/")
print(f"Links saved to {OUTPUT_DIR}/links.jsonl")
print(f"\nNext steps:")
print(f"  1. Zip the folder:  zip -r sample_pdfs.zip sample_pdfs/")
print(f"  2. Zip your model:  zip -r exam_rag_model.zip exam_rag_model/")
print(f"  3. Upload both zips to Google Colab")