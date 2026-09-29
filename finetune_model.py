"""
prepare_finetune_data.py

Reconstructs fine-tuning data by:
1. Re-downloading PDFs from pdf_links.jsonl
2. Extracting full text using pdfplumber
3. Pairing with existing outputs in dspace_questions_metadata.jsonl
4. Producing train/val splits in MLX-LM chat format

Output files:
  - data/train.jsonl
  - data/valid.jsonl       (the name mlx_lm.lora expects)
  - data/prep_report.json   (stats + failed URLs)

Usage:
  python prepare_finetune_data.py
"""

import json
import os
import io
import re
import random
import requests
import pdfplumber
from tqdm import tqdm

# ===================== CONFIG =====================
LINKS_FILE = "pdf_links.jsonl"
METADATA_FILE = "dspace_questions_metadata.jsonl"
OUTPUT_DIR = "data"
MAX_TEXT_CHARS = 6000       # truncate very long PDFs (context limit)
FIRST_PAGE_MAX = 1500       # for metadata task, only first page text
VAL_SPLIT = 0.1
RANDOM_SEED = 42
REQUEST_TIMEOUT = 30
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ---------- System prompts ----------
METADATA_SYSTEM = """You are an academic document parser. Given text from the first page of an exam paper, extract the course metadata as a JSON object.
Output ONLY valid JSON with these exact keys: course_code, course_name, year, semester.
- year must be an integer (e.g. 2023) or null
- semester must be an integer (e.g. 1 or 2) or null  
- course_code is a string like "19CSE101" or null if not found
- course_name is the full subject name as a string"""

QUESTIONS_SYSTEM = """You are an exam question extractor. Given the full text of an exam paper, extract every question and return them as a JSON object.
Output ONLY valid JSON with a single key "questions" containing an array of objects.
Each object must have:
- "question_text": the full question as a clean string, no numbering or part labels
- "marks": integer mark value or null if not specified
Rules:
- Do NOT include headings, instructions, or administrative text as questions
- Attach sub-headings/context to the question text where needed
- Skip incomplete or unreadable questions"""

# ---------- Load existing metadata outputs ----------
print("Loading existing metadata outputs...")
metadata_by_url = {}
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            url = obj.get("file_link", "")
            if url:
                metadata_by_url[url] = obj
        except json.JSONDecodeError:
            continue

print(f"  Loaded {len(metadata_by_url)} existing records")

# ---------- Load PDF links ----------
with open(LINKS_FILE, "r", encoding="utf-8") as f:
    all_pdfs = [json.loads(line) for line in f if line.strip()]

# Only process PDFs we already have outputs for
pdfs_to_process = [p for p in all_pdfs if p["pdf_url"] in metadata_by_url]
print(f"  {len(pdfs_to_process)} PDFs have matching metadata outputs\n")

# ---------- Extract text from PDF bytes ----------
def extract_text_from_bytes(pdf_bytes):
    """Returns (first_page_text, full_text)"""
    first_page = ""
    all_text_parts = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()

            # Also pull table content as plain text
            for table in page.extract_tables():
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row if cell)
                    if row_text.strip():
                        text += "\n" + row_text

            if i == 0:
                first_page = text[:FIRST_PAGE_MAX]

            all_text_parts.append(text)

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text_parts)
    return first_page, full_text[:MAX_TEXT_CHARS]


# ---------- Build training examples ----------
def make_metadata_example(first_page_text, obj):
    """Chat-format example for metadata extraction task."""
    target = {
        "course_code": obj.get("course_code"),
        "course_name": obj.get("course_name"),
        "year": obj.get("year"),
        "semester": obj.get("semester")
    }
    # Skip if we have no useful metadata at all
    if not any(target.values()):
        return None

    return {
        "messages": [
            {"role": "system", "content": METADATA_SYSTEM},
            {"role": "user", "content": first_page_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ]
    }


def make_questions_example(full_text, obj):
    """Chat-format example for question extraction task."""
    questions = obj.get("questions", [])
    # Need at least 2 questions to be useful training data
    if len(questions) < 2:
        return None

    # Clean questions: only keep question_text and marks
    clean_qs = []
    for q in questions:
        qt = q.get("question_text", "").strip()
        if qt:
            clean_qs.append({
                "question_text": qt,
                "marks": q.get("marks")
            })

    if len(clean_qs) < 2:
        return None

    target = {"questions": clean_qs}
    return {
        "messages": [
            {"role": "system", "content": QUESTIONS_SYSTEM},
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ]
    }


# ---------- Main loop ----------
session = requests.Session()
all_examples = []
failed_urls = []
skipped_no_text = 0

print("Re-downloading and extracting PDF text...\n")
for pdf in tqdm(pdfs_to_process):
    url = pdf["pdf_url"]
    obj = metadata_by_url[url]

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        first_page, full_text = extract_text_from_bytes(resp.content)

        if not full_text.strip():
            skipped_no_text += 1
            continue

        # Metadata example (from first page)
        meta_ex = make_metadata_example(first_page, obj)
        if meta_ex:
            all_examples.append(("metadata", meta_ex))

        # Questions example (from full text)
        q_ex = make_questions_example(full_text, obj)
        if q_ex:
            all_examples.append(("questions", q_ex))

    except Exception as e:
        failed_urls.append({"url": url, "error": str(e)})
        continue


print(f"\n  Built {len(all_examples)} total examples")
print(f"  Failed downloads: {len(failed_urls)}")
print(f"  Skipped (no text extracted): {skipped_no_text}")

# ---------- Deduplicate ----------
seen = set()
unique_examples = []
for task, ex in all_examples:
    # Include the task: the metadata and questions examples of one paper start
    # with the same text, so keying on the input prefix alone dropped every
    # questions example.
    key = (task, ex["messages"][1]["content"])
    if key not in seen:
        seen.add(key)
        unique_examples.append(ex)

print(f"  After dedup: {len(unique_examples)} unique examples")

# ---------- Train / val split ----------
random.shuffle(unique_examples)
val_size = max(1, int(len(unique_examples) * VAL_SPLIT))
val_set = unique_examples[:val_size]
train_set = unique_examples[val_size:]

print(f"\n  Train: {len(train_set)} | Val: {len(val_set)}")

# ---------- Write output ----------
def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_set)
write_jsonl(os.path.join(OUTPUT_DIR, "valid.jsonl"), val_set)

# ---------- Report ----------
report = {
    "total_examples": len(unique_examples),
    "train": len(train_set),
    "val": len(val_set),
    "failed_downloads": len(failed_urls),
    "skipped_no_text": skipped_no_text,
    "failed_urls": failed_urls[:50]   # cap to keep report small
}
with open(os.path.join(OUTPUT_DIR, "prep_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\n[DONE] Files written to {OUTPUT_DIR}/")
print(f"  train.jsonl  → {len(train_set)} examples")
print(f"  valid.jsonl  → {len(val_set)} examples")
print(f"  prep_report.json")
print("\nNext step: run finetune.sh")