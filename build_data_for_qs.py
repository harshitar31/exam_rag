"""
build_questions_training_data.py

Builds question-splitting training data by:
1. Reading existing dspace_questions_metadata.jsonl (has questions + file_link)
2. Re-downloading each PDF to get raw text (the model input)
3. Pairing raw text → existing questions (the model output)
4. Combining with existing metadata train/val data
5. Writing new data_v2/train.jsonl and data_v2/val.jsonl for retraining

Usage:
    python build_data_for_qs.py

Output:
    data_v2/train.jsonl
    data_v2/valid.jsonl     (the name mlx_lm.lora expects)
    data_v2/report.json

To repair an already-built data_v2/ without re-downloading PDFs, use
fix_training_data.py instead.
"""

import hashlib
import json
import os
import random
import requests
from tqdm import tqdm

from cleaning import normalize_course_code, normalize_semester, normalize_year
from prep_utils import extract_pdf_text, grounded_questions, split_parts
from prompts import METADATA_SYSTEM, QUESTIONS_SYSTEM

# ===================== CONFIG =====================
METADATA_FILE     = "dspace_questions_metadata.jsonl"
EXISTING_TRAIN    = "data/train.jsonl"
EXISTING_VAL      = "data/valid.jsonl"
OUTPUT_DIR        = "data_v2"
MAX_PDFS          = None        # None = process all, set e.g. 300 to limit
MAX_TEXT_CHARS    = 8000
FIRST_PAGE_MAX    = 1500
VAL_SPLIT         = 0.1
RANDOM_SEED       = 42
REQUEST_TIMEOUT   = 30
MIN_QUESTIONS     = 3           # skip papers with fewer questions than this
# ==================================================

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- PDF helpers --------------------

def download_and_extract(url):
    """Returns (first_page_text, full_text)"""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return extract_pdf_text(resp.content, FIRST_PAGE_MAX, MAX_TEXT_CHARS)


# -------------------- Training example builders --------------------

def make_questions_examples(full_text, questions):
    """
    Splits the paper into Part A / Part B and creates one training example per
    section. Each target holds exactly the questions whose text appears in that
    section (in reading order) -- never questions cut off by MAX_TEXT_CHARS or
    belonging to the other section, which would teach the model to hallucinate.
    """
    if not questions or len(questions) < MIN_QUESTIONS:
        return []

    part_a_text, part_b_text = split_parts(full_text)
    sections = [part_a_text, part_b_text] if part_b_text else [full_text]

    examples = []
    for section_text in sections:
        target_qs = grounded_questions(section_text, questions)
        if len(target_qs) < 2:
            continue
        examples.append({
            "messages": [
                {"role": "system",    "content": QUESTIONS_SYSTEM},
                {"role": "user",      "content": section_text},
                {"role": "assistant", "content": json.dumps({"questions": target_qs}, ensure_ascii=False)}
            ]
        })

    return examples


def normalize_metadata_target(target):
    """Make metadata targets match the prompt: integer year/semester, clean course code."""
    return {
        "course_code": normalize_course_code(target.get("course_code")),
        "course_name": (target.get("course_name") or "").strip() or None,
        "year":        normalize_year(target.get("year")),
        "semester":    normalize_semester(target.get("semester")),
    }


def make_metadata_example(first_page, obj):
    """Chat-format example for metadata extraction."""
    target = normalize_metadata_target(obj)
    if not any(target.values()):
        return None

    return {
        "messages": [
            {"role": "system",    "content": METADATA_SYSTEM},
            {"role": "user",      "content": first_page},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
        ]
    }


# -------------------- Load existing data --------------------

print("Loading existing metadata training data...")
existing_examples = []
for path in [EXISTING_TRAIN, EXISTING_VAL]:
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    # Old examples were built before semester normalisation
                    if ex["messages"][0]["content"].startswith("You are an academic document parser"):
                        target = normalize_metadata_target(json.loads(ex["messages"][2]["content"]))
                        ex["messages"][0]["content"] = METADATA_SYSTEM
                        ex["messages"][2]["content"] = json.dumps(target, ensure_ascii=False)
                    existing_examples.append(ex)

print(f"  Loaded {len(existing_examples)} existing examples")

# -------------------- Load metadata JSONL --------------------

print(f"\nLoading {METADATA_FILE}...")
records = []
with open(METADATA_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("file_link") and obj.get("questions"):
                records.append(obj)
        except json.JSONDecodeError:
            continue

print(f"  Found {len(records)} records with questions")

if MAX_PDFS:
    records = records[:MAX_PDFS]
    print(f"  Limiting to {MAX_PDFS} PDFs")

# -------------------- Main loop --------------------

new_q_examples  = []
new_meta_examples = []
failed          = []
skipped_few_qs  = 0
skipped_no_text = 0

print(f"\nProcessing {len(records)} PDFs...\n")

session = requests.Session()

for obj in tqdm(records):
    url = obj["file_link"]
    questions = obj.get("questions", [])

    if len(questions) < MIN_QUESTIONS:
        skipped_few_qs += 1
        continue

    try:
        first_page, full_text = download_and_extract(url)

        if not full_text.strip():
            skipped_no_text += 1
            continue

        # Question splitting examples
        q_examples = make_questions_examples(full_text, questions)
        new_q_examples.extend(q_examples)

        # Metadata example (from this PDF — may differ from existing data)
        meta_ex = make_metadata_example(first_page, obj)
        if meta_ex:
            new_meta_examples.append(meta_ex)

    except Exception as e:
        failed.append({"url": url, "error": str(e)})
        continue

print(f"\nResults:")
print(f"  New question examples   : {len(new_q_examples)}")
print(f"  New metadata examples   : {len(new_meta_examples)}")
print(f"  Failed downloads        : {len(failed)}")
print(f"  Skipped (few questions) : {skipped_few_qs}")
print(f"  Skipped (no text)       : {skipped_no_text}")

# -------------------- Deduplicate --------------------

def dedup(examples):
    # Key on task (system prompt) + full input. Keying on the first 200 chars
    # of the input alone made the metadata and questions examples of the same
    # paper collide, silently dropping every questions example.
    seen = set()
    out = []
    for ex in examples:
        key = hashlib.md5((ex["messages"][0]["content"] + ex["messages"][1]["content"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out

new_q_examples   = dedup(new_q_examples)
new_meta_examples = dedup(new_meta_examples)

# -------------------- Combine all examples --------------------

# Existing examples are mostly metadata — combine with new question examples
# and deduplicated new metadata examples
all_examples = existing_examples + new_q_examples + new_meta_examples
all_examples = dedup(all_examples)

print(f"\nTotal unique examples     : {len(all_examples)}")
print(f"  (existing: {len(existing_examples)}, "
      f"new questions: {len(new_q_examples)}, "
      f"new metadata: {len(new_meta_examples)})")

# -------------------- Split --------------------

random.shuffle(all_examples)
val_size  = max(1, int(len(all_examples) * VAL_SPLIT))
val_set   = all_examples[:val_size]
train_set = all_examples[val_size:]

# -------------------- Write --------------------

def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_set)
write_jsonl(os.path.join(OUTPUT_DIR, "valid.jsonl"), val_set)

report = {
    "total_examples": len(all_examples),
    "train": len(train_set),
    "val": len(val_set),
    "new_question_examples": len(new_q_examples),
    "new_metadata_examples": len(new_meta_examples),
    "existing_examples_reused": len(existing_examples),
    "failed_downloads": len(failed),
    "skipped_few_questions": skipped_few_qs,
    "skipped_no_text": skipped_no_text,
    "failed_urls": failed[:50]
}
with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"\nTrain : {len(train_set)} examples → {OUTPUT_DIR}/train.jsonl")
print(f"Val   : {len(val_set)} examples → {OUTPUT_DIR}/valid.jsonl")
print(f"Report: {OUTPUT_DIR}/report.json")
print(f"\nNext: bash finetune.sh (or upload {OUTPUT_DIR}/ to Colab and retrain with --data {OUTPUT_DIR}/)")