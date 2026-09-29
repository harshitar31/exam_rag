"""
fix_training_data.py

Repairs an already-built training set (data_v2/) without re-downloading any
PDFs -- the PDF text is already inside each example.

1. Questions examples: the target is rebuilt from the paper's full question
   list (dspace_questions_metadata.jsonl), keeping only questions whose text
   appears in the example's input, in reading order. In data_v2, ~47% of
   target questions were not in the input (cut off at 8000 chars or assigned
   to the wrong Part A/B by the marks heuristic).
2. Metadata examples: year/semester become integers ("First" -> 1) and course
   codes are normalised, matching what the system prompt asks for.
3. Deduplicates on task + full input and writes valid.jsonl (not val.jsonl),
   which is the file name mlx_lm.lora looks for.

Usage:
    python fix_training_data.py            # data_v2/ -> data_v3/
"""

import hashlib
import json
import os
import random
import re
from collections import Counter

from prompts import METADATA_SYSTEM, QUESTIONS_SYSTEM
from cleaning import clean_question_text, normalize_course_code, normalize_semester, normalize_year
from prep_utils import grounded_questions

# ===================== CONFIG =====================
INPUT_DIR = "data_v2"
OUTPUT_DIR = "data_v3"
METADATA_FILE = "dspace_questions_metadata.jsonl"
VAL_SPLIT = 0.1
RANDOM_SEED = 42
MIN_QUESTIONS = 2
# ==================================================

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---------- index every paper's questions by a text prefix ----------
papers = []
with open(METADATA_FILE, encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("questions"):
            papers.append(obj["questions"])

prefix_to_papers = {}
for paper_idx, questions in enumerate(papers):
    for q in questions:
        key = norm(clean_question_text(q.get("question_text", ""))[0])[:30]
        if len(key) >= 15:
            prefix_to_papers.setdefault(key, set()).add(paper_idx)


def find_paper(target_questions):
    votes = Counter()
    for q in target_questions:
        key = norm(clean_question_text(q.get("question_text", ""))[0])[:30]
        for paper_idx in prefix_to_papers.get(key, ()):
            votes[paper_idx] += 1
    return votes.most_common(1)[0][0] if votes else None


# ---------- load + repair ----------
examples = []
for name in ("train.jsonl", "valid.jsonl", "val.jsonl"):
    path = os.path.join(INPUT_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            examples.extend(json.loads(line) for line in f if line.strip())

stats = Counter()
fixed = []
for ex in examples:
    system, user, assistant = (m["content"] for m in ex["messages"])

    if system.startswith("You are an exam question"):
        stats["questions_in"] += 1
        old_target = json.loads(assistant).get("questions", [])
        paper_idx = find_paper(old_target)
        source_questions = papers[paper_idx] if paper_idx is not None else old_target
        new_target = grounded_questions(user, source_questions)
        stats["target_questions_before"] += len(old_target)
        stats["target_questions_after"] += len(new_target)
        if len(new_target) < MIN_QUESTIONS:
            stats["questions_dropped"] += 1
            continue
        assistant = json.dumps({"questions": new_target}, ensure_ascii=False)
        system = QUESTIONS_SYSTEM
    else:
        stats["metadata_in"] += 1
        target = json.loads(assistant)
        if isinstance(target.get("semester"), str):
            stats["metadata_semester_fixed"] += 1
        target = {
            "course_code": normalize_course_code(target.get("course_code")),
            "course_name": (target.get("course_name") or "").strip() or None,
            "year": normalize_year(target.get("year")),
            "semester": normalize_semester(target.get("semester")),
        }
        if not any(target.values()):
            continue
        assistant = json.dumps(target, ensure_ascii=False)
        system = METADATA_SYSTEM

    fixed.append({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]})

# ---------- dedup on task + full input ----------
seen, unique = set(), []
for ex in fixed:
    key = hashlib.md5((ex["messages"][0]["content"] + ex["messages"][1]["content"]).encode()).hexdigest()
    if key not in seen:
        seen.add(key)
        unique.append(ex)

random.shuffle(unique)
val_size = max(1, int(len(unique) * VAL_SPLIT))
val_set, train_set = unique[:val_size], unique[val_size:]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_set)
write_jsonl(os.path.join(OUTPUT_DIR, "valid.jsonl"), val_set)

report = {**stats, "total_examples": len(unique), "train": len(train_set), "val": len(val_set)}
with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print(f"\nWrote {OUTPUT_DIR}/train.jsonl and {OUTPUT_DIR}/valid.jsonl")
