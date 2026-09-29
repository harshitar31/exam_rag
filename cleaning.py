"""
Normalises the records produced by processPDF.py so that filters and
search work on consistent values.

- semester  -> int (1-8) or None      ("Third / Fifth" -> 3, "Second Semester" -> 2)
- year      -> 4-digit int or None    (drops garbage like 2012017)
- course_code -> upper case, no inner spaces ("ENG 111" -> "ENG111")
- question_text -> no leading numbering / "Part B:" labels, no trailing
                   "(10 marks)" / "[CO02]" tags (marks are moved into "marks")
- exact duplicate questions inside one paper are dropped

Usage:
    python cleaning.py        # dspace_questions_metadata.jsonl -> dspace_questions_clean.jsonl
"""

import json
import re

INPUT_FILE = "dspace_questions_metadata.jsonl"
OUTPUT_FILE = "dspace_questions_clean.jsonl"

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}

# ---------------- metadata ----------------

def normalize_semester(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 8 else None
    if not isinstance(value, str):
        return None
    # Take the first semester mentioned: "Third / Fifth" -> 3, "S3/S5" -> 3
    for token in re.findall(r"[a-z]+|\d+", value.lower()):
        if token in ORDINALS:
            return ORDINALS[token]
        if token in ROMAN:
            return ROMAN[token]
        if token.isdigit() and 1 <= int(token) <= 8:
            return int(token)
    return None


def normalize_year(value):
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1990 <= year <= 2035 else None


def normalize_course_code(value):
    if not value:
        return None
    # "CH329 / CHY273" keeps both codes; "ENG 111" -> "ENG111"
    parts = [re.sub(r"\s+", "", p).upper() for p in str(value).split("/")]
    return " / ".join(p for p in parts if p) or None

# ---------------- question text ----------------

PART_PREFIX = re.compile(
    r"^\s*part\s*[-–—]?\s*[a-c]\b\s*(?:\([^)]{0,40}\))?\s*[:\-–—.]*\s*", re.IGNORECASE)
# "4. ", "11b. ", "13(a): ", "6.(a) ", "11.A. ", "Q.3 " -- but not "4.5 kg" or "12.0 g"
NUMBER_PREFIX = re.compile(
    r"^\s*(?:q\.?\s*\d{1,2}\s*[.):]?"
    r"|\d{1,2}\s*\.?\s*\(\s*[a-h]\s*\)\s*[.:]?"
    r"|\d{1,2}\s*(?:\.?[a-h])?\s*[.):])(?=\s)\s*", re.IGNORECASE)
LETTER_PREFIX = re.compile(r"^\s*(?:\(\s*[a-hA-H]\s*\)|[a-h][.)]):?\s+")
ROMAN_PREFIX = re.compile(r"^\s*(?:\(\s*[ivx]{1,4}\s*\)|[ivx]{1,4}[.)]):?\s+")

TRAILING_MARKS = re.compile(
    r"\s+[\(\[]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\)\]]\s*$", re.IGNORECASE)
TRAILING_TAG = re.compile(r"\s*\[\s*(?:CO|BTL|K|L)[\w\s&,\-]*\]\s*$", re.IGNORECASE)


def clean_question_text(text):
    """Returns (clean_text, marks_found_in_text_or_None)."""
    text = re.sub(r"[ \t  ]+", " ", text or "").strip()

    for _ in range(4):
        before = text
        for pattern in (PART_PREFIX, NUMBER_PREFIX, LETTER_PREFIX, ROMAN_PREFIX):
            text = pattern.sub("", text, count=1)
        if text == before:
            break

    marks = None
    for _ in range(3):
        before = text
        text = TRAILING_TAG.sub("", text)
        m = TRAILING_MARKS.search(text)
        if m:
            marks = marks or int(m.group(1))
            text = text[:m.start()]
        if text == before:
            break

    return text.strip(), marks


def clean_record(record):
    record = dict(record)
    record["course_code"] = normalize_course_code(record.get("course_code"))
    record["course_name"] = (record.get("course_name") or "").strip() or None
    record["semester"] = normalize_semester(record.get("semester"))
    record["year"] = normalize_year(record.get("year"))

    seen = set()
    questions = []
    for q in record.get("questions", []):
        text, text_marks = clean_question_text(q.get("question_text", ""))
        if len(text) < 10:
            continue
        key = re.sub(r"\W+", "", text.lower())
        if key in seen:
            continue
        seen.add(key)

        marks = q.get("marks")
        if not isinstance(marks, int) or marks <= 0:
            marks = text_marks
        questions.append({**q, "question_text": text, "marks": marks})

    record["questions"] = questions
    return record


def main():
    kept = skipped = 0
    with open(INPUT_FILE, encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON on line {line_no}: {e}")
                skipped += 1
                continue
            record = clean_record(record)
            if not record["questions"]:
                skipped += 1
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] Wrote {kept} papers to {OUTPUT_FILE} ({skipped} skipped)")


if __name__ == "__main__":
    main()
