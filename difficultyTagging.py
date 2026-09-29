"""
Rule-based difficulty tagging (easy / medium / difficult).

The previous version compared each question's embedding to three one-line
descriptions of "easy", "medium" and "difficult". Those similarities were all
~0.5, so the labels were close to random (62% came out "difficult").

This version combines two signals that actually correlate with difficulty in
exam papers:
  1. marks            -- 1-2 marks is recall, 8+ marks is long-form
  2. the task verb    -- Bloom's taxonomy: define/list < explain/solve < derive/design
Each available signal gives a level 0/1/2; the average is rounded.
"""

import json
import re
from collections import Counter

INPUT_FILE = "dspace_questions_clean.jsonl"     # output of cleaning.py
OUTPUT_FILE = "dspace_questions_tagged.jsonl"   # input of databaseCreate.py

LEVELS = ["easy", "medium", "difficult"]

# Bloom's taxonomy verbs, grouped into our three levels
VERB_LEVELS = {
    0: [
        "define", "what is", "what are", "what do you mean", "what is meant", "list", "state",
        "name", "mention", "expand", "recall", "identify", "give the", "give any", "fill in",
        "match the", "choose", "select", "true or false", "who ", "when ", "which ", "write the full form",
        "what does", "enumerate", "label",
    ],
    1: [
        "explain", "describe", "discuss", "compare", "distinguish", "differentiate", "illustrate",
        "calculate", "compute", "find", "determine", "solve", "draw", "sketch", "classify",
        "write a program", "write a c", "write a short note", "write short note", "how ",
        "why ", "predict", "convert", "obtain", "plot", "show that", "estimate", "summarize",
        "outline", "brief", "interpret", "apply", "use ",
    ],
    2: [
        "derive", "prove", "design", "analyse", "analyze", "evaluate", "develop", "construct",
        "justify", "optimize", "optimise", "formulate", "critically", "propose", "assess",
        "examine", "investigate", "deduce", "synthesize", "synthesise", "validate",
    ],
}


def verb_level(text):
    """Level of the hardest task verb in the question, or None."""
    lowered = " " + text.lower()
    found = None
    for level, verbs in VERB_LEVELS.items():
        for verb in verbs:
            # whole words only, so "constructor" doesn't count as "construct"
            if re.search(r"[\s(\"'‘“]" + re.escape(verb.strip()) + r"\b", lowered):
                found = level if found is None else max(found, level)
                break
    return found


def marks_level(marks):
    if not isinstance(marks, (int, float)) or marks <= 0:
        return None
    if marks <= 2:
        return 0
    if marks <= 7:
        return 1
    return 2


def tag_difficulty(question_text, marks):
    signals = [s for s in (marks_level(marks), verb_level(question_text)) if s is not None]
    if not signals:
        # Fall back to length: long, multi-part questions tend to be harder
        return LEVELS[0 if len(question_text) < 80 else 1 if len(question_text) < 300 else 2]
    level = sum(signals) / len(signals)
    # Very long questions (multi-step problems) get nudged up
    if len(question_text) > 350:
        level += 0.5
    return LEVELS[min(2, int(level + 0.5))]


def main():
    counts = Counter()
    papers = 0
    with open(INPUT_FILE, encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for q in data.get("questions", []):
                q.pop("confidence", None)
                q["difficulty"] = tag_difficulty(q["question_text"], q.get("marks"))
                counts[q["difficulty"]] += 1
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            papers += 1

    total = sum(counts.values())
    print(f"✅ Tagged {total} questions from {papers} papers -> {OUTPUT_FILE}")
    for level in LEVELS:
        print(f"   {level:<10} {counts[level]:>6}  ({counts[level] / total:.0%})")


if __name__ == "__main__":
    main()
