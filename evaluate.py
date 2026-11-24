"""
evaluate.py

Scores an extraction model on the labelled sample papers in sample_pdfs/
(no network needed -- the PDFs are local).

Metrics
  metadata   per-field exact match (course_code, course_name, year, semester)
  questions  json_ok      model returned parseable JSON
             recall       share of labelled questions the model extracted
             grounded     share of extracted questions that really are in the
                          PDF text (1 - hallucination rate); needs no labels
             marks_acc    marks correct, among matched questions

Papers that were part of the training data (in_training=true in labels.jsonl)
are reported separately -- only the held-out numbers say how well the model
generalises.

Usage:
    python evaluate.py                                   # fine-tuned model in exam_rag_model/
    python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit   # base model, for comparison
    python evaluate.py --tasks metadata                  # quick run
    python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter adapters   # after finetune.sh
"""

import argparse
import json
import re
import time
from collections import defaultdict

from prep_utils import PAGE_BREAK, extract_pdf_text, grounded_questions, split_parts
from prompts import METADATA_SYSTEM, QUESTIONS_SYSTEM

# ===================== CONFIG =====================
LABELS_FILE = "sample_pdfs/labels.jsonl"
PDF_DIR = "sample_pdfs"
OUTPUT_FILE = "evaluation_output.jsonl"
FIRST_PAGE_MAX = 1500
CHUNK_CHARS = 6000        # training inputs were <= 8000 chars; stay inside that
MAX_NEW_TOKENS_META = 200
MAX_NEW_TOKENS_QUESTIONS = 3000
MATCH_THRESHOLD = 0.7     # share of a labelled question's words the prediction must contain
# ==================================================


# -------------------- model backends --------------------

class MLXModel:
    """Apple Silicon (fast). Loads HF-format folders or mlx-community repos."""
    def __init__(self, path, adapter=None):
        from mlx_lm import load
        self.model, self.tokenizer = load(path, adapter_path=adapter)

    def chat(self, system, user, max_tokens):
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


class HFModel:
    """transformers backend, for machines without MLX (e.g. Colab)."""
    def __init__(self, path, adapter=None):
        import torch
        if adapter:
            raise SystemExit("--adapter is only supported with --backend mlx")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # transformers 4.56 renamed torch_dtype to dtype
        version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if version >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", **{dtype_kw: torch.float16})
        self.model.eval()

    def chat(self, system, user, max_tokens):
        # Render to text, then tokenize: apply_chat_template's tokenized return
        # type differs between transformers versions
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# -------------------- parsing --------------------

def safe_parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


QUESTION_OBJECT = re.compile(
    r'\{\s*"question_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"marks"\s*:\s*(null|-?\d+)\s*\}')


def parse_questions(text):
    """
    Questions JSON, or -- when the output was cut off (the model sometimes
    loops until max_tokens and never closes the array) -- every complete
    {"question_text": ..., "marks": ...} object it did produce.
    Returns (questions, complete_json).
    """
    parsed = safe_parse_json(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        return [q for q in parsed["questions"] if isinstance(q, dict) and q.get("question_text")], True
    questions = []
    for m in QUESTION_OBJECT.finditer(text):
        try:
            question_text = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            continue
        questions.append({"question_text": question_text,
                          "marks": None if m.group(2) == "null" else int(m.group(2))})
    return questions, False


def chunks_for_extraction(full_text):
    """
    Part A / Part B sections (as in training), further split on page breaks so
    no chunk exceeds CHUNK_CHARS. A single 2000-token generation over a whole
    paper gets cut off before the last questions.
    """
    part_a, part_b = split_parts(full_text)
    sections = [part_a, part_b] if part_b else [full_text]
    chunks = []
    for section in sections:
        current = ""
        for page in section.split(PAGE_BREAK):
            if current and len(current) + len(page) > CHUNK_CHARS:
                chunks.append(current)
                current = ""
            current = (current + PAGE_BREAK + page) if current else page
        if current.strip():
            chunks.append(current[:CHUNK_CHARS * 2])
    return chunks


# -------------------- scoring --------------------

def norm_text(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def score_metadata(pred, gold):
    pred = pred or {}
    codes = [gold["course_code"]] + gold.get("course_code_alternatives", [])
    return {
        "course_code": norm_text(pred.get("course_code")) in {norm_text(c) for c in codes},
        "course_name": norm_text(pred.get("course_name")) == norm_text(gold["course_name"]),
        "year": pred.get("year") == gold["year"],
        "semester": pred.get("semester") == gold["semester"],
    }


def words(text):
    return re.findall(r"[a-z0-9]{3,}", str(text).lower())


def match_questions(pred_qs, gold_qs):
    """Greedy one-to-one match: a gold question is found if a prediction contains most of its words."""
    used = set()
    matches = []
    for g in gold_qs:
        gw = set(words(g["question_text"]))
        best, best_score = None, 0
        for i, p in enumerate(pred_qs):
            if i in used or not gw:
                continue
            score = len(gw & set(words(p.get("question_text", "")))) / len(gw)
            if score > best_score:
                best, best_score = i, score
        if best is not None and best_score >= MATCH_THRESHOLD:
            used.add(best)
            matches.append((g, pred_qs[best]))
    return matches


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exam_rag_model")
    parser.add_argument("--adapter", default=None, help="LoRA adapter folder from finetune.sh")
    parser.add_argument("--backend", choices=["mlx", "hf"], default="mlx")
    parser.add_argument("--tasks", default="metadata,questions")
    parser.add_argument("--labels", default=LABELS_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tasks = set(args.tasks.split(","))

    with open(args.labels) as f:
        labels = [json.loads(line) for line in f if line.strip()][:args.limit]

    name = args.model + (f" + {args.adapter}" if args.adapter else "")
    print(f"Loading {name} ({args.backend})...")
    model = (MLXModel if args.backend == "mlx" else HFModel)(args.model, args.adapter)

    rows = []
    for item in labels:
        with open(f"{PDF_DIR}/{item['local_file']}", "rb") as f:
            first_page, full_text = extract_pdf_text(f.read(), FIRST_PAGE_MAX, max_chars=None)
        row = {"file": item["local_file"], "in_training": item["in_training"]}
        start = time.time()

        if "metadata" in tasks:
            raw = model.chat(METADATA_SYSTEM, first_page, MAX_NEW_TOKENS_META)
            pred = safe_parse_json(raw)
            row["metadata_pred"] = pred if pred is not None else raw
            row["metadata_correct"] = score_metadata(pred, item["metadata"])

        if "questions" in tasks:
            pred_qs, parsed_chunks, chunks = [], 0, chunks_for_extraction(full_text)
            seen = set()
            for chunk in chunks:
                questions, complete = parse_questions(model.chat(QUESTIONS_SYSTEM, chunk, MAX_NEW_TOKENS_QUESTIONS))
                parsed_chunks += complete
                for q in questions:
                    key = norm_text(q["question_text"])
                    if key not in seen:     # looping output repeats questions
                        seen.add(key)
                        pred_qs.append(q)
            matches = match_questions(pred_qs, item["questions"])
            grounded = grounded_questions(full_text, pred_qs)
            row.update({
                "questions_pred": pred_qs,
                "json_ok": parsed_chunks / len(chunks) if chunks else 0,
                "n_pred": len(pred_qs),
                "n_gold": len(item["questions"]),
                "recall": len(matches) / len(item["questions"]) if item["questions"] else None,
                "grounded": len(grounded) / len(pred_qs) if pred_qs else None,
                "marks_acc": (sum(p.get("marks") == g["marks"] for g, p in matches) / len(matches)) if matches else None,
            })

        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        summary = {k: row[k] for k in ("recall", "grounded", "n_pred", "n_gold") if k in row}
        print(f"  {item['local_file']:<34} {row.get('metadata_correct', '')} {summary} {row['seconds']}s")

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps({"model": name, **row}, ensure_ascii=False) + "\n")

    # ---------- summary ----------
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    print(f"\n===== {name} =====")
    for split_name, in_training in (("held-out", False), ("seen in training", True)):
        split = [r for r in rows if r["in_training"] == in_training]
        if not split:
            continue
        print(f"\n[{split_name}] {len(split)} papers")
        if "metadata" in tasks:
            fields = defaultdict(list)
            for r in split:
                for field, ok in r["metadata_correct"].items():
                    fields[field].append(ok)
            print("  metadata  " + "  ".join(f"{k}={mean(v):.0%}" for k, v in fields.items()))
        if "questions" in tasks:
            metrics = {k: mean(r[k] for r in split) for k in ("json_ok", "recall", "grounded", "marks_acc")}
            print("  questions " + "  ".join(f"{k}={v:.0%}" if v is not None else f"{k}=n/a" for k, v in metrics.items()))
    print(f"\nPer-paper outputs: {args.output}")


if __name__ == "__main__":
    main()
