from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm

# ===== Model =====
print("[INFO] Loading model BAAI/bge-base-en-v1.5 ...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ===== Reference difficulty embeddings =====
difficulty_descriptions = {
    "easy": "This is a simple and direct question that tests recall or basic understanding.",
    "medium": "This question needs reasoning, moderate problem solving, or code tracing.",
    "difficult": "This question involves analysis, multi-step reasoning, or synthesis of knowledge."
}

difficulty_embs = {
    k: model.encode(v, normalize_embeddings=True) for k, v in difficulty_descriptions.items()
}

# ===== Input / Output =====
INPUT_FILE = "dspace_questions_metadata.jsonl"
OUTPUT_FILE = "dspace_difficulty_bge.jsonl"

# ===== Process =====
valid_count = 0
skip_count = 0
line_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in tqdm(fin, desc="Classifying with BGE"):
        line_count += 1
        line = line.strip()
        if not line:
            skip_count += 1
            continue  # skip empty lines

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping malformed JSON on line {line_count}: {e}")
            skip_count += 1
            continue

        # Process questions only if they exist
        questions = data.get("questions", [])
        if not questions:
            continue

        for q in questions:
            q_text = q.get("question_text", "").strip()
            if not q_text:
                continue

            # Encode question
            q_emb = model.encode(q_text, normalize_embeddings=True)

            # Compute cosine similarity with each difficulty level
            scores = {lvl: util.cos_sim(q_emb, emb).item() for lvl, emb in difficulty_embs.items()}
            best_level = max(scores, key=scores.get)

            # Assign results
            q["difficulty"] = best_level
            q["confidence"] = round(scores[best_level], 3)

        # Write back to output file
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        valid_count += 1

print(f"\n✅ Done! Saved difficulty predictions to {OUTPUT_FILE}")
print(f"[STATS] Processed {valid_count} valid records, skipped {skip_count} malformed/empty lines.")
