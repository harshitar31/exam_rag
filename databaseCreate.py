import argparse
import hashlib
import json
import os
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from common import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE, embedding_text, get_client

# ================= CONFIG =================
JSONL_FILE = "dspace_questions_tagged.jsonl"    # Output of cleaning.py -> difficultyTagging.py
LAST_COUNT_FILE = "last_count.txt"              # Track processed count

parser = argparse.ArgumentParser(description="Embed questions and upsert them into Qdrant.")
parser.add_argument("--rebuild", action="store_true",
                    help="Drop the collection and re-index everything (needed after re-cleaning the data).")
args = parser.parse_args()

# ================= QDRANT SETUP =================
client = get_client()

if args.rebuild and client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"[REBUILD] Dropped collection '{COLLECTION_NAME}'")

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    # Indexes for the filters used by the search UI
    for field, schema in [("course_code", PayloadSchemaType.KEYWORD),
                          ("difficulty", PayloadSchemaType.KEYWORD),
                          ("year", PayloadSchemaType.INTEGER),
                          ("semester", PayloadSchemaType.INTEGER),
                          ("marks", PayloadSchemaType.INTEGER)]:
        client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
    print(f"[INIT] Created Qdrant collection '{COLLECTION_NAME}'")
else:
    print(f"[INFO] Using existing Qdrant collection '{COLLECTION_NAME}'")

# ================= PROGRESS TRACKING =================
last_count = 0
if not args.rebuild and os.path.exists(LAST_COUNT_FILE):
    with open(LAST_COUNT_FILE, "r") as f:
        last_count = int(f.read().strip() or 0)

# ================= LOAD JSONL DATA =================
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

new_lines = lines[last_count:]
print(f"[INFO] Found {len(new_lines)} new JSONL entries to process.\n")

# ================= LOAD EMBEDDING MODEL =================
print(f"[MODEL] Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print("[MODEL] Model loaded successfully.\n")

# ================= INSERT INTO QDRANT =================
for idx, line in enumerate(new_lines, start=1):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"[WARN] Skipping invalid JSON line {last_count + idx}: {e}")
        last_count += 1
        with open(LAST_COUNT_FILE, "w") as f:
            f.write(str(last_count))
        continue

    course_code = obj.get("course_code") or ""
    course_name = obj.get("course_name") or ""
    file_link = obj.get("file_link", "")

    questions = [q for q in obj.get("questions", []) if q.get("question_text", "").strip()]
    if questions:
        # Embed the whole paper in one batch; year/semester are left out of the
        # text because they are handled by filters, and digits only add noise.
        texts = [embedding_text(course_name, q["question_text"].strip()) for q in questions]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64)

        points = []
        for q, emb in zip(questions, embeddings):
            question_text = q["question_text"].strip()
            # Deterministic ID per (paper, question): the same question asked in
            # different years is kept, so the UI can show "asked in 2016, 2018".
            qid = str(uuid.UUID(hashlib.md5(f"{file_link}{question_text}".encode()).hexdigest()))
            payload = {
                "course_code": course_code,
                "course_name": course_name,
                "semester": obj.get("semester"),
                "year": obj.get("year"),
                "marks": q.get("marks"),
                "difficulty": q.get("difficulty"),
                "question_text": question_text,
                "file_link": file_link
            }
            points.append(PointStruct(id=qid, vector=emb.tolist(), payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Update progress file
    last_count += 1
    with open(LAST_COUNT_FILE, "w") as f:
        f.write(str(last_count))

    if idx % 25 == 0:
        print(f"[INFO] Processed {idx}/{len(new_lines)} entries")

print("\n[DONE] All new questions inserted into Qdrant with embeddings and difficulty levels.")
