import json
import os
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# ================= CONFIG =================
JSONL_FILE = "dspace_questions_metadata.jsonl"
LAST_COUNT_FILE = "last_count.txt"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "questions"

# ================= QDRANT =================
client = QdrantClient(host="localhost", port=6333)

# Create collection if not exists
existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"}  # adjust size based on embedding model
    )

# ================= PROGRESS =================
if os.path.exists(LAST_COUNT_FILE):
    with open(LAST_COUNT_FILE, "r") as f:
        last_count = int(f.read().strip() or 0)
else:
    last_count = 0

# ================= LOAD JSONL =================
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

new_lines = lines[last_count:]
print(f"[INFO] Found {len(new_lines)} new JSONL entries to process.")

# ================= EMBEDDING MODEL =================
model = SentenceTransformer(EMBEDDING_MODEL)

# ================= INSERT INTO QDRANT =================
for idx, line in enumerate(new_lines, start=1):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"[WARN] Skipping invalid JSON line {last_count + idx}: {e}")
        # Update last_count even if skipping, to avoid re-processing
        last_count += 1
        with open(LAST_COUNT_FILE, "w") as f:
            f.write(str(last_count))
        continue

    course_code = obj.get("course_code", "")
    course_name = obj.get("course_name", "")
    semester = obj.get("semester", "")
    year = obj.get("year")
    file_link = obj.get("file_link", "")

    points = []
    for q in obj.get("questions", []):
        question_text = q.get("question_text", "").strip()
        marks = q.get("marks", None)

        if not question_text:
            continue

        # Unique ID
        qid = hashlib.md5(f"{course_code}{course_name}{question_text}".encode()).hexdigest()

        # Embedding
        text_for_embedding = f"{course_name} {semester} {year or ''} {question_text}"
        emb = model.encode(text_for_embedding, normalize_embeddings=True).tolist()

        payload = {
            "course_code": course_code,
            "course_name": course_name,
            "semester": semester,
            "year": year,
            "marks": marks,
            "question_text": question_text,
            "file_link": file_link
        }

        points.append(PointStruct(id=qid, vector=emb, payload=payload))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Update last_count after processing this JSON object
    last_count += 1
    with open(LAST_COUNT_FILE, "w") as f:
        f.write(str(last_count))

    if idx % 5 == 0:
        print(f"[INFO] Processed {idx}/{len(new_lines)} JSONL entries")

print("[DONE] All new questions inserted into Qdrant with embeddings.")
