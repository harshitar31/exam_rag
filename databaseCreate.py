import json
import os
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# ================= CONFIG =================
JSONL_FILE = "dspace_difficulty_bge.jsonl"  # Input data file
LAST_COUNT_FILE = "last_count.txt"              # Track processed count
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"       # Semantic model
COLLECTION_NAME = "questions"                   # Qdrant collection name

# ================= QDRANT SETUP =================
client = QdrantClient(host="localhost", port=6333)

# Check if collection exists, else create it
existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"}  # 768D for BAAI base
    )
    print(f"[INIT] Created Qdrant collection '{COLLECTION_NAME}'")
else:
    print(f"[INFO] Using existing Qdrant collection '{COLLECTION_NAME}'")

# ================= PROGRESS TRACKING =================
if os.path.exists(LAST_COUNT_FILE):
    with open(LAST_COUNT_FILE, "r") as f:
        last_count = int(f.read().strip() or 0)
else:
    last_count = 0

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

    course_code = obj.get("course_code", "")
    course_name = obj.get("course_name", "")
    semester = obj.get("semester", "")
    year = obj.get("year")
    file_link = obj.get("file_link", "")

    points = []
    for q in obj.get("questions", []):
        question_text = q.get("question_text", "").strip()
        marks = q.get("marks", None)
        difficulty = q.get("difficulty", None)  # <-- Added field

        if not question_text:
            continue

        # Generate unique deterministic ID
        qid = hashlib.md5(f"{course_code}{course_name}{question_text}".encode()).hexdigest()

        # Create text for embedding (no metadata included)
        text_for_embedding = f"{course_name} {semester} {year or ''} {question_text}"
        emb = model.encode(text_for_embedding, normalize_embeddings=True).tolist()

        # Build payload (metadata)
        payload = {
            "course_code": course_code,
            "course_name": course_name,
            "semester": semester,
            "year": year,
            "marks": marks,
            "difficulty": difficulty,        # <-- Stored in payload only
            "question_text": question_text,
            "file_link": file_link
        }

        points.append(PointStruct(id=qid, vector=emb, payload=payload))

    # Upsert all question points from this object
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    # Update progress file
    last_count += 1
    with open(LAST_COUNT_FILE, "w") as f:
        f.write(str(last_count))

    if idx % 5 == 0:
        print(f"[INFO] Processed {idx}/{len(new_lines)} entries")

print("\n[DONE] All new questions inserted into Qdrant with embeddings and difficulty levels.")
