import sqlite3
import json
import os
from sentence_transformers import SentenceTransformer

# ---------- DB SETUP ----------
conn = sqlite3.connect("questions.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,
    course_name TEXT,
    year INTEGER,
    semester INTEGER,
    marks INTEGER,
    question_text TEXT,
    file_link TEXT,
    images TEXT,      -- store list of base64 images as JSON
    embedding TEXT
)
""")
conn.commit()

# ---------- FILES ----------
jsonl_file = "dspace_questions_metadata.jsonl"
count_file = "last_count.txt"

# Load last processed count
last_count = 0
if os.path.exists(count_file):
    with open(count_file, "r") as f:
        last_count = int(f.read().strip() or 0)

# Read JSONL
with open(jsonl_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

new_lines = lines[last_count:]
print(f"[INFO] Found {len(new_lines)} new entries to insert.")

# ---------- INSERT NEW QUESTIONS ----------
for idx, line in enumerate(new_lines, start=1):
    obj = json.loads(line)
    cur.execute("""
        INSERT INTO questions (course_code, course_name, year, semester, marks, question_text, file_link, images)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        obj.get("course_code"),
        obj.get("course_name"),
        obj.get("year"),
        obj.get("semester"),
        obj.get("marks") if obj.get("marks") is not None else None,
        obj.get("question_text"),
        obj.get("file_link"),
        json.dumps(obj.get("images", []))  # store images as JSON string
    ))
    if idx % 50 == 0:
        conn.commit()
conn.commit()
print(f"[INFO] Inserted {len(new_lines)} new rows.")

# Update last_count
with open(count_file, "w") as f:
    f.write(str(len(lines)))

# ---------- EMBEDDINGS ----------
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def get_embedding(course_name: str, semester: str, question_text: str):
    if not question_text or not question_text.strip() or not course_name or not course_name.strip():
        return None
    text = f"{course_name} semester {semester} {question_text}"
    emb = embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.tolist()

rows = cur.execute("""
SELECT id, course_name, semester, question_text 
FROM questions 
WHERE embedding IS NULL
""").fetchall()

for idx, (qid, cname, sem, qtext) in enumerate(rows, start=1):
    emb = get_embedding(cname, str(sem), qtext)
    if emb is not None:
        cur.execute("UPDATE questions SET embedding = ? WHERE id = ?", (json.dumps(emb), qid))
    if idx % 20 == 0:
        conn.commit()
        print(f"[INFO] Processed {idx}/{len(rows)} embeddings")
conn.commit()
conn.close()

print("[DONE] Added new JSONL rows + updated embeddings + stored images.")
