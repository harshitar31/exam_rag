import sqlite3
import json
import os
from sentence_transformers import SentenceTransformer, models

conn = sqlite3.connect("questions.db")
cur = conn.cursor()

# --- Updated schema with marks ---
cur.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,
    course_name TEXT,
    year INTEGER,
    semester INTEGER,
    marks INTEGER,              -- added marks
    question_text TEXT,
    file_link TEXT,
    embedding TEXT
)
""")
conn.commit()

jsonl_file = "dspace_questions_metadata_new.jsonl"
count_file = "last_count.txt"

# Load last processed count
if os.path.exists(count_file):
    with open(count_file, "r") as f:
        last_count = int(f.read().strip() or 0)
else:
    last_count = 0

# Read all JSONL lines
with open(jsonl_file, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

# Get only new lines
new_lines = lines[last_count:]
print(f"Found {len(new_lines)} new entries to insert.")

# Insert new questions with marks + semester
for line in new_lines:
    obj = json.loads(line)
    cur.execute("""
        INSERT INTO questions (course_code, course_name, year, semester, marks, question_text, file_link)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        obj.get("course_code"),
        obj.get("course_name"),
        obj.get("year"),
        obj.get("semester"),
        obj.get("marks"),    # added marks field
        obj.get("question_text"),
        obj.get("file_link")
    ))

conn.commit()

# Save new count
with open(count_file, "w") as f:
    f.write(str(len(lines)))

# ---- Embeddings Part ----
model_name = "D:/VolumeEStuff/SEM 5/NLP/case study/bge-base-en-v1.5"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def get_embedding(cname: str, sem: str, qtext: str):
    if not qtext or not qtext.strip() or not cname or not cname.strip():
        return None
    text = f"{cname} semester{sem} {qtext}"
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.tolist()

rows = cur.execute("SELECT id, course_name, semester, question_text FROM questions WHERE embedding IS NULL").fetchall()

for qid, cname, sem, qtext in rows:
    emb = get_embedding(cname, sem,qtext)
    if emb is not None:
        cur.execute("UPDATE questions SET embedding = ? WHERE id = ?", (json.dumps(emb), qid))
        print(f"Done for {qid}")

conn.commit()
conn.close()

print("[DONE]: Added only new JSONL rows since last run + embeddings updated.")
