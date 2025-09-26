import sqlite3
import json
import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---- Load BGE-base model ----
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ---- Load data from DB ----
conn = sqlite3.connect("questions.db", check_same_thread=False)
cur = conn.cursor()
rows = cur.execute("""
    SELECT id, question_text, course_code, course_name, year, semester, marks, file_link, embedding, images
    FROM questions
    WHERE embedding IS NOT NULL
""").fetchall()

# ---- Store data in lists ----
ids, texts, codes, names, years, semesters, marks_list, links, embs, images_list = [], [], [], [], [], [], [], [], [], []

for qid, qtext, ccode, cname, year, semester, marks, link, emb_json, images_json in rows:
    try:
        emb_array = np.array(json.loads(emb_json))
    except:
        continue
    ids.append(qid)
    texts.append(qtext)
    codes.append(ccode)
    names.append(cname)
    years.append(year)
    semesters.append(semester)
    marks_list.append(marks)
    links.append(link)
    embs.append(emb_array)
    images_list.append(json.loads(images_json) if images_json else [])

if embs:
    embs = np.vstack(embs)
else:
    embs = np.zeros((0, model.get_sentence_embedding_dimension()))

# ---- Search function ----
def search(query, top_k=10):
    q_emb = model.encode([query], normalize_embeddings=True)
    if embs.shape[0] == 0:
        return []
    sims = cosine_similarity(q_emb, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "id": ids[i],
            "question_text": texts[i],
            "course_code": codes[i],
            "course_name": names[i],
            "year": years[i],
            "semester": semesters[i],
            "marks": marks_list[i],
            "file_link": links[i],
            "images": images_list[i],
            "score": float(sims[i])
        })
    return results

# ---- Flask routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        results = search(query, top_k=50)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
