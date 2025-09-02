import sqlite3, json, numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---- Load local BGE-base ----
model_name = "D:/VolumeEStuff/SEM 5/NLP/case study/bge-base-en-v1.5"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load embeddings once at startup
conn = sqlite3.connect("questions.db")
cur = conn.cursor()
rows = cur.execute("""
    SELECT id, question_text, course_code, course_name, year, semester, marks, file_link, embedding 
    FROM questions
""").fetchall()
conn.close()

# Store data into lists
ids, texts, codes, names, years, semesters, marks_list, links, embs = [], [], [], [], [], [], [], [], []
for qid, qtext, ccode, cname, year, semester, marks, link, emb_json in rows:
    ids.append(qid)
    texts.append(qtext)
    codes.append(ccode)
    names.append(cname)
    years.append(year)
    semesters.append(semester)
    marks_list.append(marks)
    links.append(link)
    embs.append(np.array(json.loads(emb_json)))

embs = np.vstack(embs)  # shape: (num_questions, dim)


# Search function
def search(query, top_k=10):
    q_emb = model.encode([query], normalize_embeddings=True)
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
            "score": float(sims[i])
        })
    return results


# Flask routes
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
