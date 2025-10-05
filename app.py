from flask import Flask, request, jsonify, render_template, session
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import random

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# ================= CONFIG =================
COLLECTION_NAME = "questions"
SNAPSHOT_DIR = "snapshots"
TOP_K = 20
THRESHOLD = 0.1

# ================= QDRANT =================
client = QdrantClient(url="http://localhost:6333")

# Restore snapshot if collection doesn't exist
existing_collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    # Recreate collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"}
    )

    # Restore latest snapshot
    if os.path.exists(SNAPSHOT_DIR) and os.listdir(SNAPSHOT_DIR):
        latest_snapshot = sorted(os.listdir(SNAPSHOT_DIR))[-1]
        client.snapshot.restore(
            collection_name=COLLECTION_NAME,
            location=os.path.join(SNAPSHOT_DIR, latest_snapshot)
        )
        print(f"[INFO] Restored {COLLECTION_NAME} from snapshot {latest_snapshot}")
    else:
        print("[WARN] No snapshot found! Database will be empty.")

# ================= MODEL =================
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ================= SEARCH FUNCTION =================
def search(query, offset=0):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()
    
    # Search in Qdrant
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb[0],
        limit=TOP_K + offset  # fetch extra for pagination
    )

    # Filter by threshold and paginate
    filtered = [hit for hit in hits if hit.score >= THRESHOLD]
    paginated = filtered[offset: offset + TOP_K]

    results = []
    for hit in paginated:
        payload = hit.payload
        payload["score"] = hit.score
        # Only include year/semester if present
        if not payload.get("year"):
            payload.pop("year", None)
        if not payload.get("semester"):
            payload.pop("semester", None)
        results.append(payload)
    return results

# ================= FLASK ROUTES =================
@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json()
    query = data.get("query", "").strip()
    offset = int(data.get("offset", 0))
    results = search(query, offset=offset)
    return jsonify({"results": results})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_paper", methods=["POST"])
def generate_paper_route():
    data = request.get_json()
    subject = data.get("subject", "").strip()
    if not subject:
        return jsonify({"paper": [], "paper_id": None})

    # Create embedding for subject
    query_vec = model.encode(subject, normalize_embeddings=True).tolist()

    # Fetch top questions
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=30,
        with_payload=True
    )

    if not hits:
        return jsonify({"paper": [], "paper_id": None})

    # Pick 10 random questions from top hits
    selected = random.sample(hits, min(10, len(hits)))

    paper = []
    for i, h in enumerate(selected, start=1):
        q_text = h.payload.get("question_text", "No text found.")
        marks = h.payload.get("marks", "")
        paper.append({
            "number": i,
            "question": q_text,
            "marks": marks
        })

    # Generate unique paper ID and store in session
    paper_id = os.urandom(8).hex()
    session[paper_id] = {
        "subject": subject,
        "questions": paper
    }

    return jsonify({"paper": paper, "paper_id": paper_id})

@app.route("/paper/<paper_id>")
def view_paper(paper_id):
    paper_data = session.get(paper_id)
    if not paper_data:
        return "Paper not found or expired", 404
    
    return render_template("paper.html", 
                         subject=paper_data["subject"],
                         questions=paper_data["questions"])

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)