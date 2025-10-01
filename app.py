from flask import Flask, request, jsonify, render_template
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

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
    return render_template("index.html")  # HTML in templates/index.html

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
