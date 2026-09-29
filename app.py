from flask import Flask, request, jsonify, render_template
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import SentenceTransformer
import os
import re
import requests

from common import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH, QDRANT_URL, QUERY_PREFIX, get_client

app = Flask(__name__)

# ================= CONFIG =================
SNAPSHOT_DIR = "snapshots"
TOP_K = 20
# Minimum cosine similarity. With the query prefix, on-topic queries score
# 0.55-0.8 and an off-topic one ("pizza recipe") tops out at 0.50, so the old
# cutoff of 0.1 let everything through.
THRESHOLD = 0.55
# Duplicates collapse into one card, so fetch extra hits to still fill a page
OVERFETCH = 3
MAX_FETCH = 600

# ================= QDRANT =================
client = get_client()

# Restore snapshot if collection doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    if not QDRANT_PATH and os.path.exists(SNAPSHOT_DIR) and os.listdir(SNAPSHOT_DIR):
        latest_snapshot = sorted(os.listdir(SNAPSHOT_DIR))[-1]
        # The snapshot lives on this machine, not inside the Qdrant container,
        # so upload it rather than pointing Qdrant at a local path.
        with open(os.path.join(SNAPSHOT_DIR, latest_snapshot), "rb") as f:
            resp = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/upload?priority=snapshot",
                files={"snapshot": (latest_snapshot, f)}
            )
        resp.raise_for_status()
        print(f"[INFO] Restored {COLLECTION_NAME} from snapshot {latest_snapshot}")
    else:
        print("[WARN] No collection or snapshot found! Run databaseCreate.py first.")

# ================= MODEL =================
model = SentenceTransformer(EMBEDDING_MODEL)

# ================= FILTER OPTIONS =================
def load_facets():
    """Courses / years / semesters present in the collection, for the filter dropdowns."""
    courses, years, semesters = {}, set(), set()
    if not client.collection_exists(COLLECTION_NAME):
        return {"courses": [], "years": [], "semesters": []}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=1000, offset=offset, with_vectors=False,
            with_payload=["course_code", "course_name", "year", "semester"]
        )
        for p in points:
            code = p.payload.get("course_code")
            if code:
                courses.setdefault(code, p.payload.get("course_name") or "")
            if p.payload.get("year"):
                years.add(p.payload["year"])
            if p.payload.get("semester"):
                semesters.add(p.payload["semester"])
        if offset is None:
            break
    return {
        "courses": [{"code": c, "name": n} for c, n in sorted(courses.items())],
        "years": sorted(years, reverse=True),
        "semesters": sorted(semesters),
    }

FACETS = load_facets()

# ================= SEARCH FUNCTION =================
def build_filter(filters):
    must = []
    for field in ("course_code", "difficulty"):
        if filters.get(field):
            must.append(FieldCondition(key=field, match=MatchValue(value=str(filters[field]))))
    for field in ("year", "semester"):
        if filters.get(field):
            must.append(FieldCondition(key=field, match=MatchValue(value=int(filters[field]))))
    min_marks, max_marks = filters.get("min_marks"), filters.get("max_marks")
    if min_marks or max_marks:
        must.append(FieldCondition(key="marks", range=Range(
            gte=int(min_marks) if min_marks else None,
            lte=int(max_marks) if max_marks else None
        )))
    return Filter(must=must) if must else None


def question_key(payload):
    # Text only: renumbered courses ("CH322 / CHY270" vs "CHY270") repeat questions
    return re.sub(r"\W+", "", payload.get("question_text", "").lower())


def search(query, offset=0, filters=None):
    q_emb = model.encode(QUERY_PREFIX + query, convert_to_numpy=True, normalize_embeddings=True)

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb.tolist(),
        query_filter=build_filter(filters or {}),
        limit=min((offset + TOP_K) * OVERFETCH, MAX_FETCH),
        score_threshold=THRESHOLD,
        with_payload=True
    ).points

    # Collapse the same question asked in several papers into one result
    groups = {}
    for hit in hits:
        key = question_key(hit.payload)
        if key not in groups:
            payload = dict(hit.payload)
            payload["score"] = round(hit.score, 3)
            payload["difficulty_tag"] = (payload.get("difficulty") or "Unknown").capitalize()
            payload["occurrences"] = []
            groups[key] = payload
        group = groups[key]
        occurrence = {"year": hit.payload.get("year"), "file_link": hit.payload.get("file_link")}
        if occurrence not in group["occurrences"]:
            group["occurrences"].append(occurrence)

    results = list(groups.values())
    for r in results:
        r["occurrences"].sort(key=lambda o: o["year"] or 0)
    return results[offset: offset + TOP_K], len(results) > offset + TOP_K

# ================= FLASK ROUTES =================
@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"results": [], "has_more": False})
    try:
        offset = max(0, int(data.get("offset", 0)))
        results, has_more = search(query, offset=offset, filters=data.get("filters") or {})
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid offset or filter value"}), 400
    return jsonify({"results": results, "has_more": has_more})

@app.route("/facets")
def facets_route():
    return jsonify(FACETS)

@app.route("/")
def index():
    return render_template("index.html")  # frontend template

# ================= RUN =================
if __name__ == "__main__":
    # The reloader imports this file twice, which embedded Qdrant (QDRANT_PATH) can't handle
    app.run(debug=True, use_reloader=not QDRANT_PATH, port=int(os.environ.get("PORT", 5000)))
