from flask import Flask, request, jsonify, render_template
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import hashlib
import json
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

# ================= RESULT CACHE =================
# Exam-season queries repeat a lot and the index only changes once a semester,
# so search results are cached. Redis (REDIS_URL) is shared by every worker
# and server; without it each worker keeps a small in-memory LRU cache.
CACHE_TTL = int(os.environ.get("CACHE_TTL", 6 * 3600))
LOCAL_CACHE_SIZE = 2000


class ResultCache:
    def __init__(self, redis_url=None):
        self.redis = None
        self.local = OrderedDict()
        if redis_url:
            import redis
            self.redis = redis.Redis.from_url(redis_url, socket_timeout=0.2)

    def get(self, key):
        if self.redis:
            try:
                value = self.redis.get(key)
                return json.loads(value) if value else None
            except redis_errors():
                return None             # cache down: just compute the result
        if key in self.local:
            self.local.move_to_end(key)
            return self.local[key]
        return None

    def set(self, key, value):
        if self.redis:
            try:
                self.redis.setex(key, CACHE_TTL, json.dumps(value))
            except redis_errors():
                pass
            return
        self.local[key] = value
        if len(self.local) > LOCAL_CACHE_SIZE:
            self.local.popitem(last=False)


def redis_errors():
    import redis
    return (redis.exceptions.RedisError, OSError)


cache = ResultCache(os.environ.get("REDIS_URL"))

# ================= PDF LINKS =================
# With PDF_BASE_URL=/dspace, PDF links go through nginx's cache
# (deploy/nginx.conf) instead of straight to the library's DSpace server.
DSPACE_BASE = "http://dspace.amritanet.edu:8080"
PDF_BASE_URL = os.environ.get("PDF_BASE_URL", "").rstrip("/")


def pdf_link(url):
    if PDF_BASE_URL and url and url.startswith(DSPACE_BASE):
        return PDF_BASE_URL + url[len(DSPACE_BASE):]
    return url

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

    query_args = dict(
        collection_name=COLLECTION_NAME,
        query=q_emb.tolist(),
        query_filter=build_filter(filters or {}),
        limit=min((offset + TOP_K) * OVERFETCH, MAX_FETCH),
        score_threshold=THRESHOLD,
        with_payload=True
    )
    try:
        hits = client.query_points(**query_args).points
    except ResponseHandlingException:
        # Qdrant closes idle keep-alive connections; the first request on a
        # stale one fails with "Server disconnected". One retry opens a new one.
        hits = client.query_points(**query_args).points

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
        group["file_link"] = pdf_link(group.get("file_link"))
        occurrence = {"year": hit.payload.get("year"), "file_link": pdf_link(hit.payload.get("file_link"))}
        if occurrence not in group["occurrences"]:
            group["occurrences"].append(occurrence)

    results = list(groups.values())
    for r in results:
        r["occurrences"].sort(key=lambda o: o["year"] or 0)
    return results[offset: offset + TOP_K], len(results) > offset + TOP_K

# ================= PAPER CATALOG =================
# Every paper in DSpace by course (built by catalog.py), so students can find
# papers for courses whose questions haven't been extracted yet.
CATALOG_FILE = "papers_catalog.jsonl"
MAX_COURSES = 30


def load_catalog():
    courses = {}
    if not os.path.exists(CATALOG_FILE):
        print(f"[WARN] {CATALOG_FILE} not found; run catalog.py for the paper browser.")
        return []
    with open(CATALOG_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            code = row.get("course_code")
            if not code or "error" in row:
                continue
            course = courses.setdefault(code, {"code": code, "name": row.get("course_name") or "", "papers": []})
            paper = {k: row.get(k) for k in ("year", "exam", "semester")}
            paper["pdf_url"] = pdf_link(row.get("pdf_url"))
            course["papers"].append(paper)
    for course in courses.values():
        course["papers"].sort(key=lambda p: (-(p["year"] or 0), p["exam"] or ""))
        course["search_key"] = f"{course['code']} {course['name']}".lower()
        course["code_key"] = re.sub(r"\W", "", course["code"].lower())
    return list(courses.values())


CATALOG = load_catalog()


def find_courses(query):
    """Courses whose code or name contains every word of the query, best matches first."""
    q = query.lower().strip()
    q_code = re.sub(r"\W", "", q)
    words = q.split()
    ranked = []
    for course in CATALOG:
        if q_code and q_code in course["code_key"]:
            rank = 0 if course["code_key"].endswith(q_code) else 1      # "cse435" / "19cse435"
        elif all(w in course["search_key"] for w in words):
            rank = 2 if course["name"].lower().startswith(q) else 3
        else:
            continue
        ranked.append((rank, -len(course["papers"]), course))
    ranked.sort(key=lambda r: r[:2])
    return [{k: c[k] for k in ("code", "name", "papers")} for _, _, c in ranked[:MAX_COURSES]]

# ================= FLASK ROUTES =================
@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"results": [], "has_more": False})
    filters = data.get("filters") or {}
    if not isinstance(filters, dict):
        return jsonify({"error": "Invalid filters"}), 400
    try:
        offset = max(0, int(data.get("offset", 0)))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid offset"}), 400

    key = "search:" + hashlib.sha1(json.dumps(
        [" ".join(query.lower().split()), {k: v for k, v in sorted(filters.items()) if v}, offset]
    ).encode()).hexdigest()
    body = cache.get(key)
    hit = body is not None
    if not hit:
        try:
            results, has_more = search(query, offset=offset, filters=filters)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid filter value"}), 400
        body = {"results": results, "has_more": has_more}
        cache.set(key, body)
    response = jsonify(body)
    response.headers["X-Cache"] = "HIT" if hit else "MISS"
    return response

@app.route("/healthz")
def healthz():
    """For the load balancer: is this instance able to answer searches?"""
    try:
        ok = client.collection_exists(COLLECTION_NAME)
    except Exception:
        ok = False
    return (jsonify({"status": "ok"}), 200) if ok else (jsonify({"status": "qdrant unavailable"}), 503)

@app.route("/facets")
def facets_route():
    return jsonify(FACETS)

@app.route("/papers")
def papers_route():
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"courses": []})
    return jsonify({"courses": find_courses(query)})

@app.route("/")
def index():
    return render_template("index.html")  # frontend template

# ================= RUN =================
# Local development only. In production run it under gunicorn (see Dockerfile);
# never expose the Flask debugger, which can execute code.
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    # The reloader imports this file twice, which embedded Qdrant (QDRANT_PATH) can't handle
    app.run(debug=debug, use_reloader=debug and not QDRANT_PATH, port=int(os.environ.get("PORT", 5000)))
