"""
search_eval.py -- measures search quality on a fixed set of student-style
queries, so changes to search can be compared with numbers.

Three variants are compared:
    dense    vector search only (what the app did before)
    hybrid   vector + BM25 keyword search, merged with reciprocal rank fusion
    rerank   hybrid top 30, re-ordered by a cross-encoder (ms-marco-MiniLM-L-6-v2)

Workflow
    python search_eval.py pool      # writes search_eval_pool.json: unjudged top-5 results
    (judge: fill search_eval_qrels.json with {query: {question_key: 0|1}})
    python search_eval.py score     # precision@5, success@1, MRR per variant

Run with QDRANT_PATH=qdrant_local if Qdrant isn't running as a server.
"""

import json
import re
import statistics
import sys
import time

from sentence_transformers import CrossEncoder, SentenceTransformer

from common import COLLECTION_NAME, EMBEDDING_MODEL, QUERY_PREFIX, get_client
from hybrid import BM25Index, reciprocal_rank_fusion

QUERIES_FILE = "search_eval_queries.json"
QRELS_FILE = "search_eval_qrels.json"
POOL_FILE = "search_eval_pool.json"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small + fast; matters under exam load
K = 5
CANDIDATES = 50
RERANK_DEPTH = 30


def question_key(text):
    return re.sub(r"\W+", "", (text or "").lower())[:200]


def dedup(ids, payloads):
    seen, out = set(), []
    for i in ids:
        key = question_key(payloads[i]["question_text"])
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    client = get_client()
    payloads, offset = {}, None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=2000, offset=offset, with_vectors=False)
        payloads.update({p.id: p.payload for p in points})
        if offset is None:
            break
    bm25 = BM25Index([(i, f"{p.get('course_name', '')} {p['question_text']}") for i, p in payloads.items()])
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    timings = {"dense": [], "hybrid": [], "rerank": []}
    rankings = {}
    for q in queries:
        text = q["query"]
        t0 = time.perf_counter()
        vec = embedder.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
        dense = [p.id for p in client.query_points(COLLECTION_NAME, query=vec, limit=CANDIDATES).points]
        t1 = time.perf_counter()
        hybrid = reciprocal_rank_fusion(dense, [i for i, _ in bm25.search(text, CANDIDATES)])
        t2 = time.perf_counter()
        pool = dedup(hybrid, payloads)[:RERANK_DEPTH]
        scores = reranker.predict([(text, payloads[i]["question_text"]) for i in pool])
        rerank = [i for _, i in sorted(zip(scores, pool), key=lambda s: -s[0])]
        t3 = time.perf_counter()
        timings["dense"].append(t1 - t0)
        timings["hybrid"].append(t2 - t0)
        timings["rerank"].append(t3 - t0)
        rankings[text] = {name: dedup(r, payloads)[:K] for name, r in
                          (("dense", dense), ("hybrid", hybrid), ("rerank", rerank))}

    try:
        with open(QRELS_FILE) as f:
            qrels = json.load(f)
    except FileNotFoundError:
        qrels = {}

    if mode == "pool":
        pool_out = {}
        for q in queries:
            judged = qrels.get(q["query"], {})
            items = {}
            for ids in rankings[q["query"]].values():
                for i in ids:
                    key = question_key(payloads[i]["question_text"])
                    if key not in judged:
                        items[key] = f"[{payloads[i].get('course_name')}] {payloads[i]['question_text'][:220]}"
            pool_out[q["query"]] = {"intent": q["intent"], "unjudged": items}
        with open(POOL_FILE, "w") as f:
            json.dump(pool_out, f, indent=1, ensure_ascii=False)
        print(f"{sum(len(v['unjudged']) for v in pool_out.values())} results to judge -> {POOL_FILE}")
        return

    print(f"{len(queries)} queries, top {K}, judged results only\n")
    print(f"{'variant':<8} {'P@5':>6} {'success@1':>10} {'MRR':>6} {'median ms':>10}")
    for name in ("dense", "hybrid", "rerank"):
        p_at_k, success, rr = [], [], []
        for q in queries:
            judged = qrels.get(q["query"], {})
            rels = [judged.get(question_key(payloads[i]["question_text"])) for i in rankings[q["query"]][name]]
            if any(r is None for r in rels):
                print(f"  [{name}] {q['query']!r}: {rels.count(None)} unjudged results; run `pool` and judge them")
                rels = [r or 0 for r in rels]
            p_at_k.append(sum(rels) / K)
            success.append(1.0 if rels and rels[0] else 0.0)
            rr.append(next((1 / (n + 1) for n, r in enumerate(rels) if r), 0.0))
        print(f"{name:<8} {statistics.mean(p_at_k):6.2f} {statistics.mean(success):10.2f} "
              f"{statistics.mean(rr):6.2f} {statistics.median(timings[name]) * 1000:10.0f}")


if __name__ == "__main__":
    main()
