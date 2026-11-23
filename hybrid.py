"""
Keyword (BM25) search over question text, and Reciprocal Rank Fusion to merge
it with the vector search. Embeddings match meaning but blur exact terms
("Pilling-Bedworth", "Huffman", "CHY270"); BM25 matches exact terms but not
paraphrases. Fusing the two ranked lists gets both.
"""

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set("""a an and are as at be by for from how in is it of on or the to what which why with
explain describe discuss write give define state find determine calculate show briefly""".split())


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of (doc_id, text)."""
        self.k1, self.b = k1, b
        self.ids = []
        self.lengths = []
        self.postings = defaultdict(list)        # term -> [(doc_index, term_freq)]
        for idx, (doc_id, text) in enumerate(docs):
            tokens = tokenize(text)
            self.ids.append(doc_id)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))
        self.n = len(self.ids)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query, limit=50):
        """[(doc_id, score)], best first."""
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avg_len)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        return [(self.ids[idx], score) for idx, score in best]


def reciprocal_rank_fusion(*rankings, k=60):
    """Merge ranked lists of ids: each list contributes 1 / (k + rank)."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] += 1 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
