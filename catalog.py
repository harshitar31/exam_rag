"""
Builds papers_catalog.jsonl: one row per question paper with course code,
course name, exam, year and semester, read from the paper's header. This lets
the website list every paper for a course right away, before (slow) question
extraction has reached it.

PDFs are cached in pdf_cache/ so later processing doesn't download them again.

Usage:
    python catalog.py            # resumes; only new links are fetched
"""

import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import unquote

import pdfplumber
import requests

from cleaning import normalize_course_code, normalize_semester

LINKS_FILE = "pdf_links.jsonl"
CATALOG_FILE = "papers_catalog.jsonl"
CACHE_DIR = "pdf_cache"
WORKERS = 6                 # parallel processes; at most 6 downloads at a time from the library server
HEADER_LINES = 25

# optional trailing letter only if it isn't the start of a word ("15ENG111Communicative")
CODE = r"(?:\d{2}\s?)?[A-Z]{2,4}\s?\d{3}(?:[A-Z](?![a-z]))?"
# "19CSE435 Computer Vision", "CH329 / CHY273 Polymer Chemistry", "15ENG111Communicative English"
COURSE_LINE = re.compile(rf"^\s*({CODE}(?:\s*[/&,]\s*{CODE})*)\s*[-–:.]?\s*([A-Za-z(].{{2,150}})$")
FILENAME_CODE = re.compile(rf"^({CODE})")
NOT_A_COURSE = re.compile(r"^(CO\s?\d|BTL|Page\b|Roll\b|Time\b|Max)", re.IGNORECASE)

session = requests.Session()


def cache_path(url):
    handle, filename = re.search(r"/handle/\d+/(\d+)/(.+)$", url).groups()
    return os.path.join(CACHE_DIR, handle, unquote(filename))


def fetch_pdf(url):
    path = cache_path(url)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), path
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=90)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(resp.content)
    return resp.content, path


def parse_header(text):
    """(course_code, course_name) from the first lines of page 1."""
    for line in text.splitlines()[:HEADER_LINES]:
        line = line.strip()
        if NOT_A_COURSE.match(line):
            continue
        m = COURSE_LINE.match(line)
        if m:
            name = re.sub(r"\s*\((?:common|for)\b.*$", "", m.group(2), flags=re.IGNORECASE)
            name = re.sub(r"\s*[-–]\s*(?:end|mid)\b.*$", "", name, flags=re.IGNORECASE).strip(" -–:.")
            if name.isupper():
                name = name.title()
            return normalize_course_code(m.group(1)), name
    return None, None


def exam_type(text):
    t = text.lower().replace("_", " ")
    if "supp" in t:
        return "Supplementary"
    if "mid" in t:
        return "Mid-Term"
    if re.search(r"\b(ii|2nd|second)\s*ass|\bass\w*\s*(ii|2)\b", t):
        return "Second Assessment"
    if re.search(r"\bass", t):
        return "First Assessment"
    return "End Semester"


def catalog_entry(link):
    url, title = link["pdf_url"], link.get("item_title", "")
    filename = unquote(url.rsplit("/", 1)[-1])
    row = {"pdf_url": url, "item_title": title, "file": filename}
    try:
        data, path = fetch_pdf(url)
        row["cache_path"] = path
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            header = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
    except Exception as e:
        row["error"] = str(e)[:200]
        header = ""

    code, name = parse_header(header)
    file_code = FILENAME_CODE.match(filename.replace(" ", ""))
    year = (re.search(r"\b(19[89]\d|20[0-4]\d)\b", f"{title} {filename}")
            or re.search(r"\b(20[0-4]\d)\b", header[:600]))
    row.update({
        "course_code": code or (normalize_course_code(file_code.group(1)) if file_code else None),
        "course_name": name,
        "exam": exam_type(f"{title} {filename}"),
        "year": int(year.group(1)) if year else None,
        "semester": normalize_semester(title),
        "has_text": bool(header.strip()),
    })
    return row


def fill_missing_names(rows):
    """Papers whose header couldn't be parsed get the most common name for their code."""
    names = defaultdict(Counter)
    for r in rows:
        if r.get("course_code") and r.get("course_name"):
            names[r["course_code"]][r["course_name"]] += 1
    for r in rows:
        if r.get("course_code") and not r.get("course_name") and names[r["course_code"]]:
            r["course_name"] = names[r["course_code"]].most_common(1)[0][0]
    return rows


def main():
    with open(LINKS_FILE, encoding="utf-8") as f:
        links = [json.loads(line) for line in f if line.strip()]

    done = {}
    if os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "error" not in row:
                    done[row["pdf_url"]] = row
    todo = [l for l in links if l["pdf_url"] not in done]
    print(f"{len(links)} links, {len(done)} already catalogued, {len(todo)} to fetch")

    # Rewrite without failed rows, then append as we go so an interrupted run resumes
    rows = list(done.values())
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Processes, not threads: header parsing is CPU-bound
    with ProcessPoolExecutor(WORKERS) as pool, open(CATALOG_FILE, "a", encoding="utf-8") as out:
        for i, row in enumerate(pool.map(catalog_entry, todo, chunksize=8), start=1):
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 200 == 0 or i == len(todo):
                out.flush()
                print(f"  {i}/{len(todo)}", flush=True)

    rows = fill_missing_names(rows)
    rows.sort(key=lambda r: (r.get("course_code") or "~", -(r.get("year") or 0), r["exam"]))
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = Counter("error" if "error" in r else "no code" if not r.get("course_code")
                    else "no name" if not r.get("course_name") else "ok" for r in rows)
    print(f"[DONE] {len(rows)} papers in {CATALOG_FILE}: {dict(stats)}; "
          f"{len({r['course_code'] for r in rows if r.get('course_code')})} courses")


if __name__ == "__main__":
    main()
