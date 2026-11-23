"""
Collects every question-paper PDF link from the B.Tech community on Amrita
DSpace into pdf_links.jsonl.

The previous crawler only read the first page of each collection listing, so
it found 93 of the 228 exam bundles and nothing after ~2023. This version
pages through the community's full browse listing, and merges with the
existing pdf_links.jsonl so "processed" flags survive a re-crawl.

Usage:
    python scrapeLinks.py
"""

import json
import os
import re
import time
from html import unescape
from urllib.parse import unquote

import requests

BASE_URL = "http://dspace.amritanet.edu:8080"
COMMUNITY = "/xmlui/handle/123456789/150"       # B.Tech question papers
OUTPUT_LINKS_FILE = "pdf_links.jsonl"
PAGE_SIZE = 100
DELAY = 0.3                                      # be gentle with the library's server

session = requests.Session()

ITEM_LINK = re.compile(r'<div class="artifact-title">\s*<a href="(/xmlui/handle/[0-9/]+)">([^<]*)</a>')
PDF_LINK = re.compile(r'href="/xmlui/bitstream(/handle/[0-9]+/[0-9]+/[^"?]+?\.pdf)(?:\?[^"]*)?"', re.IGNORECASE)


def get(url):
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"  retry {url} ({e.__class__.__name__})")
            time.sleep(2 * (attempt + 1))


def list_items():
    """All exam bundles in the community, newest first."""
    items, offset = {}, 0
    while True:
        html = get(f"{BASE_URL}{COMMUNITY}/browse?type=dateissued&sort_by=2&order=DESC"
                   f"&rpp={PAGE_SIZE}&offset={offset}")
        page = ITEM_LINK.findall(html)
        if not page:
            break
        for handle, title in page:
            items.setdefault(handle, unescape(title).strip())
        offset += PAGE_SIZE
        time.sleep(DELAY)
    return items


def item_pdf_urls(handle):
    html = get(BASE_URL + handle)
    # Same URL shape as earlier crawls (unencoded file name, no ?sequence) so
    # links match the file_link values already in dspace_questions_metadata.jsonl
    paths = dict.fromkeys(unquote(unescape(p)) for p in PDF_LINK.findall(html))
    return [f"{BASE_URL}/xmlui/bitstream{p}" for p in paths]


def main():
    existing = {}
    if os.path.exists(OUTPUT_LINKS_FILE):
        with open(OUTPUT_LINKS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing[record["pdf_url"]] = record

    print("Listing exam bundles...")
    items = list_items()
    print(f"{len(items)} bundles\n")

    results, new = [], 0
    seen = set()
    for i, (handle, title) in enumerate(items.items(), start=1):
        urls = item_pdf_urls(handle)
        added = 0
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            record = existing.get(url) or {"item_title": title, "pdf_url": url, "processed": False}
            if url not in existing:
                added += 1
            results.append(record)
        new += added
        print(f"[{i}/{len(items)}] {title}: {len(urls)} PDFs ({added} new)")
        time.sleep(DELAY)

    # Keep links from earlier crawls that are no longer listed (moved/withdrawn)
    for url, record in existing.items():
        if url not in seen:
            results.append(record)

    with open(OUTPUT_LINKS_FILE, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[DONE] {len(results)} PDF links ({new} new) saved to {OUTPUT_LINKS_FILE}")


if __name__ == "__main__":
    main()
