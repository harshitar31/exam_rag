import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "http://dspace.amritanet.edu:8080"
START_URL = "http://dspace.amritanet.edu:8080/xmlui/handle/123456789/150"
OUTPUT_LINKS_FILE = "pdf_links.jsonl"

session = requests.Session()

def crawl_collection(url, visited, seen_pdfs, results):
    """Recursively crawl collections and collect unique PDF links."""
    if url in visited:
        return
    visited.add(url)

    print(f"Fetching collection/page: {url}")
    resp = session.get(url)
    if resp.status_code != 200:
        print(f"Failed to load: {url}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    for div in soup.find_all("div", class_="artifact-title"):
        a = div.find("a", href=True)
        if not a:
            continue

        link_url = BASE_URL + a["href"]
        title = a.text.strip()

        sub_resp = session.get(link_url)
        if sub_resp.status_code != 200:
            continue
        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")

        sub_titles = sub_soup.find_all("div", class_="artifact-title")
        if sub_titles and any("handle" in (t.find("a")["href"] if t.find("a") else "") for t in sub_titles):
            # Recurse into sub-collection
            crawl_collection(link_url, visited, seen_pdfs, results)
        else:
            # Item page → extract PDFs
            for span in sub_soup.find_all("span", title=True):
                if span["title"].lower().endswith(".pdf"):
                    pdf_url = link_url.replace("/handle/", "/bitstream/handle/") + "/" + span["title"]

                    if pdf_url not in seen_pdfs:
                        seen_pdfs.add(pdf_url)
                        results.append({
                            "item_title": title,
                            "pdf_url": pdf_url,
                            "processed": False
                        })
                        print(f"Found PDF: {pdf_url}")


def main():
    print("Starting crawl of DSpace repository...")
    visited = set()
    seen_pdfs = set()
    results = []

    crawl_collection(START_URL, visited, seen_pdfs, results)

    print(f"Total unique PDFs collected: {len(results)}")

    with open(OUTPUT_LINKS_FILE, "w", encoding="utf-8") as f:
        for pdf in results:
            f.write(json.dumps(pdf, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved all links to {OUTPUT_LINKS_FILE}")


if __name__ == '__main__':
    main()

