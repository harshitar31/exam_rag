import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "http://dspace.amritanet.edu:8080"
START_URL = "http://dspace.amritanet.edu:8080/xmlui/handle/123456789/150"
OUTPUT_LINKS_FILE = "pdf_links.jsonl"

session = requests.Session()

def crawl_collection(url, visited=None):
    """Recursively crawl collections and collect PDF links."""
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)

    pdf_links = []
    print(f"Fetching collection/page: {url}")
    resp = session.get(url)
    if resp.status_code != 200:
        print(f"Failed to load: {url}")
        return pdf_links

    soup = BeautifulSoup(resp.text, "html.parser")

    for div in soup.find_all("div", class_="artifact-title"):
        a = div.find("a", href=True)
        if not a:
            continue

        link_url = BASE_URL + a["href"]
        title = a.text.strip()

        sub_resp = session.get(link_url)
        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")

        sub_titles = sub_soup.find_all("div", class_="artifact-title")
        if sub_titles and any("handle" in (t.find("a")["href"] if t.find("a") else "") for t in sub_titles):
            # Recurse into sub-collection
            pdf_links.extend(crawl_collection(link_url, visited))
        else:
            # Actual item page
            for span in sub_soup.find_all("span", title=True):
                if span["title"].lower().endswith(".pdf"):
                    pdf_url = link_url.replace("/handle/", "/bitstream/handle/") + "/" + span["title"]
                    pdf_links.append({"item_title": title, "pdf_url": pdf_url})
                    print(f"Found PDF: {pdf_url}")

    return pdf_links


def main():
    print("Starting crawl of DSpace repository...")
    all_pdfs = crawl_collection(START_URL)
    print(f"Total PDFs collected: {len(all_pdfs)}")

    with open(OUTPUT_LINKS_FILE, "w", encoding="utf-8") as f:
        for pdf in all_pdfs:
            f.write(json.dumps(pdf, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved all links to {OUTPUT_LINKS_FILE}")


if __name__ == '__main__':
    main()
