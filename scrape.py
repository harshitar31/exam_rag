import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import io
import json
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage


# ================= CONFIG =================
BASE_URL = "http://dspace.amritanet.edu:8080"
START_URL = "http://dspace.amritanet.edu:8080/xmlui/handle/123456789/150" 
OUTPUT_FILE = "dspace_questions_metadata3.jsonl"
PROGRESS_FILE = "progress.txt"

# Initialize Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

session = requests.Session()

# ================== PROGRESS TRACKING ==================
def get_progress():
    """Read last processed PDF index from file (default 0)."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip() or 0)
    return 0

def save_progress(index):
    """Write last processed PDF index to file."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))


# ================== CRAWLER ==================
PDF_LIMIT = 200  

def crawl_collection(url, visited=None, pdf_counter=None):
    """Recursively crawl collections and collect PDF links, stopping after PDF_LIMIT."""
    if visited is None:
        visited = set()
    if pdf_counter is None:
        pdf_counter = {"count": 0}

    if url in visited:
        print(f"Skipping already visited: {url}")
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
        if pdf_counter["count"] >= PDF_LIMIT:
            print(f"\nReached PDF limit ({PDF_LIMIT}). Stopping crawl.")
            return pdf_links

        a = div.find("a", href=True)
        if not a:
            continue

        link_url = BASE_URL + a["href"]
        title = a.text.strip()

        print(f"Visiting: {title} -> {link_url}")
        sub_resp = session.get(link_url)
        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")

        sub_titles = sub_soup.find_all("div", class_="artifact-title")
        if sub_titles and any("handle" in (t.find("a")["href"] if t.find("a") else "") for t in sub_titles):
            print(f"Found sub-collection: {title}")
            pdf_links.extend(crawl_collection(link_url, visited, pdf_counter))
            if pdf_counter["count"] >= PDF_LIMIT:
                return pdf_links
        else:
            print(f"Found item: {title}")
            for span in sub_soup.find_all("span", title=True):
                if span["title"].lower().endswith(".pdf"):
                    pdf_url = link_url.replace("/handle/", "/bitstream/handle/") + "/" + span["title"]
                    print(f"  Found PDF: {pdf_url}")
                    pdf_links.append({"item_title": title, "pdf_url": pdf_url})
                    pdf_counter["count"] += 1
                    if pdf_counter["count"] >= PDF_LIMIT:
                        print(f"\nReached PDF limit ({PDF_LIMIT}). Stopping crawl.")
                        return pdf_links

    return pdf_links


# ================== PDF & TEXT ==================
def extract_text_from_pdf_url(url):
    """Fetch PDF from URL and extract text in memory."""
    print(f"Downloading PDF in memory: {url}")
    resp = session.get(url)
    reader = PdfReader(io.BytesIO(resp.content))
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        print(f"  Extracted text from page {i+1}")
    return text.strip()


def safe_json_loads(text):
    """Try to extract JSON array from LLM response, even if extra text exists."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return [{"raw_text": text}]
        return [{"raw_text": text}]


# ================== METADATA WITH LLM ==================
def extract_questions_metadata_with_llm(pdf_text, file_link=""):
    """Send PDF text to Gemini 1.5 Flash and get JSON per question."""
    print(f"Sending text to LLM for: {file_link}")
    prompt = f"""
You are given a question paper text.
Split it into individual questions and extract metadata for each question
in the following JSON structure:

{{
  "course_code": "...",
  "course_name": "...",
  "year": ...,
  "semester": ...,
  "question_text": "..."
}}

Rules:
- Only include questions that are **standalone**.
- Do not include just headings like Fill in the blanks a separate object without the subdivisions. Add heading to each sub division and make each of them a separate object!!
- DO NOT MAKE SEPARATE OBJECTS FOR HEADINGS LIKE THIS "Fill in the blanks using the correct tense forms of the verbs given in brackets" SEPARATELY.
- If a question is incomplete, dependent on context, or unclear, set "question_text": "None".
- Ignore picture questions.
- Return ONLY a JSON array. Do not include any extra text.
- Each valid question should be one object in the array. 

Question paper text:
{pdf_text}
"""
    response = llm([HumanMessage(content=prompt)])
    print(f"LLM response received for: {file_link}")
    return response.content.strip()


# ================== MAIN ==================
def main():
    print("Starting crawl of DSpace repository...")
    all_pdfs = crawl_collection(START_URL)
    print(f"Total PDFs found: {len(all_pdfs)}")

    start_index = get_progress()
    if start_index >= len(all_pdfs):
        print("[DONE]: All PDFs already processed.")
        return

    print(f"Resuming from PDF #{start_index}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for idx, pdf in enumerate(all_pdfs[start_index:], start=start_index):
            print(f"\nProcessing PDF {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")
            try:
                pdf_text = extract_text_from_pdf_url(pdf["pdf_url"])
                questions_json_text = extract_questions_metadata_with_llm(pdf_text, file_link=pdf["pdf_url"])
                questions = safe_json_loads(questions_json_text)

                for q in questions:
                    if not q.get("question_text") or q["question_text"].strip().lower() == "none":
                        print("[SKIP]: Skipping non-standalone question")
                        continue
                    q["file_link"] = pdf["pdf_url"]
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")

                # Save progress only after full success
                save_progress(idx + 1)

            except Exception as e:
                print(f"[ERROR]: Error processing {pdf['pdf_url']}: {e}")
                print("Will retry this PDF next run.")
                break

    print(f"\n[SAVED]: Progress saved: {get_progress()}/{len(all_pdfs)} PDFs processed.")
    print(f"[APPENDED]: Appended questions metadata to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
