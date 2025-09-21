import requests, io, os, re, json
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_ollama import ChatOllama   # modern import
from langchain.schema import HumanMessage

# ================= CONFIG =================
LINKS_FILE = "pdf_links.jsonl"   
OUTPUT_FILE = "dspace_questions_metadata_new1.jsonl"
PROGRESS_FILE = "progress1.txt"

llm = ChatOllama(model="llama3:8b", temperature=0)
session = requests.Session()

# ================== PROGRESS ==================
def get_progress():
    return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ================== PDF ==================
def extract_text_from_pdf_url(url):
    print(f"Downloading PDF: {url}")
    resp = session.get(url)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        print(f"  Extracted text from page {i+1}")
    return text.strip()

def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return [{"raw_text": text}]
        return [{"raw_text": text}]

def extract_questions_metadata_with_llm(pdf_text, file_link=""):
    prompt = f"""
You are given a question paper text.
Split it into individual questions and extract metadata for each question
in JSON format:
{{
  "course_code": "...",
  "course_name": "...",
  "year": ...,
  "semester": ...,
  "question_text": "...",
  "marks": ...
}}

Rules:
- Do NOT output headings as separate objects (e.g. "Fill in the blanks" should not be its own question).
- Make sure the entire question text is present in "question_text."
- Attach headings to each sub-question if needed.
- Semester and Marks should be in integer
- Skip incomplete or unclear questions (set question_text="None").
- Ignore picture/image-based questions.
- Output ONLY a JSON array.
"""
    response = llm([HumanMessage(content=prompt + "\n\n" + pdf_text)])
    return response.content.strip()

# ================== MAIN ==================
def main():
    print("Loading all PDF links from file...")
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        all_pdfs = [json.loads(line) for line in f]

    start_idx = get_progress()
    print(f"Resuming from PDF #{start_idx}/{len(all_pdfs)}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for idx, pdf in enumerate(all_pdfs[start_idx:], start=start_idx):
            try:
                print(f"\nProcessing {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")
                pdf_text = extract_text_from_pdf_url(pdf["pdf_url"])
                questions_json_text = extract_questions_metadata_with_llm(pdf_text, pdf["pdf_url"])
                questions = safe_json_loads(questions_json_text)

                for q in questions:
                    if not q.get("question_text") or q["question_text"].lower() == "none":
                        continue
                    q["file_link"] = pdf["pdf_url"]
                    f_out.write(json.dumps(q, ensure_ascii=False) + "\n")

                save_progress(idx + 1)
            except Exception as e:
                print(f"[ERROR]: {e}")
                print("Will retry this PDF next run.")
                break

    print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
    print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
