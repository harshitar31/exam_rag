import requests
import io
import os
import re
import json
import pdfplumber
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# ================= CONFIG =================
LINKS_FILE = "pdf_links.jsonl"         # input file with {"item_title": "...", "pdf_url": "...", "processed": false}
ARCHIVE_DIR = "pdf_archives"           # temporary archive
OUTPUT_FILE = "dspace_questions_metadata.jsonl"
PROGRESS_FILE = "progress.txt"

os.makedirs(ARCHIVE_DIR, exist_ok=True)

llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
session = requests.Session()

# ================= PROGRESS =================
def get_progress():
    return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ================= PDF EXTRACTION =================
def extract_pdf_text(resp_bytes, pdf_name):
    pdf_data = []
    with pdfplumber.open(io.BytesIO(resp_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_dict = {"page": page_number, "text": "", "tables": []}
            text = page.extract_text()
            page_dict["text"] = text if text else ""
            for table in page.extract_tables():
                page_dict["tables"].append(table)
            pdf_data.append(page_dict)
    archive_path = os.path.join(ARCHIVE_DIR, f"{pdf_name}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=2, ensure_ascii=False)
    return pdf_data, archive_path

# ================= JSON SAFETY =================
def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_text = match.group(0)
            json_text = re.sub(r'\n+', ' ', json_text)
            try:
                return json.loads(json_text)
            except:
                return {"raw_text": text}
        return {"raw_text": text}

# ================= UPDATE PROCESSED FLAG =================
def mark_as_processed(file_path, pdf_url):
    updated_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["pdf_url"] == pdf_url:
                record["processed"] = True
            updated_lines.append(json.dumps(record, ensure_ascii=False))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

# ================= MAIN =================
def main():
    print("Loading all PDF links...")
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        all_pdfs = [json.loads(line) for line in f]

    start_idx = get_progress()
    print(f"Resuming from PDF #{start_idx}/{len(all_pdfs)}")

    for idx, pdf in enumerate(all_pdfs[start_idx:], start=start_idx):
        if pdf.get("processed", False):
            print(f"Skipping already processed: {pdf['pdf_url']}")
            save_progress(idx + 1)
            continue

        try:
            print(f"\nProcessing {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")
            resp = session.get(pdf["pdf_url"])
            resp.raise_for_status()

            pdf_name = f"doc_{idx+1}"
            pdf_pages, archive_path = extract_pdf_text(resp.content, pdf_name)

            # ---------- Extract course metadata from first page ----------
            course_metadata = None
            for page_json in pdf_pages:
                if page_json.get("text", "").strip():
                    prompt = """
                    Extract the following metadata from this question paper text in JSON format:
                    {
                      "course_code": "...",
                      "course_name": "...",
                      "year": ...,
                      "semester": ...
                    }
                    Year and semester should be integers.
                    Only output JSON.
                    """
                    response = llm.invoke([HumanMessage(content=prompt + "\n\n" + page_json["text"])]).content
                    course_metadata = safe_json_loads(response.strip())
                    break

            if not course_metadata:
                print("[WARN] Could not extract course metadata. Skipping PDF.")
                save_progress(idx + 1)
                continue

            # ---------- Extract questions page by page ----------
            all_questions = []
            for page_json in pdf_pages:
                page_text = page_json.get("text", "").strip()
                if not page_text:
                    continue

                prompt_questions = f"""
                You are given a question paper text from a single page.
                Split it into individual questions and extract metadata for each question
                in JSON format:
                {{
                  "questions": [
                    {{
                      "question_text": "...",
                      "marks": ...
                    }}
                  ]
                }}

                Rules:
                - Do NOT output headings as separate objects.
                - Attach headings to each sub-question if needed.
                - "marks" must be integer if present, otherwise null.
                - "question_text" must be a clean string. If incomplete/unclear, skip question.
                - Do NOT include question numbers, part labels and marks in "question_text".
                - Output ONLY a JSON object with a "questions" array, no explanations.
                """

                try:
                    response = llm.invoke([HumanMessage(content=prompt_questions + "\n\n" + page_text)]).content
                    parsed = safe_json_loads(response.strip())
                    page_questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
                    all_questions.extend(page_questions)
                except Exception as e:
                    print(f"[WARN] Failed to extract questions from page {page_json['page']}: {e}")
                    continue

            # ---------- Combine into hierarchical JSON ----------
            output_doc = {
                "course_code": course_metadata.get("course_code"),
                "course_name": course_metadata.get("course_name"),
                "year": int(course_metadata.get("year")) if course_metadata.get("year") else None,
                "semester": course_metadata.get("semester"),
                "file_link": pdf["pdf_url"],
                "item_title": pdf.get("item_title", ""),
                "questions": all_questions
            }

            # ----------- WRITE OUTPUT IMMEDIATELY -----------
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(output_doc, ensure_ascii=False) + "\n")

            print(f"[INFO] Wrote {len(all_questions)} questions from PDF {idx+1}")

            # Mark processed + save progress immediately
            mark_as_processed(LINKS_FILE, pdf["pdf_url"])
            save_progress(idx + 1)

            # Cleanup
            os.remove(archive_path)

        except Exception as e:
            print(f"[ERROR]: {e}")
            print("Will retry this PDF next run.")
            break

    print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
    print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
