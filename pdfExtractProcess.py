import requests
import io
import os
import re
import json
import pdfplumber
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# ================= CONFIG =================
LINKS_FILE = "pdf_links.jsonl"         # input file with {"item_title": "...", "pdf_url": "..."}
ARCHIVE_DIR = "pdf_archives"            # temporary archive
OUTPUT_FILE = "dspace_questions_metadata.jsonl"
PROGRESS_FILE = "progress.txt"

os.makedirs(ARCHIVE_DIR, exist_ok=True)

# LLaMA LLM
llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
session = requests.Session()

# ================= PROGRESS =================
def get_progress():
    return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ================= OCR =================
def ocr_pdf_page(page):
    img = page.to_image(resolution=300).original
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return pytesseract.image_to_string(Image.open(buf))

# ================= PDF EXTRACTION =================
def extract_pdf_with_base64(resp_bytes, pdf_name):
    pdf_data = []
    with pdfplumber.open(io.BytesIO(resp_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_dict = {"page": page_number, "text": "", "tables": [], "images": []}

            # Text extraction
            text = page.extract_text()
            if not text or "cid:" in text:
                try:
                    text = ocr_pdf_page(page)
                except Exception:
                    text = ""
            page_dict["text"] = text if text else ""

            # Tables
            for table in page.extract_tables():
                page_dict["tables"].append(table)

            # Images
            for img_index, img in enumerate(page.images):
                try:
                    im = page.within_bbox((img['x0'], img['top'], img['x1'], img['bottom'])).to_image(resolution=150).original
                    buf = BytesIO()
                    im.save(buf, format="PNG")
                    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    page_dict["images"].append({
                        "image_index": img_index,
                        "bbox": [img['x0'], img['top'], img['x1'], img['bottom']],
                        "base64": img_str
                    })
                except Exception:
                    continue

            pdf_data.append(page_dict)

    # Save temporary archive
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

# ================= MAIN =================
def main():
    print("Loading all PDF links...")
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        all_pdfs = [json.loads(line) for line in f]

    start_idx = get_progress()
    print(f"Resuming from PDF #{start_idx}/{len(all_pdfs)}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for idx, pdf in enumerate(all_pdfs[start_idx:], start=start_idx):
            try:
                print(f"\nProcessing {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")
                resp = session.get(pdf["pdf_url"])
                resp.raise_for_status()

                pdf_name = f"doc_{idx+1}"
                pdf_pages, archive_path = extract_pdf_with_base64(resp.content, pdf_name)

                # ---------- Extract course metadata from first page ----------
                course_metadata = None
                for page_json in pdf_pages:
                    if page_json.get("text", "").strip():
                        prompt = f"""
                        Extract the following metadata from this question paper text in JSON format:
                        {{
                          "course_code": "...",
                          "course_name": "...",
                          "year": ...,
                          "semester": ...
                        }}
                        Only output JSON.
                        """
                        response = llm.invoke([HumanMessage(content=prompt + "\n\n" + page_json["text"])])
                        course_metadata = safe_json_loads(response.content.strip())
                        break

                if not course_metadata:
                    print("[WARN] Could not extract course metadata. Skipping PDF.")
                    save_progress(idx + 1)
                    continue

                # ---------- Extract questions per page ----------
                total_written = 0
                for page_json in pdf_pages:
                    page_text = page_json.get("text", "").strip()
                    if not page_text:
                        continue

                    prompt_questions = f"""
                    You are given a question paper text.
                    Split it into individual questions and extract metadata for each question
                    in JSON format:
                    {{
                      "questions": [
                        {{
                          "question_text": "...",
                          "marks": ...,
                          "images": ""  # if a diagram/figure is referenced, leave as empty string; else null
                        }}
                      ]
                    }}

                    Rules:
                    - Do NOT output headings as separate objects.
                    - Attach headings to each sub-question if needed.
                    - "marks" must be integer if present, otherwise null.
                    - "question_text" must be a clean string. If incomplete/unclear, set to "None".
                    - Output ONLY a JSON array inside "questions", no explanations.
                    """
                    response = llm.invoke([HumanMessage(content=prompt_questions + "\n\n" + page_text)])
                    page_questions = safe_json_loads(response.content.strip())

                    if not page_questions or "questions" not in page_questions:
                        continue

                    page_images = page_json.get("images", [])

                    for q in page_questions["questions"]:
                        if not q.get("question_text") or q["question_text"].lower() == "none":
                            continue

                        # Attach images if flagged by LLM
                        if q.get("images") == "" or q.get("images") is None:
                            q["images"] = page_images.copy()
                        else:
                            q["images"] = []

                        # Add course-level metadata to each question
                        q["course_code"] = course_metadata.get("course_code")
                        q["course_name"] = course_metadata.get("course_name")
                        q["year"] = int(course_metadata.get("year")) if course_metadata.get("year") else None
                        q["semester"] = course_metadata.get("semester") if course_metadata.get("semester") else None
                        q["file_link"] = pdf["pdf_url"]
                        q["item_title"] = pdf.get("item_title", "")

                        f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                        total_written += 1

                print(f"[INFO] Wrote {total_written} questions from PDF {idx+1}")
                save_progress(idx + 1)

                # Delete temp archive
                os.remove(archive_path)

            except Exception as e:
                print(f"[ERROR]: {e}")
                print("Will retry this PDF next run.")
                break

    print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
    print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
