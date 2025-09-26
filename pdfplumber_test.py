# import requests
# import io
# import os
# import re
# import json
# import pdfplumber
# import base64
# from io import BytesIO
# from langchain_ollama import ChatOllama
# from langchain.schema import HumanMessage

# # ================= CONFIG =================
# LINKS_FILE = "pdf_links1.jsonl"         # input file with {"item_title": "...", "pdf_url": "..."}
# ARCHIVE_DIR = "pdf_archives"           # stores raw extracted JSON with base64
# OUTPUT_FILE = "dspace_questions_metadata1.jsonl"  # structured output from llama
# PROGRESS_FILE = "progress1.txt"

# os.makedirs(ARCHIVE_DIR, exist_ok=True)

# llm = ChatOllama(model="llama3:8b", temperature=0)
# session = requests.Session()

# # ================= PROGRESS =================
# def get_progress():
#     return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

# def save_progress(idx):
#     with open(PROGRESS_FILE, "w") as f:
#         f.write(str(idx))

# # ================= PDF EXTRACTION =================
# def extract_pdf_with_base64(resp_bytes, pdf_name):
#     """Extract text + tables + base64 images and save to archive JSON."""
#     pdf_data = []
#     with pdfplumber.open(io.BytesIO(resp_bytes)) as pdf:
#         for page_number, page in enumerate(pdf.pages, start=1):
#             page_dict = {"page": page_number, "text": "", "tables": [], "images": []}

#             # Text
#             text = page.extract_text()
#             page_dict["text"] = text if text else ""

#             # Tables
#             for table in page.extract_tables():
#                 page_dict["tables"].append(table)

#             # Images (base64)
#             for img_index, img in enumerate(page.images):
#                 try:
#                     im = page.within_bbox((img['x0'], img['top'], img['x1'], img['bottom'])).to_image(resolution=150).original
#                     buffered = BytesIO()
#                     im.save(buffered, format="PNG")
#                     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#                     page_dict["images"].append({
#                         "image_index": img_index,
#                         "bbox": [img['x0'], img['top'], img['x1'], img['bottom']],
#                         "base64": img_str
#                     })
#                 except Exception as e:
#                     print(f"  [WARN] Failed to extract image {img_index} on page {page_number}: {e}")

#             pdf_data.append(page_dict)

#     # Save archive file (with base64 intact)
#     archive_path = os.path.join(ARCHIVE_DIR, f"{pdf_name}.json")
#     with open(archive_path, "w", encoding="utf-8") as f:
#         json.dump(pdf_data, f, indent=2, ensure_ascii=False)

#     return pdf_data

# # ================= STRIP BASE64 =================
# def get_text_only(pdf_json):
#     """Flatten only the text content from PDF JSON (ignore base64 + tables)."""
#     text_chunks = []
#     for page in pdf_json:
#         if page.get("text"):
#             text_chunks.append(page["text"])
#     return "\n".join(text_chunks).strip()

# # ================= JSON SAFETY =================
# def safe_json_loads(text):
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         match = re.search(r'\[.*\]', text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group(0))
#             except:
#                 return [{"raw_text": text}]
#         return [{"raw_text": text}]

# # ================= LLM EXTRACTION =================
# def extract_questions_metadata_with_llm(pdf_json, file_link):
#     pdf_text = get_text_only(pdf_json)

#     prompt = f"""
# You are given a question paper text.
# Split it into individual questions and extract metadata for each question
# in JSON format:
# {{
#   "course_code": "...",
#   "course_name": "...",
#   "year": ...,
#   "semester": ...,
#   "question_text": "...",
#   "marks": ...
# }}

# Rules:
# - Do NOT output headings as separate objects.
# - "question_text" must contain the full question as one clean string.
# - Extract marks if present (like [5] -> 5). If missing, set null.
# - Semester and Year must be integers.
# - Skip incomplete or unclear questions (set "question_text": "None").
# - Ignore image-based questions.
# - Output ONLY a JSON array.
# """
#     response = llm([HumanMessage(content=prompt + "\n\n" + pdf_text)])
#     return response.content.strip()

# # ================= MAIN =================
# def main():
#     print("Loading all PDF links from file...")
#     with open(LINKS_FILE, "r", encoding="utf-8") as f:
#         all_pdfs = [json.loads(line) for line in f]

#     start_idx = get_progress()
#     print(f"Resuming from PDF #{start_idx}/{len(all_pdfs)}")

#     with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
#         for idx, pdf in enumerate(all_pdfs[start_idx:], start=start_idx):
#             try:
#                 print(f"\nProcessing {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")

#                 # Download
#                 resp = session.get(pdf["pdf_url"])
#                 resp.raise_for_status()

#                 # Extract + Save archive JSON with base64
#                 pdf_name = f"doc_{idx+1}"
#                 pdf_json = extract_pdf_with_base64(resp.content, pdf_name)

#             #     # Run LLM on stripped text
#             #     questions_json_text = extract_questions_metadata_with_llm(pdf_json, pdf["pdf_url"])
#             #     questions = safe_json_loads(questions_json_text)

#             #     # Save structured JSONL
#             #     for q in questions:
#             #         if not q.get("question_text") or q["question_text"].lower() == "none":
#             #             continue
#             #         q["file_link"] = pdf["pdf_url"]
#             #         q["item_title"] = pdf.get("item_title", "")
#             #         f_out.write(json.dumps(q, ensure_ascii=False) + "\n")

#                 save_progress(idx + 1)
#             except Exception as e:
#                 print(f"[ERROR]: {e}")
#                 print("Will retry this PDF next run.")
#                 break

#     print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
#     print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()


# import requests
# import io
# import os
# import re
# import json
# import pdfplumber
# import base64
# from io import BytesIO
# from langchain_ollama import ChatOllama
# from langchain.schema import HumanMessage

# import pytesseract
# from PIL import Image

# # ================= CONFIG =================
# LINKS_FILE = "pdf_links1.jsonl"         # input file with {"item_title": "...", "pdf_url": "..."}
# ARCHIVE_DIR = "pdf_archives"           # stores raw extracted JSON with base64
# OUTPUT_FILE = "dspace_questions_metadata1.jsonl"  # structured output from llama
# PROGRESS_FILE = "progress1.txt"

# os.makedirs(ARCHIVE_DIR, exist_ok=True)

# llm = ChatOllama(model="llama3:8b", temperature=0)
# session = requests.Session()

# # ================= PROGRESS =================
# def get_progress():
#     return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

# def save_progress(idx):
#     with open(PROGRESS_FILE, "w") as f:
#         f.write(str(idx))

# # ================= OCR FIX =================
# def ocr_pdf_page(page):
#     """Convert a pdfplumber page to image and run Tesseract OCR fully in memory."""
#     img = page.to_image(resolution=300).original  # PIL.Image
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)
#     text = pytesseract.image_to_string(Image.open(buf))
#     return text

# # ================= PDF EXTRACTION =================
# def extract_pdf_with_base64(resp_bytes, pdf_name):
#     """Extract text + tables + base64 images and save to archive JSON."""
#     pdf_data = []
#     with pdfplumber.open(io.BytesIO(resp_bytes)) as pdf:
#         for page_number, page in enumerate(pdf.pages, start=1):
#             page_dict = {"page": page_number, "text": "", "tables": [], "images": []}

#             # Text with fallback
#             text = page.extract_text()
#             if not text or "cid:" in text:
#                 print(f"  [INFO] Falling back to OCR on page {page_number}")
#                 try:
#                     text = ocr_pdf_page(page)
#                 except Exception as e:
#                     print(f"  [WARN] OCR failed on page {page_number}: {e}")
#                     text = ""
#             page_dict["text"] = text if text else ""

#             # Tables
#             for table in page.extract_tables():
#                 page_dict["tables"].append(table)

#             # Images (base64)
#             for img_index, img in enumerate(page.images):
#                 try:
#                     im = page.within_bbox((img['x0'], img['top'], img['x1'], img['bottom'])).to_image(resolution=150).original
#                     buffered = BytesIO()
#                     im.save(buffered, format="PNG")
#                     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#                     page_dict["images"].append({
#                         "image_index": img_index,
#                         "bbox": [img['x0'], img['top'], img['x1'], img['bottom']],
#                         "base64": img_str
#                     })
#                 except Exception as e:
#                     print(f"  [WARN] Failed to extract image {img_index} on page {page_number}: {e}")

#             pdf_data.append(page_dict)

#     # Save archive file (with base64 intact)
#     archive_path = os.path.join(ARCHIVE_DIR, f"{pdf_name}.json")
#     with open(archive_path, "w", encoding="utf-8") as f:
#         json.dump(pdf_data, f, indent=2, ensure_ascii=False)

#     return pdf_data

# # ================= STRIP BASE64 =================
# def get_text_only(pdf_json):
#     """Flatten only the text content from PDF JSON (ignore base64 + tables)."""
#     text_chunks = []
#     for page in pdf_json:
#         if page.get("text"):
#             text_chunks.append(page["text"])
#     return "\n".join(text_chunks).strip()

# # ================= JSON SAFETY =================
# def safe_json_loads(text):
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         match = re.search(r'\[.*\]', text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group(0))
#             except:
#                 return [{"raw_text": text}]
#         return [{"raw_text": text}]

# # ================= LLM EXTRACTION =================
# # ================= LLM EXTRACTION =================
# def extract_questions_metadata_with_llm(pdf_json, file_link):
#     pdf_text = get_text_only(pdf_json)

#     prompt = f"""
# You are given a question paper text.
# Split it into individual questions and extract metadata for each question
# in JSON format:
# {{
#   "course_code": "...",
#   "course_name": "...",
#   "year": ...,
#   "semester": ...,
#   "question_text": "...",
#   "marks": ...
# }}

# Rules:
# - Do NOT output headings as separate objects.
# - "question_text" must contain the full question as one clean string.
# - Extract marks if present (like [5] -> 5). If missing, set null.
# - Semester and Year must be integers.
# - Skip incomplete or unclear questions (set "question_text": "None").
# - Ignore image-based questions.
# - Output ONLY a JSON array.
# """
#     # Use invoke instead of deprecated __call__
#     response = llm.invoke([HumanMessage(content=prompt + "\n\n" + pdf_text)])
#     raw_output = response.content.strip()
#     print("\n[DEBUG] LLM raw output preview:\n", raw_output[:1000])  # first 1000 chars
#     return raw_output


# # ================= MAIN =================
# def main():
#     print("Loading all PDF links from file...")
#     with open(LINKS_FILE, "r", encoding="utf-8") as f:
#         all_pdfs = [json.loads(line) for line in f]

#     start_idx = get_progress()
#     print(f"Resuming from PDF #{start_idx}/{len(all_pdfs)}")

#     with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
#         for idx, pdf in enumerate(all_pdfs[start_idx:], start=start_idx):
#             try:
#                 print(f"\nProcessing {idx+1}/{len(all_pdfs)}: {pdf['pdf_url']}")

#                 # Download
#                 resp = session.get(pdf["pdf_url"])
#                 resp.raise_for_status()

#                 # Extract + Save archive JSON with base64
#                 pdf_name = f"doc_{idx+1}"
#                 pdf_json = extract_pdf_with_base64(resp.content, pdf_name)

#                 # LLM extraction
#                 questions_json_text = extract_questions_metadata_with_llm(pdf_json, pdf["pdf_url"])
#                 questions = safe_json_loads(questions_json_text)
#                 print(f"[DEBUG] Parsed {len(questions)} entries from LLM output")

#                 # Save structured JSONL
#                 written = 0
#                 for q in questions:
#                     if not q.get("question_text") or q["question_text"].lower() == "none":
#                         print("[DEBUG] Skipping invalid question entry:", q)
#                         continue
#                     q["file_link"] = pdf["pdf_url"]
#                     q["item_title"] = pdf.get("item_title", "")
#                     f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
#                     written += 1

#                 print(f"[INFO] Wrote {written} questions to JSONL")

#                 save_progress(idx + 1)
#             except Exception as e:
#                 print(f"[ERROR]: {e}")
#                 print("Will retry this PDF next run.")
#                 break

#     print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
#     print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()

import requests
import io
import os
import re
import json
import pdfplumber
import base64
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

import pytesseract
from PIL import Image

# ================= CONFIG =================
LINKS_FILE = "pdf_links1.jsonl"         # input file with {"item_title": "...", "pdf_url": "..."}
ARCHIVE_DIR = "pdf_archives"           # stores raw extracted JSON with base64
OUTPUT_FILE = "dspace_questions_metadata1.jsonl"  # structured output from LLM
PROGRESS_FILE = "progress1.txt"

os.makedirs(ARCHIVE_DIR, exist_ok=True)

llm = ChatOllama(model="llama3:8b", temperature=0)
session = requests.Session()

# ================= PROGRESS =================
def get_progress():
    return int(open(PROGRESS_FILE).read().strip()) if os.path.exists(PROGRESS_FILE) else 0

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ================= OCR =================
def ocr_pdf_page(page):
    """Convert a pdfplumber page to image and run Tesseract OCR fully in memory."""
    img = page.to_image(resolution=300).original  # PIL.Image
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    text = pytesseract.image_to_string(Image.open(buf))
    return text

# ================= PDF EXTRACTION =================
def extract_pdf_with_base64(resp_bytes, pdf_name):
    """Extract text + tables + base64 images and save to archive JSON."""
    pdf_data = []
    with pdfplumber.open(io.BytesIO(resp_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_dict = {"page": page_number, "text": "", "tables": [], "images": []}

            # Text with fallback
            text = page.extract_text()
            if not text or "cid:" in text:
                print(f"  [INFO] Falling back to OCR on page {page_number}")
                try:
                    text = ocr_pdf_page(page)
                except Exception as e:
                    print(f"  [WARN] OCR failed on page {page_number}: {e}")
                    text = ""
            page_dict["text"] = text if text else ""

            # Tables
            for table in page.extract_tables():
                page_dict["tables"].append(table)

            # Images (base64)
            for img_index, img in enumerate(page.images):
                try:
                    im = page.within_bbox((img['x0'], img['top'], img['x1'], img['bottom'])).to_image(resolution=150).original
                    buffered = BytesIO()
                    im.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    page_dict["images"].append({
                        "image_index": img_index,
                        "bbox": [img['x0'], img['top'], img['x1'], img['bottom']],
                        "base64": img_str
                    })
                except Exception as e:
                    print(f"  [WARN] Failed to extract image {img_index} on page {page_number}: {e}")

            pdf_data.append(page_dict)

    # Save archive file (with base64 intact)
    archive_path = os.path.join(ARCHIVE_DIR, f"{pdf_name}.json")
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=2, ensure_ascii=False)

    return pdf_data

# ================= TEXT EXTRACTION =================
def get_text_only(pdf_json):
    """Flatten only the text content from PDF JSON (ignore base64 + tables)."""
    text_chunks = []
    for page in pdf_json:
        if page.get("text"):
            text_chunks.append(page["text"])
    return "\n".join(text_chunks).strip()

# ================= JSON SAFETY =================
def safe_json_loads(text):
    """
    Extracts a JSON array from LLM output safely.
    Falls back to raw_text if parsing fails.
    """
    try:
        # Try normal JSON load first
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract only the JSON array part
        match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
        if match:
            json_text = match.group(0)
            # Replace newlines outside strings
            json_text = re.sub(r'\n+', ' ', json_text)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return [{"raw_text": text}]
        return [{"raw_text": text}]

# ================= LLM EXTRACTION =================
def extract_questions_metadata_with_llm(pdf_json, file_link):
    pdf_text = get_text_only(pdf_json)

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
- Do NOT output headings as separate objects.
- "question_text" must contain the full question as one clean string.
- Extract marks if present (like [5] -> 5). If missing, set null.
- Semester and Year must be integers.
- Skip incomplete or unclear questions (set "question_text": "None").
- Ignore image-based questions.
- Output ONLY a JSON array.
"""
    # Use invoke to call LLM
    response = llm.invoke([HumanMessage(content=prompt + "\n\n" + pdf_text)])
    raw_output = response.content.strip()
    print("\n[DEBUG] LLM raw output preview:\n", raw_output[:1000])  # first 1000 chars
    return raw_output

# ================= MAIN =================
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

                # Download
                resp = session.get(pdf["pdf_url"])
                resp.raise_for_status()

                # Extract + Save archive JSON with base64
                pdf_name = f"doc_{idx+1}"
                pdf_json = extract_pdf_with_base64(resp.content, pdf_name)

                # LLM extraction
                questions_json_text = extract_questions_metadata_with_llm(pdf_json, pdf["pdf_url"])
                questions = safe_json_loads(questions_json_text)
                print(f"[DEBUG] Parsed {len(questions)} entries from LLM output")

                # Save structured JSONL
                written = 0
                for q in questions:
                    if not q.get("question_text") or q["question_text"].lower() == "none":
                        print("[DEBUG] Skipping invalid question entry:", q)
                        continue
                    q["file_link"] = pdf["pdf_url"]
                    q["item_title"] = pdf.get("item_title", "")
                    f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                    written += 1

                print(f"[INFO] Wrote {written} questions to JSONL")
                save_progress(idx + 1)

            except Exception as e:
                print(f"[ERROR]: {e}")
                print("Will retry this PDF next run.")
                break

    print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
    print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
