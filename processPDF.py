import requests, os, re, json, shutil, subprocess
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# ================= CONFIG =================
LINKS_FILE = "pdf_links.jsonl"
OUTPUT_FILE = "dspace_questions_metadata_new1.jsonl"
PROGRESS_FILE = "progress1.txt"
TEMP_FOLDER = "temp_pdfs"

# Initialize Ollama LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ================== PROGRESS ==================
def get_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            value = int(open(PROGRESS_FILE, "r").read().strip())
            return value
        except ValueError:
            return 0  # file exists but empty or invalid
    return 0  # file doesn't exist


def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(idx))

# ================== MINERU EXTRACTION ==================
def extract_structured_text_with_mineru(pdf_path):
    """
    Runs MinerU CLI on a local PDF and returns a list of text blocks
    """
    output_json = pdf_path.replace(".pdf", "_mineru.json")
    subprocess.run(["mineru", "-p", pdf_path, "--output", output_json], check=True)

    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_blocks = [block["text"] for block in data.get("content", []) if "text" in block]
    # Clean up MinerU JSON
    os.remove(output_json)
    return text_blocks

# ================== JSON CLEANING ==================
def safe_json_loads(text):
    text = re.sub(r"```(json)?", "", text).strip()
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    fixed_text = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(fixed_text)
    except:
        pass
    return [{"raw_text": text}]

# ================== LLM CALL ==================
def extract_questions_metadata_with_llm(blocks, file_link=""):
    all_questions = []
    for block in blocks:
        prompt = f"""
You are given a block of structured exam paper text.
Extract all valid questions in JSON format:
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
- Attach headings to each sub-question if needed.
- Skip incomplete or unclear questions (set question_text="None").
- Ignore images.
- Output ONLY a JSON array.
Block Text:
{block}
"""
        response = llm([HumanMessage(content=prompt)])
        questions = safe_json_loads(response.content)
        for q in questions:
            if not q.get("question_text") or q["question_text"].lower() == "none":
                continue
            q["file_link"] = file_link
            all_questions.append(q)
    return all_questions

# ================== MAIN PIPELINE ==================
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

                # Download PDF locally
                pdf_resp = requests.get(pdf["pdf_url"])
                local_pdf = os.path.join(TEMP_FOLDER, f"temp_{idx}.pdf")
                with open(local_pdf, "wb") as wf:
                    wf.write(pdf_resp.content)

                # MinerU extraction
                text_blocks = extract_structured_text_with_mineru(local_pdf)

                # LLM extraction
                questions = extract_questions_metadata_with_llm(text_blocks, pdf["pdf_url"])

                # Save output
                for q in questions:
                    f_out.write(json.dumps(q, ensure_ascii=False) + "\n")

                # Delete temp PDF
                os.remove(local_pdf)

                save_progress(idx + 1)

            except Exception as e:
                print(f"[ERROR]: {e}")
                print("Will retry this PDF next run.")
                break

    print(f"\n[SAVED]: Progress saved at {get_progress()}/{len(all_pdfs)}")
    print(f"[APPENDED]: Data appended to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
