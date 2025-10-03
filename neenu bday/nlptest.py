    git checkout <your-branch-name>import requests, io, os, re, json
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM

# ================= OLLAMA SETUP =================
print("🦙 Initializing Ollama...")

# Initialize Ollama LLM
llm = OllamaLLM(
    model="mistral:7b",  # Change this if you have a different model
    temperature=0,
    base_url="http://localhost:11434"
)

# Test Ollama connection
try:
    test_response = llm.invoke("Hello")
    print("✅ Ollama connection successful!")
except Exception as e:
    print(f"❌ Ollama connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Check if model is installed: ollama list")
    print("3. If no model, install one: ollama pull llama3.2:3b")
    exit(1)

# ================= CONFIG =================
LINKS_FILE = "C:\\Users\\niran\\Downloads\\neenu bday\\neenu bday\\pdf_links.jsonl"   
OUTPUT_FILE = "dspace_questions_metadata_new.jsonl"
PROGRESS_FILE = "progress.txt"

# Setup session for PDF downloads
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# ================== PROGRESS FUNCTIONS ==================
def get_progress():
    """Get the current progress (which PDF we're on)"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_progress(idx):
    """Save current progress"""
    try:
        with open(PROGRESS_FILE, "w", encoding='utf-8') as f:
            f.write(str(idx))
        print(f"💾 Progress saved: {idx}")
    except Exception as e:
        print(f"⚠️ Could not save progress: {e}")

# ================== PDF PROCESSING ==================
def extract_text_from_pdf_url(url):
    """Download PDF and extract text"""
    print(f"📥 Downloading: {url}")
    
    try:
        # Download PDF
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        
        # Extract text from PDF
        reader = PdfReader(io.BytesIO(resp.content))
        text = ""
        
        total_pages = len(reader.pages)
        print(f"📄 Processing {total_pages} pages...")
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Clean up text
                    page_text = re.sub(r'\s+', ' ', page_text.strip())
                    text += page_text + "\n"
                print(f"  ✅ Page {i+1}/{total_pages}")
            except Exception as e:
                print(f"  ⚠️ Page {i+1} failed: {e}")
        
        print(f"📝 Extracted {len(text):,} characters total")
        return text.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return ""
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return ""

def safe_json_loads(text):
    """Parse JSON from Ollama response with multiple strategies"""
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON array in text
    try:
        # Look for JSON array pattern
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract individual JSON objects
    try:
        # Find all JSON objects
        objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        results = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                if isinstance(obj, dict) and obj.get('question_text'):
                    results.append(obj)
            except:
                continue
        if results:
            return results
    except:
        pass
    
    # Strategy 4: Manual parsing for common Ollama patterns
    try:
        # Sometimes Ollama returns each object on a new line
        lines = text.split('\n')
        results = []
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    if obj.get('question_text'):
                        results.append(obj)
                except:
                    continue
        if results:
            return results
    except:
        pass
    
    # Final fallback
    print(f"⚠️ JSON parsing failed. Raw response: {text[:200]}...")
    return [{
        "course_code": "unknown",
        "course_name": "unknown", 
        "year": 0,
        "semester": 0,
        "question_text": f"Parse failed: {text[:100]}...",
        "marks": 0
    }]

def extract_questions_metadata_with_llm(pdf_text, file_link=""):
    """Extract questions using Ollama with your exact prompt"""
    
    if not pdf_text or len(pdf_text.strip()) < 100:
        print("⚠️ PDF text too short, skipping...")
        return "[]"
    
    # Limit text size for Ollama (adjust based on your model's context window)
    max_chars = 8000  # Conservative limit for llama3.2:3b
    
    if len(pdf_text) > max_chars:
        print(f"📏 Truncating text from {len(pdf_text)} to {max_chars} characters")
        # Try to cut at a sentence boundary
        truncated = pdf_text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # If we find a good cut point
            pdf_text = truncated[:last_period + 1]
        else:
            pdf_text = truncated
    
    # Your exact prompt
    prompt = f"""You are given a question paper text.
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
- Attach headings to each sub-question if needed.
- Skip incomplete or unclear questions (set question_text="None").
- Ignore picture/image-based questions.
- Output ONLY a JSON array.

{pdf_text}"""

    try:
        print("🤖 Processing with Ollama...")
        response = llm.invoke(prompt)
        print(f"📤 Ollama responded with {len(response)} characters")
        return response
        
    except Exception as e:
        print(f"❌ Ollama processing failed: {e}")
        return f'[{{"course_code": "error", "course_name": "error", "year": 0, "semester": 0, "question_text": "Ollama error: {str(e)}", "marks": 0}}]'

# ================== MAIN FUNCTION ==================
def main():
    """Main processing function"""
    print("🚀 PDF Question Extractor (Ollama Only)")
    print("=" * 50)
    
    # Check input file
    if not os.path.exists(LINKS_FILE):
        print(f"❌ {LINKS_FILE} not found!")
        print("\nCreate this file with your PDF URLs:")
        print('{"pdf_url": "https://example.com/paper.pdf"}')
        print('{"pdf_url": "https://example.com/paper2.pdf"}')
        return
    
    # Load PDF links
    print(f"📂 Loading PDF links...")
    try:
        with open(LINKS_FILE, "r", encoding="utf-8") as f:
            all_pdfs = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        pdf_data = json.loads(line)
                        if 'pdf_url' in pdf_data:
                            all_pdfs.append(pdf_data)
                        else:
                            print(f"⚠️ Line {line_num}: Missing 'pdf_url'")
                    except json.JSONDecodeError:
                        print(f"⚠️ Line {line_num}: Invalid JSON")
    except Exception as e:
        print(f"❌ Error reading {LINKS_FILE}: {e}")
        return
    
    if not all_pdfs:
        print("❌ No valid PDF URLs found!")
        return
    
    # Get progress
    start_idx = get_progress()
    print(f"📊 Found {len(all_pdfs)} PDFs")
    if start_idx > 0:
        print(f"📍 Resuming from PDF #{start_idx + 1}")
    
    # Process PDFs
    processed_count = 0
    total_questions = 0
    
    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
            for idx, pdf_data in enumerate(all_pdfs[start_idx:], start=start_idx):
                try:
                    print(f"\n{'='*50}")
                    print(f"📄 PDF {idx+1}/{len(all_pdfs)}")
                    
                    # Extract text from PDF
                    pdf_text = extract_text_from_pdf_url(pdf_data["pdf_url"])
                    
                    if not pdf_text:
                        print("❌ No text extracted, skipping...")
                        save_progress(idx + 1)
                        continue
                    
                    # Process with Ollama
                    questions_json_text = extract_questions_metadata_with_llm(pdf_text, pdf_data["pdf_url"])
                    
                    # Parse response
                    questions = safe_json_loads(questions_json_text)
                    
                    if not questions:
                        print("⚠️ No valid questions found")
                        save_progress(idx + 1)
                        continue
                    
                    # Save questions
                    saved_count = 0
                    for q in questions:
                        if (isinstance(q, dict) and 
                            q.get("question_text") and 
                            str(q["question_text"]).lower() not in ["none", "error", ""] and
                            len(str(q["question_text"]).strip()) > 15):
                            
                            # Add file link
                            q["file_link"] = pdf_data["pdf_url"]
                            
                            # Ensure all fields exist
                            q.setdefault("course_code", "")
                            q.setdefault("course_name", "")
                            q.setdefault("year", 0)
                            q.setdefault("semester", 0)
                            q.setdefault("marks", 0)
                            
                            # Save to file
                            f_out.write(json.dumps(q, ensure_ascii=False) + "\n")
                            saved_count += 1
                    
                    print(f"✅ Saved {saved_count} questions")
                    total_questions += saved_count
                    processed_count += 1
                    
                    # Save progress
                    save_progress(idx + 1)
                    f_out.flush()  # Write immediately
                    
                except KeyboardInterrupt:
                    print("\n⏹️ Stopped by user")
                    break
                except Exception as e:
                    print(f"❌ Error processing PDF {idx+1}: {e}")
                    break
                    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    
    # Summary
    print(f"\n🏁 COMPLETE!")
    print(f"📊 Processed: {processed_count} PDFs")
    print(f"📝 Questions: {total_questions}")
    print(f"💾 Output: {OUTPUT_FILE}")

# ================== RUN ==================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Bye!")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        input("Press Enter to exit...")  # For Windows