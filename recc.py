import random
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import json

# ================= CONFIG =================
COLLECTION_NAME = "questions"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
NUM_QUESTIONS = 10  # questions from DB
NUM_LLM_QUESTIONS = 5  # number of extra generated questions

# GPT-OSS120B API Configuration
GPT_API_URL = "http://localhost:8000/v1/chat/completions"  # Adjust to your API endpoint
# If using a different API, update the URL accordingly

# ================= QDRANT =================
client = QdrantClient(host="localhost", port=6333)
model_embed = SentenceTransformer(EMBEDDING_MODEL)

# ================= FUNCTION TO GENERATE PAPER =================
def generate_question_paper(subject: str):
    # --- Step 1: Semantic search ---
    query_vec = model_embed.encode(subject, normalize_embeddings=True).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=NUM_QUESTIONS*2,
        with_payload=True
    )
    if not results:
        print("[WARN] No questions found for this subject!")
        return []

    selected = random.sample(results, min(NUM_QUESTIONS, len(results)))
    original_questions = [r.payload["question_text"] for r in selected]

    # --- Step 2: Use GPT-OSS120B API to generate new questions ---
    llm_prompt = (
        "You are a question generator for exams. "
        "Given the following questions, generate exactly 5 new, similar questions "
        "on the same subject, keeping the difficulty similar. "
        "Output only the questions, numbered 1-5:\n\n"
        + "\n".join(f"- {q}" for q in original_questions)
        + "\n\nNew questions:"
    )

    # Call GPT-OSS120B API
    try:
        response = requests.post(
            GPT_API_URL,
            json={
                "model": "gpt-oss120b",
                "messages": [
                    {"role": "system", "content": "You are a helpful exam question generator."},
                    {"role": "user", "content": llm_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            print(f"[ERROR] API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            generated_text = ""
            
    except requests.exceptions.Timeout:
        print("[ERROR] API request timed out")
        generated_text = ""
    except Exception as e:
        print(f"[ERROR] Failed to call API: {e}")
        generated_text = ""

    # Parse generated questions (remove numbering if present)
    generated_questions = []
    if generated_text:
        lines = generated_text.split("\n")
        for line in lines:
            line = line.strip()
            # Remove common prefixes like "1.", "Q1:", etc.
            if line and len(line) > 5:
                # Remove leading numbers and punctuation
                cleaned = line.lstrip("0123456789.)-Q: ")
                if cleaned:
                    generated_questions.append(cleaned)

    # --- Step 3: Combine original + generated ---
    final_paper = original_questions + generated_questions[:NUM_LLM_QUESTIONS]

    # Shuffle for variety
    random.shuffle(final_paper)

    # Format with numbering
    formatted_paper = [f"{i+1}. {q}" for i, q in enumerate(final_paper)]
    return formatted_paper

# ================= RUN EXAMPLE =================
if __name__ == "__main__":
    subject = input("Enter subject: ")
    paper = generate_question_paper(subject)

    print("\nGenerated Question Paper:")
    for q in paper:
        print(q)