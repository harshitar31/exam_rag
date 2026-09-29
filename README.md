# EXAM RAG
 
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/) [![LangChain](https://img.shields.io/badge/LangChain-purple)](https://www.langchain.com/) [![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)](https://qdrant.tech/)

---

### **Project Description** 

**Exam RAG** is an **AI-powered system** that automates finding, organizing, and searching past exam papers from the Institutional Repository. It's a two-stage process: first, it crawls a website to find and download PDF files into memory, then it uses an AI to extract and structure the questions from those PDFs. These structured questions are stored in a **Qdrant vector database** along with their **vector embeddings**, enabling **semantic search** with filters for course, year, semester, difficulty and marks.

---

### **Features** 

* **Web Crawler:** Finds and collects all PDF links from a specified repository (e.g., Amrita DSpace).
* **PDF Miner:** Extracts raw text from PDF documents on the fly.
* **AI Question Splitter:** Uses **LangChain** to convert raw text into a clean, structured JSON format, with each line representing a single question.
* **Modular:** The project is divided into separate scripts for crawling and processing for a clean workflow.
* **Semantic Search:** Allows for searching questions based on meaning and context, not just keywords, by leveraging vector embeddings from the database.
* **Filters & de-duplication:** Narrow results by course, year, semester, difficulty and marks; a question asked in several years shows up once with "Asked 3× (2012, 2014, 2016)".
* **Difficulty tags:** Rule-based easy / medium / difficult tags from marks and Bloom's-taxonomy task verbs.

---

### **Tech Stack & Dependencies** 

### **Core Libraries** 

-   **`qdrant-client`**: Vector database for questions, embeddings and filterable metadata.
-   **`json`**: For handling JSON data (`.jsonl` files).
-   **`os`**: For file and directory management.
-   **`numpy`**: For numerical operations on vector embeddings.

### **Web & PDF Processing** 

-   **`requests`**: For downloading web pages and PDFs.
-   **`BeautifulSoup` (`bs4`)**: For parsing HTML and extracting PDF links.
-   **`pdfplumber`**: For extracting text and tables from PDF files.
-   **`io`**: For handling in-memory data streams.

### **AI & Semantic Search** 

-   **`langchain`**: To interact with the model.
-   **`sentence_transformers`**: For generating text embeddings.
-   **`BAAI/bge-base-en-v1.5`**: Embedding model (queries use BGE's retrieval instruction prefix).

### **Web Interface** 

-   **`flask`**: For building a simple web application to demonstrate the search functionality.
---

### **Running it**

```bash
pip install -r requirements.txt
docker compose up -d                       # Qdrant on :6333

python cleaning.py                         # dspace_questions_metadata.jsonl -> dspace_questions_clean.jsonl
python difficultyTagging.py                # -> dspace_questions_tagged.jsonl
python databaseCreate.py --rebuild         # embed + index into Qdrant
python app.py                              # http://localhost:5000
```

No Docker? Prefix the last two commands with `QDRANT_PATH=qdrant_local` to run Qdrant embedded in-process. On macOS, port 5000 is often taken by AirPlay Receiver; use `PORT=5050 python app.py`.

### **Pipeline**

| Step | Script | Output |
|---|---|---|
| Crawl DSpace for PDF links | `scrapeLinks.py` | `pdf_links.jsonl` |
| LLM question + metadata extraction | `processPDF.py` | `dspace_questions_metadata.jsonl` |
| Normalise semester/year/course code, strip numbering | `cleaning.py` | `dspace_questions_clean.jsonl` |
| Difficulty tagging | `difficultyTagging.py` | `dspace_questions_tagged.jsonl` |
| Embed + index | `databaseCreate.py` | Qdrant collection `questions` |
| Search UI | `app.py` | |

### **Fine-tuned extractor**

A LoRA fine-tune of Qwen2.5-3B-Instruct replaces the cloud LLM for extraction.

```bash
python fix_training_data.py                # data_v2/ -> data_v3/ (grounded targets, int semesters, valid.jsonl)
bash finetune.sh                           # MLX LoRA on Apple Silicon
python evaluate.py                         # score on hand-labelled sample_pdfs/
python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit   # base model, for comparison
```

`build_data_for_qs.py` rebuilds the training set from scratch (needs access to the DSpace server).
`evaluate.py` reports metadata accuracy, question recall, and the share of extracted questions that are actually in the PDF (hallucination check), with held-out papers reported separately from papers seen in training.

### **Implementation** 
![Image 1](implementation/cc.png) ![Image 2](implementation/eee.png)
![Image 3](implementation/eng.png) ![Image 4](implementation/qm.png)

