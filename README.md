# EXAM RAG
 
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/) [![LangChain](https://img.shields.io/badge/LangChain-purple)](https://www.langchain.com/) [![SQLite](https://img.shields.io/badge/SQLite-07405E?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

---

### **Project Description** 

**Exam RAG** is an **AI-powered system** that automates finding, organizing, and searching past exam papers from the Institutional Repository. It's a two-stage process: first, it crawls a website to find and download PDF files into memory, then it uses an AI to extract and structure the questions from those PDFs. These structured questions are stored in an **SQLite database** along with their **vector embeddings**, enabling **semantic search**.

---

### **Features** 

* **Web Crawler:** Finds and collects all PDF links from a specified repository (e.g., Amrita DSpace).
* **PDF Miner:** Extracts raw text from PDF documents on the fly.
* **AI Question Splitter:** Uses **LangChain** to convert raw text into a clean, structured JSON format, with each line representing a single question.
* **Modular:** The project is divided into separate scripts for crawling and processing for a clean workflow.
* **Semantic Search:** Allows for searching questions based on meaning and context, not just keywords, by leveraging vector embeddings from the database.

---

### **Tech Stack & Dependencies** 

### **Core Libraries** 

-   **`sqlite3`**: For database operations (storing questions and embeddings).
-   **`json`**: For handling JSON data (`.jsonl` files).
-   **`os`**: For file and directory management.
-   **`numpy`**: For numerical operations on vector embeddings.

### **Web & PDF Processing** 

-   **`requests`**: For downloading web pages and PDFs.
-   **`BeautifulSoup` (`bs4`)**: For parsing HTML and extracting PDF links.
-   **`PyPDF2`**: For extracting text from PDF files.
-   **`io`**: For handling in-memory data streams.

### **AI & Semantic Search** 

-   **`langchain`**: To interact with the model.
-   **`sentence_transformers`**: For generating text embeddings.
-   **`sklearn.metrics.pairwise.cosine_similarity`**: To calculate the similarity between embeddings for semantic search.

### **Web Interface** 

-   **`flask`**: For building a simple web application to demonstrate the search functionality.
---

