# Biodiversity RAG Chatbot App

This repository implements a Retrieval-Augmented Generation (RAG) pipeline using open-source embedding models and Pinecone for semantic search, enabling users to query biodiversity and conservation-related knowledge with natural language.

---

## 🧠 Features

-  **Retrieval-Augmented Generation (RAG)** using Pinecone and embedding models
-  Multiple **chunking strategies**: `none`, `sentence`, `fixed`, `semantic`
-  Multilingual support using `BAAI/bge-m3`
-  Query-response generation using **Groq LLM**
-  Frontend-ready with Streamlit

---

## 📁 Repository Structure

├── main.py 

├── app.py 

├── api.py 

├── chunking.py 

├── clean_text.py 

├── upload_chunks.py 

├── query.py 

├── requirements.txt

└── README.md


---

## 🚀 Getting Started

### 1. Set Up Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
PINECONE_API_KEY=your_key
PINECONE_ENV=your_env
GROQ_API_KEY=your_key
```

### 4. Running the Query Code

#### Option A: Streamlit Frontend

```bash
streamlit run app.py
```

#### Option B: From Python (Terminal Or Jupyter Notebook)

```bash
python query.py
```
```bash
# Code Block
query = "How is AI used in conservation to support biodiversity and protect endangered species?"

results = search_pinecone(
    query=query,
    model=models["bge-m3"],
    index=index_bge,
    top_k=10,
    chunking_type="sentence"  # or 'semantic', 'fixed', or None
)

answer = generate_answer_groq(results['matches'], query)
print("💡 Answer:", answer)
```

---

## 📊 Chunking Options

Supported strategies for dividing documents:

- "none" — No filter on chunking type

- "sentence" — NLTK-based sentence chunking

- "fixed" — Fixed token window with overlap

- "semantic" — Embedding-based clustering (optional advanced)

Choose your chunking strategy in the search_pinecone() function.

---

## Thank You!





