# 📚 Company Knowledge Chat (RAG-based QA System)

A simple yet robust **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about companies (e.g., Microsoft, Tesla, Google) and receive **factually grounded answers** based strictly on provided documents.

This project demonstrates a complete **end-to-end RAG pipeline** with a clean backend, modern frontend, and vector-based retrieval.

---

## 🚀 Features

- 🔍 **Semantic Search with Vector Database**
- 🧠 **Retrieval-Augmented Generation (RAG)**
- ❌ **No Hallucinations** – answers only from retrieved context
- 💬 **Chat-style Web Interface**
- ⚡ **FastAPI Backend**
- 🎨 **Modern HTML + CSS Frontend**
- 🧩 **Modular & Clean Code Structure**

---

## 🏗️ Architecture Overview

User (Browser)

↓

HTML / CSS / JavaScript

↓
POST /chat

FastAPI Backend

↓

Chroma Vector Database

↓

Relevant Text Chunks

↓

LLM (OpenAI)

↓

Grounded Answer


---

## 📂 Project Structure

.

├── main.py # FastAPI application

├── rag_chat.py # RAG pipeline (retrieval + prompt + LLM)

├── templates/

  │ └── index.html # Frontend UI

├── static/ # Static assets 

├── db/

  │ └── chroma_db/ # Vector database (ignored in git)

├── .env # API keys (ignored in git)

├── .gitignore

└── README.md

---

## 🧠 How RAG Works in This Project

1. Company information is stored in `.txt` files.
2. Text is chunked and embedded using **OpenAI embeddings**.
3. Embeddings are stored in **ChromaDB**.
4. When a user asks a question:
   - The question is embedded
   - Top-K similar chunks are retrieved
   - The LLM is prompted **only with retrieved context**
5. If the answer is not present in context → the system refuses to answer.

This guarantees **trustworthy, non-fabricated responses**.

---

## ⚙️ Tech Stack

**Backend**
- Python
- FastAPI
- ChromaDB
- OpenAI API

**Frontend**
- HTML
- CSS
- Vanilla JavaScript

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/ANURAG-sys-08/RAG-project.git
cd <RAG-project>
### 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
### 3️⃣ Install Dependencies
### 4️⃣ Set Environment Variables
OPENAI_API_KEY=your_api_key_here
### 5️⃣ Run the App
uvicorn main:app --reload
