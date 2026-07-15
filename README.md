# Production-Grade RAG Python App (Dual OpenAI & Gemini Support)

A structured, event-driven Retrieval-Augmented Generation (RAG) system built to handle document ingestion, chunking, vector embedding, storage, and contextual querying.

This application features event-driven task orchestration using **Inngest**, vector search with **Qdrant**, and a user interface powered by **Streamlit**. It dynamically supports both **OpenAI** and **Google Gemini** API keys.

---

## 🏗️ Architecture

```
                       ┌────────────────────────┐
                       │   Streamlit Frontend   │
                       │     (Port 8501)        │
                       └───────────┬────────────┘
                                   │ Sends Ingestion/Query Events
                                   ▼
                       ┌────────────────────────┐
                       │  Inngest Dev Server    │
                       │     (Port 8288)        │
                       └───────────┬────────────┘
                                   │ Executes background functions
                                   ▼
                       ┌────────────────────────┐
                       │    FastAPI Backend     │
                       │     (Port 8001)        │
                       └─────┬───────────┬──────┘
                             │           │
           Generates Chunks  │           │  Embeddings & Search
           & Embeddings      ▼           ▼
                     ┌───────────┐   ┌───────────────┐
                     │ LLM API   │   │ Qdrant DB     │
                     │ (OpenAI/  │   │ (Local        │
                     │  Gemini)  │   │  Embedded)    │
                     └───────────┘   └───────────────┘
```

---

## ✨ Features

* **Dual Model Provider Support**: Automatically switches to Google Gemini (`gemini-embedding-2` and `gemini-3.5-flash`) if `GEMINI_API_KEY` is provided, otherwise falls back to OpenAI (`text-embedding-3-large` and `gpt-4o-mini`).
* **Docker-Free Vector DB**: Runs Qdrant in embedded mode locally under the `./qdrant_storage` folder (requires no local Docker daemon or setup).
* **Event-Driven Workflows**: Asynchronous document ingestion and embedding queries handled cleanly by Inngest, complete with concurrency throttling and rate limits.
* **Streamlit UI**: A clean, premium dashboard to upload PDFs and ask questions about your documents in real-time.

---

## 🛠️ Prerequisites

* **Python 3.13+** (recreates virtual environments automatically with `uv`)
* **Node.js & npm** (to run the Inngest CLI/dev server)
* **`uv` Package Manager** (highly recommended for lightning-fast package installations)

---

## ⚙️ Project Setup

1. **Configure Environment Variables**
   Create a `.env` file in the root of the project directory and supply one of the API keys:
   ```env
   # To use Gemini (default if provided):
   GEMINI_API_KEY=your_gemini_api_key_here

   # OR to use OpenAI:
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Install Dependencies**
   Run the following command to sync your Python environment:
   ```powershell
   uv sync
   ```

---

## 🚀 How to Run the Program

This application requires three distinct components to run in parallel. Open **three separate terminal windows** and run the following:

### 💻 Terminal 1: FastAPI Backend
Start the backend FastAPI server on port `8001`:
```powershell
uv run uvicorn main:app --port 8001
```

### 💻 Terminal 2: Inngest Dev Server
Start the background worker orchestration dashboard (points Inngest to the FastAPI endpoint):
```powershell
npx inngest-cli@latest dev -u http://127.0.0.1:8001/api/inngest
```

### 💻 Terminal 3: Streamlit UI
Start the web interface:
```powershell
uv run streamlit run streamlit_app.py
```

---

## 🌐 Application Ports

Once started, you can access the following services in your browser:
* **Streamlit Web Dashboard**: [http://localhost:8501](http://localhost:8501)
* **Inngest Task Dashboard**: [http://localhost:8288](http://localhost:8288)

---

## 📁 File Structure

* [streamlit_app.py](file:///streamlit_app.py): The Streamlit web page for uploading PDFs and asking questions.
* [main.py](file:///main.py): The FastAPI application endpoints serving Inngest's background triggers (`rag/ingest_pdf`, `rag/query_pdf_ai`).
* [data_loader.py](file:///data_loader.py): Logic for parsing PDF content, text chunk splitting, and calling the LLM embeddings.
* [vector_db.py](file:///vector_db.py): Local embedded Qdrant client operations (Upsert, Cosine search).
* [custom_types.py](file:///custom_types.py): Pydantic schemas validating input/output interfaces.
* [pyproject.toml](file:///pyproject.toml): Modern python project specifications and dependencies.
