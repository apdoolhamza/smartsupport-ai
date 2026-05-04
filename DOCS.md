# SmartSupport AI - Documentation

Technical reference for SmartSupport AI.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [RAG Engine](#2-rag-engine)
3. [Index Storage](#3-index-storage)
4. [API Server](#4-api-server)
5. [Widget Script](#5-widget-script)
6. [Configuration](#6-configuration)
7. [Build Index Script](#7-build-index-script)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Tuning](#9-performance-tuning)
10. [Upgrading Storage for Scale](#10-upgrading-storage-for-scale)

---

## 1. Architecture

SmartSupport AI has three parts:

**Backend** (`backend/`)
A FastAPI server that receives customer questions, searches the index, generates answers, and serves the widget script.

**Widget** (`widget/`)
A self-contained JavaScript file that injects a chat button and panel into any webpage. It reads its configuration from a global object injected by the server.

**Index** (`index/`)
The processed knowledge built from your documents. Created by running `build_index.py`. Contains the FAISS vector index, metadata, and embedding cache.

### Request flow

```
Customer types a question in the chat widget
        |
        | HTTP POST /api/chat
        v
FastAPI receives the question
        |
        v
RAGEngine.query(question)
        |
        |-- Embeds the question into a vector (384 numbers)
        |-- Searches FAISS for the top-K most similar document chunks
        |-- Builds a prompt: system instruction + retrieved chunks + question
        |-- SmolLM2 reads the prompt and generates an answer
        |
        v
FastAPI returns JSON
        |
        v
Widget renders the answer in the chat panel
```

## 2. RAG Engine

The RAG engine (`backend/rag_engine.py`) handles everything related to documents and answer generation.

### Document chunking

Documents are split into overlapping chunks before indexing.

```
CHUNK_SIZE    = 400 characters  (~4 to 5 sentences)
CHUNK_OVERLAP = 80  characters  (repeated at start of next chunk)
```

Overlap preserves context that would otherwise be lost at chunk boundaries. For example, if a question and its answer are split across a boundary, the overlap ensures both appear in at least one chunk.

### Embedding

Each chunk is converted to a 384-dimensional float vector using the `all-MiniLM-L6-v2` sentence-transformers model. Vectors that represent similar meaning are close together in vector space.

The same model is used at query time to embed the customer's question. The question vector is then compared to all stored chunk vectors.

### Embedding cache

Computing embeddings is the slowest part of the indexing process. The engine caches every embedding keyed by the MD5 hash of the chunk text.

```
cache key   = md5(chunk_text)
cache value = [0.12, -0.34, 0.88, ...]  (384 floats)
```

If you run `build_index.py` again after editing only one file, chunks from the unchanged file are loaded from cache instantly. Only new or changed content needs to be re-embedded.

The cache is stored at `index/embed_cache.json`.

### FAISS index

FAISS (Facebook AI Similarity Search) stores all chunk vectors and performs fast nearest-neighbor search.

Index type: `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity).

Cosine similarity scores range from 0.0 to 1.0. Scores above 0.7 indicate a strong match. Scores below 0.3 indicate that no relevant content was found.

### Answer generation

Retrieved chunks are assembled into a prompt for SmolLM2-135M-Instruct:

```
[system]
You are a professional customer support assistant.
Answer the user's question using ONLY the context provided below.
Be concise, accurate, and helpful.
If the answer is not in the context, say so clearly.

Context:
[Source: faq.txt]
We accept returns within 7 days...

[Source: products.txt]
Product: HP Pavilion 15, Price: 350,000 NGN...

[user]
What is your return policy?

[assistant]
```

The model completes the assistant turn, generating a natural language answer grounded in the retrieved context.

## 3. Index Storage

SmartSupport AI does not use a database. All data is stored as files in the `index/` directory.

### Files created by build_index.py

```
index/
|-- faiss.index       Binary FAISS index file. Not human-readable.
|-- metadata.json     JSON array of all text chunks with their source filenames.
`-- embed_cache.json  JSON object mapping md5(chunk) to its float vector.
```

### metadata.json structure

```json
[
  {
    "text": "We accept returns within 7 days for faulty or incorrectly sent items...",
    "source": "faq.txt"
  },
  {
    "text": "Product: HP Pavilion 15. Price: 350,000 NGN. Processor: Intel Core i5...",
    "source": "products.txt"
  }
]
```

### What is NOT stored

The original document files are read during indexing but not stored in the `index/` directory. The original files remain in `documents/`. The index only contains the processed, vectorized knowledge.

### Persistence

The `index/` directory is written to disk and persists across server restarts. When the server starts, it loads the index from disk into memory.

If you delete the `index/` directory, run `python scripts/build_index.py` again to rebuild it.

## 4. API Server

`backend/main.py` is a FastAPI application.

### Endpoints

**GET /health**

Returns the server status and index statistics.

**POST /api/chat**

Accepts a question and returns an answer.

**POST /api/chat/stream**

Same as `/api/chat` but returns a Server-Sent Events stream. Each event contains one token.

**GET /widget.js**

Returns the chat widget JavaScript file with the business configuration injected at the top. The widget is served with `Cache-Control: no-cache` so configuration changes take effect immediately after a server restart.

### Running the server

Development mode (with auto-reload on file changes):
```bash
ENV=development python backend/main.py
```

Production mode:
```bash
python backend/main.py
```

With uvicorn directly:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 5. Widget Script

`widget/widget.js` is a self-contained JavaScript file with no external dependencies.

### How it works

1. On load, it reads `window.__SS_CONFIG__` for configuration (injected by the server)
2. It injects a `<style>` block with all widget CSS into `<head>`
3. It creates a launcher button and a chat panel and appends them to `<body>`
4. When the customer clicks the button, the panel opens
5. When the customer sends a message, it posts to `/api/chat` and displays the response

### Browser support

The widget uses ES5-compatible syntax. It works in all modern browsers and Internet Explorer 11.

### Avoiding conflicts

All widget element IDs are prefixed with `ss-` to avoid clashing with the host website's own styles and scripts.

The widget CSS is scoped to `#ss-panel`, `#ss-launcher`, and `.ss-*` selectors.

### Mobile

On screens narrower than 420px, the panel expands to fill the available width with 10px margins on each side.

## 6. Configuration

All configuration is managed through environment variables. In production, set these in your server environment or in a `.env` file.

### Widget configuration

These values are injected into the widget script by the server.

| Variable | Description | Default |
|---|---|---|
| `BOT_NAME` | Name shown in the chat header | `Support` |
| `PRIMARY_COLOR` | Main color button, user bubbles, accents | `#2563eb` |
| `SECONDARY_COLOR` | Gradient secondary color | `#1d4ed8` |
| `WELCOME_MESSAGE` | First message the bot sends when the widget opens | See `.env.example` |
| `FALLBACK_MESSAGE` | Message shown when confidence is below threshold | See `.env.example` |
| `WIDGET_POSITION` | Button placement: `bottom-right` or `bottom-left` | `bottom-right` |
| `BUTTON_SIZE` | Launcher button diameter (CSS size value) | `56px` |
|`API_KEYS` | Set API key (Optional)| `ss-abc123` |
|`RATE_LIMIT_PER_MINUTE` | Rate limit | `20`

### Server configuration

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | Public URL of this server. Must be set for widget.js to work. | `http://localhost:8000` |
| `PORT` | Port the server listens on | `8000` |

### RAG configuration

| Variable | Description | Default |
|---|---|---|
| `EMBED_MODEL` | Hugging Face model for text embedding | `all-MiniLM-L6-v2` |
| `LLM_MODEL` | Hugging Face model for answer generation | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `CHUNK_SIZE` | Characters per document chunk | `250` |
| `CHUNK_OVERLAP` | Overlap characters between chunks | `40` |
| `TOP_K` | Number of chunks retrieved per question | `2` |
| `INDEX_DIR` | Directory where the FAISS index is stored | `./index` |

---

## 7. Build Index Script

`scripts/build_index.py` reads all supported files from the `documents/` directory, processes them, and writes the index to `index/`.

### Supported file types

| Extension | Notes |
|---|---|
| `.txt` | Plain text. Best format for FAQs. |
| `.md` | Markdown. Parsed as plain text. |
| `.pdf` | Text-based PDFs only. Scanned PDFs (images) are not supported. |
| `.csv` | Each row is treated as a text entry. |

### When to re-run

Run `build_index.py` again whenever you:
- Add a new document to `documents/`
- Edit existing content
- Delete a document

After re-running, restart the server so it loads the updated index.

## 8. Troubleshooting

**The widget does not appear on my website**

Check the browser console (F12 > Console) for errors. Common causes:
- Your website is served over HTTPS but your API server uses HTTP. Browsers block mixed content. Your API server must also use HTTPS.
- The URL in the `<script src="...">` tag is wrong.

**The bot answers with the fallback message for every question**

Possible causes:
- The documents do not cover the topic being asked about.
- The documents are too vague or too brief.
- The index was not rebuilt after the documents were edited.

**The server starts but says "Index not found"**

You have not run `build_index.py` yet, or you ran it but the index was saved to a different directory. Check that `index/faiss.index` exists.

**build_index.py finishes but shows 0 chunks**

The files in `documents/` contain no readable text. Check:
- The files are not empty.
- PDF files are text-based, not scanned images.
- The file encoding is UTF-8.

**First response takes 20-30 seconds**

This is the model cold-start. On first query, the language model loads from disk into memory. Subsequent responses are much faster (1-3 seconds). On Hugging Face Spaces free tier, the Space may also sleep after inactivity and take 30 seconds to wake.

## 9. Performance Tuning

| Issue | Adjustment |
|---|---|
| Answers too brief | Increase `TOP_K` to 4 |
| Answers include unrelated content | Decrease `TOP_K` to 2 (default) |
| Index builds slowly | Normal on first run. Cache speeds up subsequent runs. |
| Answers feel generic | Improve document quality, be more specific in your FAQ |
| Response time too slow | Switch to a smaller LLM (if available) |
| Better answer quality | Switch to a larger LLM (Phi-3, Mistral-7B on GPU) |

### Choosing a language model

| Model | RAM required | Speed | Quality |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 1GB | Fast | Basic |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` (default) | 3GB | Moderate | Good |
| `microsoft/Phi-3-mini-4k-instruct` | 8GB | Slower | Better |
| `mistralai/Mistral-7B-Instruct-v0.3` | 16GB (GPU recommended) | Slower | Best |

### Choosing an embedding model

| Model | Size | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` (default) | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Moderate | Better |
| `BAAI/bge-large-en-v1.5` | 1.3GB | Slow | Best |

## 10. Upgrading Storage for Scale

The default file-based storage is sufficient for most deployments. If you need to scale:

### Persist the FAISS index across ephemeral deployments

For platforms where the disk resets on restart (Heroku, Railway free tier), push the `index/` directory to a Hugging Face Dataset repository after building:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./index",
    repo_id="YOUR_USERNAME/smartsupport-index",
    repo_type="dataset",
)
```

On startup, download it before loading:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="YOUR_USERNAME/smartsupport-index",
    repo_type="dataset",
    local_dir="./index",
)
```

### Replace FAISS with a managed vector database

For production deployments with frequent updates or large document sets, replace the FAISS index with Pinecone, Qdrant, or pgvector. The `RAGEngine` class is designed so only the `index_documents`, `save`, `load`, and `retrieve` methods need to change.

---

*Documentation version: 2.0.0*
