from dotenv import load_dotenv
load_dotenv()

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, AsyncIterator

from transformers import TextIteratorStreamer
from threading import Thread

import numpy as np
import torch

torch.set_num_threads(4) 

logger = logging.getLogger("smartsupport.rag")

EMBED_MODEL_NAME    = os.getenv("EMBED_MODEL",    "all-MiniLM-L6-v2")
LLM_MODEL_NAME      = os.getenv("LLM_MODEL",      "HuggingFaceTB/SmolLM2-135M-Instruct")
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE",   "250"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP","40"))
TOP_K               = int(os.getenv("TOP_K",        "2"))
INDEX_DIR           = Path(os.getenv("INDEX_DIR",   "./index"))

# Lazy-loaded models
_embed_model = None
_llm_pipeline = None


def get_embed_model():
    """Lazy-load embedding model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(
            EMBED_MODEL_NAME,
            cache_folder=str(Path.home() / ".cache" / "sentence-transformers"),
        )
    return _embed_model


def get_llm():
    """
    Load the LLM with settings tuned to minimise hallucination.

    Key settings:
      do_sample=False          greedy decoding - deterministic, no random invention
      repetition_penalty=1.3   discourages the model from copying the prompt and
                               then extending it with made-up text
      max_new_tokens=80       short answers stay grounded in the context
      no device arg            avoids accelerate conflict
    """
    global _llm_pipeline
    if _llm_pipeline is None:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        logger.info("Loading LLM: %s (this may take a moment)", LLM_MODEL_NAME)

        # Load with optimizations for SPEED on CPU
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32,  # float32 for CPU stability
            low_cpu_mem_usage=True,
            cache_dir=str(Path.home() / ".cache" / "huggingface" / "hub"),
        )

        # Enable inference optimizations
        model.eval()  # Set to eval mode
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        _llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=80, 
            do_sample=False,
            repetition_penalty=1.3,
            num_beams=1,
        )
        logger.info("LLM loaded successfully")
    return _llm_pipeline

class RAGEngine:
    """RAG engine with file-based storage and fast inference."""

    def __init__(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._index:    Any = None
        self._metadata: List[Dict] = []
        self._cache:    Dict[str, List[float]] = {}
        self._load_cache()

    def _cache_path(self) -> Path:
        return INDEX_DIR / "embed_cache.json"

    def _load_cache(self):
        """Load embedding cache from disk."""
        p = self._cache_path()
        if p.exists():
            try:
                self._cache = json.loads(p.read_text())
                logger.info("Loaded %d cached embeddings", len(self._cache))
            except Exception as exc:
                logger.error("Failed to load cache: %s", exc)
                self._cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        self._cache_path().write_text(json.dumps(self._cache))

    def _embed(self, text: str) -> np.ndarray:
        """Embed single text with caching."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return np.array(self._cache[key], dtype="float32")
        
        vec = get_embed_model().encode(
            [text],
            batch_size=8,
            normalize_embeddings=True
        )[0].astype("float32")
        
        self._cache[key] = vec.tolist()
        return vec
    

     #  Embeddin

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed with caching."""
        results = {}
        new_texts, new_keys = [], []

        for text in texts:
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self._cache:
                results[key] = np.array(self._cache[key], dtype="float32")
            else:
                new_texts.append(text)
                new_keys.append(key)

        if new_texts:
            vecs = get_embed_model().encode(
                new_texts, batch_size=16, normalize_embeddings=True
            )
            for key, vec in zip(new_keys, vecs):
                arr = vec.astype("float32")
                self._cache[key] = arr.tolist()
                results[key] = arr

        ordered = []
        for text in texts:
            key = hashlib.md5(text.encode()).hexdigest()
            ordered.append(results[key])
        return np.vstack(ordered)
    

     #  Chunking

    def _chunk(self, text: str, source: str) -> List[Dict]:
        """Chunk documents into overlapping segments."""
        import re
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)

        chunks, current, current_len = [], [], 0
        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > CHUNK_SIZE and current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "source": source})
                overlap, overlap_len = [], 0
                for s in reversed(current):
                    if overlap_len + len(s) <= CHUNK_OVERLAP:
                        overlap.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current, current_len = overlap, overlap_len
            current.append(sentence)
            current_len += s_len

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": source})
        return chunks

    def index_documents(self, docs: List[Dict[str, str]]):
        """Build FAISS index from documents."""
        import faiss

        logger.info("Chunking %d document(s)...", len(docs))
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self._chunk(doc["text"], doc["source"]))

        if not all_chunks:
            logger.warning("No chunks produced")
            return

        logger.info("Embedding %d chunks...", len(all_chunks))
        texts = [c["text"] for c in all_chunks]
        vecs  = self._embed_batch(texts)
        dim   = vecs.shape[1]

        logger.info("Building FAISS index...")
        self._index    = faiss.IndexFlatIP(dim)
        self._metadata = all_chunks
        self._index.add(vecs)
        self._save_cache()
        logger.info("Index complete: %d chunks, %d vectors", len(all_chunks), self._index.ntotal)

    def save(self):
        """Save index to disk."""
        if self._index is None:
            return
        import faiss
        faiss.write_index(self._index, str(INDEX_DIR / "faiss.index"))
        (INDEX_DIR / "metadata.json").write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2)
        )
        logger.info("Index saved")

    def load(self):
        """Load index from disk."""
        import faiss
        index_file = INDEX_DIR / "faiss.index"
        meta_file  = INDEX_DIR / "metadata.json"
        if index_file.exists() and meta_file.exists():
            self._index    = faiss.read_index(str(index_file))
            self._metadata = json.loads(meta_file.read_text())
            logger.info("Loaded index: %d vectors", self._index.ntotal)
        else:
            logger.info("No saved index")

    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-K chunks from index."""
        if not self.is_ready():
            return []
        q_vec = self._embed(query).reshape(1, -1)
        k     = min(TOP_K, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                chunk = dict(self._metadata[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    def _build_prompt(self, query: str, context_chunks: str) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c.get('source', 'doc')}]\n{c['text']}"
            for c in context_chunks
        )
        return f"""<|im_start|>system
You are a helpful customer support assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that in my knowledge base."
Be concise, friendly, and accurate. Do not make up information.

Context:
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
    
    #  Query
    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG query: retrieve + generate."""
        chunks = self.retrieve(question)
        if not chunks:
            return {
                "answer": "I don't have enough information to answer that. Please contact our support team.",
                "sources": [],
                "confidence": 0.0
            }

        confidence = float(chunks[0]["score"]) if chunks else 0.0
        prompt = self._build_prompt(question, chunks)

        try:
            llm = get_llm()
            output = llm(prompt, return_full_text=False)
            answer = output[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = chunks[0]["text"][:400] + "..." if chunks else "Sorry, I couldn't generate a response."

        sources = [
            {"source": c.get("source", ""), "excerpt": c["text"][:150] + "..."}
            for c in chunks[:2]
        ]
        return {"answer": answer, "sources": sources, "confidence": confidence}

    async def stream_query(self, question: str) -> AsyncIterator[str]:
        """Stream answer tokens."""
        import asyncio
        chunks = self.retrieve(question)
        if not chunks:
            yield "I do not have enough information to answer that question."
            return

        prompt = self._build_prompt(question, chunks)
        try:
            llm       = get_llm()
            tokenizer = llm.tokenizer
            model     = llm.model
            inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer  = TextIteratorStreamer(
                tokenizer, skip_special_tokens=True, skip_prompt=True
            )
            thread = Thread(
                target=model.generate,
                kwargs={**inputs, "streamer": streamer,
                        "max_new_tokens": 80, "do_sample": False}
            )
            thread.start()
            for token in streamer:
                yield token
        except Exception as exc:
            logger.error("Streaming failed: %s", exc)
            result = self.query(question)
            for word in result["answer"].split():
                yield word + " "

    def extract_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n\n".join(parts)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(path))
                return "\n\n".join(
                    p.extract_text() for p in reader.pages if p.extract_text()
                )
            except Exception as exc:
                logger.error("PDF extraction failed for %s: %s", path.name, exc)
                return ""

    def stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_chunks":  len(self._metadata),
            "index_vectors": self._index.ntotal if self._index else 0,
            "cache_entries": len(self._cache),
            "sources":       list({c["source"] for c in self._metadata}),
        }