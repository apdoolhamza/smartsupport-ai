from dotenv import load_dotenv
load_dotenv() 

import os
import sys
import json
import time
import hashlib
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smartsupport")

# Single global engine - instantiated once, shared across all requests
engine = RAGEngine()


# Startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SmartSupport AI starting...")
    engine.load()
    if not engine.is_ready():
        logger.info("No index found — building from documents/")
        _build_index_from_docs()
    if engine.is_ready():
        s = engine.stats()
        logger.info("Ready — %d chunks | sources: %s", s["total_chunks"], s["sources"])
    else:
        logger.warning("Index is empty. Run: python scripts/build_index.py")
    yield
    logger.info("Shutting down")


def _build_index_from_docs():
    """
    Auto-build index from documents/ at startup.
    Prefers documents/rewritten/ files when they exist.
    """
    docs_dir      = Path("./documents")
    rewritten_dir = Path("./documents/rewritten")
    supported     = {".txt", ".md", ".pdf", ".csv"}

    if not docs_dir.exists():
        logger.warning("documents/ folder not found")
        return

    files = [
        f for f in docs_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in supported
        and f.name.lower() != "readme.md"
        and rewritten_dir not in f.parents
    ]
    if not files:
        logger.warning("No documents found in documents/")
        return

    docs = []
    for f in files:
        rewritten = rewritten_dir / f.name
        use       = rewritten if rewritten.exists() else f
        label     = f"{f.name} (rewritten)" if use == rewritten else f.name
        try:
            text = engine.extract_pdf(use) if use.suffix.lower() == ".pdf"                    else use.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                docs.append({"text": text, "source": f.name})
                logger.info("  loaded: %s", label)
        except Exception as e:
            logger.error("  failed: %s — %s", f.name, e)

    if docs:
        engine.index_documents(docs)
        engine.save()
        s = engine.stats()
        logger.info("Index built: %d chunks / %d vectors", s["total_chunks"], s["index_vectors"])


#  API Key auth + rate limiting
#
#  HOW TO ENABLE:
#    Add to .env:
#      API_KEYS=key-abc123,key-xyz789
#      RATE_LIMIT_PER_MINUTE=20
#
#  HOW IT WORKS:
#    - Every request to /api/chat or /api/chat/stream must include:
#        Header:     X-API-Key: your-key
#        OR Query:   ?api_key=your-key
#        OR Body:    {"message": "...", "api_key": "your-key"}
#    - Each key gets RATE_LIMIT_PER_MINUTE requests per minute (sliding window)
#    - /health and /widget.js are always public — no key needed
#    - Leave API_KEYS empty to disable auth (useful for local dev)
#
#  GENERATE A KEY:
#    python -c "import secrets; print('ss-' + secrets.token_hex(16))"

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Read auth config
# Strip whitespace from every key. Empty string after strip is ignored.
# .env example:  API_KEYS=ss-abc123,ss-xyz789
_raw_keys_env = os.getenv("API_KEYS", "").strip()
_VALID_KEYS: set = {k.strip() for k in _raw_keys_env.split(",") if k.strip()}
_AUTH_ENABLED    = bool(_VALID_KEYS)
_RATE_LIMIT      = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

# In-memory rate limit store: { api_key: [timestamp, ...] }
_rate_store: Dict[str, List[float]] = defaultdict(list)

def _check_rate_limit(key: str) -> bool:
    """Sliding window — True if key is within its rate limit."""
    now = time.time()
    _rate_store[key] = [t for t in _rate_store[key] if now - t < 60.0]
    if len(_rate_store[key]) >= _RATE_LIMIT:
        return False
    _rate_store[key].append(now)
    return True

def _auth(
    header_key: Optional[str],
    query_key:  Optional[str],
    body_key:   Optional[str],
):
    """
    Validate API key. No-op when auth is disabled (API_KEYS not set in .env).
    Key accepted from: X-API-Key header, ?api_key= query, or body field.
    """
    if not _AUTH_ENABLED:
        return

    key = header_key or query_key or body_key

    if not key:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "unauthorized",
                "message": "API key required. Send it in the X-API-Key header.",
            },
        )
    if key not in _VALID_KEYS:
        raise HTTPException(
            status_code=401,
            detail={
                "error":   "invalid_key",
                "message": "Invalid API key.",
            },
        )
    if not _check_rate_limit(key):
        raise HTTPException(
            status_code=429,
            detail={
                "error":   "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {_RATE_LIMIT} requests per minute.",
            },
        )


app = FastAPI(
    title="SmartSupport AI",
    version="2.0.0",
    docs_url=None, 
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
    expose_headers=["X-API-Key"],
)


# Request / response models

class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None
    api_key:    Optional[str] = None   # key can also come in the request body


class ChatResponse(BaseModel):
    answer:           str
    sources:          List[Dict[str, Any]] = []
    confidence:       float = 0.0
    session_id:       str
    response_time_ms: int


# Routes

@app.get("/health")
async def health():
    """Public — no auth needed."""
    if engine.is_ready():
        s = engine.stats()
        return {
            "status":      "ok",
            "index_ready": True,
            "chunks":      s["total_chunks"],
            "vectors":     s["index_vectors"],
            "sources":     s["sources"],
            "auth":        _AUTH_ENABLED,
        }
    return {
        "status":      "ok",
        "index_ready": False,
        "chunks":      0,
        "vectors":     0,
        "auth":        _AUTH_ENABLED,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    req:  Request,
    hkey: Optional[str] = Depends(_API_KEY_HEADER),
):
    """
    Main chat endpoint.
    API key accepted via header, query param, or request body.
    """
    _auth(hkey, req.query_params.get("api_key"), body.api_key)

    if not engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Run: python scripts/build_index.py",
        )

    start = time.perf_counter()
    try:
        result = engine.query(body.message)
    except Exception as e:
        logger.exception("Query error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

    elapsed = int((time.perf_counter() - start) * 1000)

    fallback = os.getenv(
        "FALLBACK_MESSAGE",
        "I do not have that information. Please contact our support team.",
    )
    answer = result["answer"] if result.get("confidence", 0) >= 0.8 else fallback

    session_id = body.session_id or hashlib.md5(
        str(time.time()).encode()
    ).hexdigest()[:12]

    return ChatResponse(
        answer=answer,
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        session_id=session_id,
        response_time_ms=elapsed,
    )


@app.post("/api/chat/stream")
async def chat_stream(
    body: ChatRequest,
    req:  Request,
    hkey: Optional[str] = Depends(_API_KEY_HEADER),
):
    """Streaming chat - same auth as /api/chat."""
    _auth(hkey, req.query_params.get("api_key"), body.api_key)

    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Index not ready.")

    async def generate():
        try:
            async for token in engine.stream_query(body.message):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/widget.js")
async def serve_widget():
    """
    Public - no auth.
    Serves widget.js with server config injected as fallback defaults.
    Page-level window.__SS_CONFIG__ values always take priority.
    """
    widget_path = Path(__file__).parent / "widget" / "widget.js"
    if not widget_path.exists():
        widget_path = Path(__file__).parent.parent / "widget" / "widget.js"
    if not widget_path.exists():
        raise HTTPException(status_code=404, detail="widget.js not found")

    code     = widget_path.read_text(encoding="utf-8")
    api_base = os.getenv("API_BASE_URL")

    injection = (
        "(function(){\n"
        "  if(typeof window.__SS_CONFIG__==='undefined'||"
        "window.__SS_CONFIG__===null){window.__SS_CONFIG__={};}\n"
        "  var d={\n"
        f"    apiBase:'{api_base}',\n"
        f"    botName:'{os.getenv('BOT_NAME')}',\n"
        f"    primaryColor:'{os.getenv('PRIMARY_COLOR')}',\n"
        f"    secondaryColor:'{os.getenv('SECONDARY_COLOR')}',\n"
        f"    welcomeMessage:'{os.getenv('WELCOME_MESSAGE')}',\n"
        f"    fallbackMessage:'{os.getenv('FALLBACK_MESSAGE')}',\n"
        f"    position:'{os.getenv('WIDGET_POSITION')}',\n"
        f"    buttonSize:'{os.getenv('BUTTON_SIZE')}'\n"
        "  };\n"
        "  for(var k in d){"
        "if(!(k in window.__SS_CONFIG__)||"
        "window.__SS_CONFIG__[k]===undefined||"
        "window.__SS_CONFIG__[k]===null){"
        "window.__SS_CONFIG__[k]=d[k];}}\n"
        "})();\n"
    )

    return StreamingResponse(
        iter([injection + code]),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")