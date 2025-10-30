import os
import json
import uuid
import traceback
from pathlib import Path
from typing import List
import shutil  # NEW
import fitz  # PyMuPDF  # NEW

import gradio as gr
import requests
import PyPDF2

from app.storage.faiss_store import LocalFaissStore

# Config (env overrides)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/generate")
DEFAULT_COLLECTION = os.getenv("FAISS_COLLECTION", None)  # if None, per-session collection
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3000"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "256"))

# NEW: persistent upload dir (avoid exposing temp paths)
UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", "./data/uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR = (UPLOAD_DIR / "previews")  # NEW
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)  # NEW

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = text or ""
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def extract_text_from_pdf(path: Path) -> str:
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

# NEW: extract each page separately so we can track page numbers
def extract_pages_from_pdf(path: Path) -> List[str]:
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

# NEW: cached page preview renderer
def render_pdf_page_preview(pdf_path: Path, page_num: int, scale: float = 2.0) -> Path:
    """
    Render given 1-based page num to PNG and return the preview image path.
    Caches by filename+page.
    """
    try:
        stem = pdf_path.stem
        out = PREVIEW_DIR / f"{stem}_p{page_num}.png"
        if out.exists():
            return out
        with fitz.open(str(pdf_path)) as doc:
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(out))
        return out
    except Exception:
        return PREVIEW_DIR / f"{pdf_path.stem}_p{page_num}.png"

def ensure_collection(state: dict) -> dict:
    state = state or {}
    if "collection" not in state or not state["collection"]:
        state["collection"] = DEFAULT_COLLECTION or f"gr_{uuid.uuid4().hex[:8]}"
    return state

def ingest_files(files: List[gr.File], state: dict, chunk_size: int, overlap: int):
    try:
        state = ensure_collection(state)
        store = LocalFaissStore()
        store.create_collection(state["collection"])
        total_chunks = 0
        total_files = 0

        for f in files or []:
            p = Path(f.name if hasattr(f, "name") else f)
            if not p.exists():
                p = Path(f) if isinstance(f, str) else None
                if not p or not p.exists():
                    continue
            if p.suffix.lower() not in [".pdf", ".txt", ".md"]:
                continue

            # copy into persistent uploads dir and use only filename in metadata
            dest = UPLOAD_DIR / p.name
            if dest.exists():
                dest = UPLOAD_DIR / f"{p.stem}-{uuid.uuid4().hex[:6]}{p.suffix}"
            try:
                shutil.copy2(p, dest)
            except Exception:
                dest = p  # fallback

            display_name = dest.name

            texts_to_add: List[str] = []
            metas_to_add: List[dict] = []

            if p.suffix.lower() == ".pdf":
                pages = extract_pages_from_pdf(p)
                for page_num, page_text in enumerate(pages, start=1):
                    if not page_text:
                        continue
                    chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                    for i, ch in enumerate(chunks):
                        texts_to_add.append(ch)
                        metas_to_add.append({
                            "source": display_name,
                            "filepath": str(dest),
                            "page": page_num,         # NEW: page number (1-based)
                            "page_chunk": i           # NEW: chunk index within the page
                        })
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for i, ch in enumerate(chunks):
                    texts_to_add.append(ch)
                    metas_to_add.append({
                        "source": display_name,
                        "filepath": str(dest),
                        "page": None,
                        "page_chunk": i
                    })

            if texts_to_add:
                store.add_texts(state["collection"], texts_to_add, metas_to_add)
                total_chunks += len(texts_to_add)
                total_files += 1

        state["ingested"] = True
        msg = f"Ingested {total_chunks} chunks from {total_files} file(s) into collection '{state['collection']}'."
        return msg, state
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return f"Ingestion failed: {e}\n{tb}", state or {}

def build_prompt(question: str, contexts: list, max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
    selected, total = [], 0
    for c in contexts:
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        if total + len(txt) > max_context_chars:
            remain = max_context_chars - total
            if remain > 0:
                selected.append({"text": txt[:remain], "metadata": c.get("metadata", {})})
            break
        selected.append({"text": txt, "metadata": c.get("metadata", {})})
        total += len(txt)

    parts = []
    for i, c in enumerate(selected, start=1):
        meta = c.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page")
        ck = meta.get("page_chunk", meta.get("chunk", "?"))
        page_str = f", page={page}" if page else ""
        parts.append(f"[{i}] Source: {src}{page_str} (chunk={ck})\n{c['text']}")
    ctx_text = "\n\n---\n\n".join(parts)

    prompt = (
        "You are a concise assistant. Use ONLY the CONTEXT to answer the QUESTION.\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "INSTRUCTIONS: Answer in one short paragraph. Cite sources inline with [1], [2], etc. "
        "If the answer is not supported by the context, say \"I don't know based on the provided context.\""
    )
    return prompt

def call_llm(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    payload = {"prompt": prompt, "max_new_tokens": int(max_new_tokens), "temperature": float(temperature)}
    try:
        r = requests.post(LOCAL_LLM_URL, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "text" in data:
            return str(data["text"]).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        return json.dumps(data)
    except requests.RequestException as e:
        body = ""
        if getattr(e, "response", None) is not None:
            try:
                body = e.response.text
            except Exception:
                body = "<no body>"
        return f"LLM call failed: {e}\n{body}"

# UPDATED: return gallery data + sources lines
def chat_fn(message: str, history: list, state: dict, top_k: int, max_new_tokens: int):
    state = ensure_collection(state)
    if not state.get("ingested"):
        return history + [[message, "Please upload and ingest documents first."]], state, [], ""

    store = LocalFaissStore()
    contexts = store.search_texts(state["collection"], message, top_k=top_k)
    if not contexts:
        return history + [[message, "No context found. Try another question."]], state, [], ""

    prompt = build_prompt(message, contexts)
    answer = call_llm(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    # Build sources list with page numbers and render previews
    previews = []
    lines = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        src = meta.get("source")
        fp = Path(meta.get("filepath", "")) if meta.get("filepath") else None
        page = meta.get("page")
        ck = meta.get("page_chunk", meta.get("chunk"))
        score = float(c.get("score", 0.0))
        line = f"[{i}] {src}, page={page} (chunk={ck}) score={score:.4f}" if page else f"[{i}] {src} (chunk={ck}) score={score:.4f}"
        lines.append(line)

        # render preview if we have a file path and page
        if fp and page and fp.exists() and fp.suffix.lower() == ".pdf":
            img_path = render_pdf_page_preview(fp, int(page))
            caption = line
            previews.append((str(img_path), caption))

    sources_md = "\n".join(lines)
    answer_full = f"{answer}\n\n---\n{sources_md}"

    return history + [[message, answer_full]], state, previews, sources_md

def reset_collection(state: dict):
    state = {"collection": f"gr_{uuid.uuid4().hex[:8]}", "ingested": False}
    return "Session reset. New collection: " + state["collection"], state

with gr.Blocks(title="Local RAG (FAISS + llama.cpp)") as demo:
    gr.Markdown("## Local RAG — upload PDFs, ingest, and chat (all local)")
    with gr.Row():
        files = gr.Files(label="Upload PDFs (multiple allowed)", file_count="multiple", type="filepath")
    with gr.Row():
        chunk_size = gr.Slider(500, 2000, value=1000, step=100, label="Chunk size")
        overlap = gr.Slider(0, 400, value=200, step=50, label="Overlap")
        top_k = gr.Slider(1, 10, value=TOP_K, step=1, label="Top-K")
        max_new = gr.Slider(32, 1024, value=MAX_NEW_TOKENS, step=32, label="Max new tokens")
    ingest_btn = gr.Button("Ingest")
    status = gr.Markdown("Status: idle")
    reset_btn = gr.Button("Reset Session")

    chat = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Ask a question about your documents", placeholder="Type your question and press Enter")

    # NEW: related pages gallery + sources list
    gallery = gr.Gallery(label="Related pages", columns=[4], height=240)
    sources_out = gr.Markdown(label="Sources")

    state = gr.State({"collection": DEFAULT_COLLECTION or f"gr_{uuid.uuid4().hex[:8]}", "ingested": False})

    ingest_btn.click(
        fn=ingest_files,
        inputs=[files, state, chunk_size, overlap],
        outputs=[status, state],
    )
    reset_btn.click(
        fn=reset_collection,
        inputs=[state],
        outputs=[status, state],
    )
    msg.submit(
        fn=chat_fn,
        inputs=[msg, chat, state, top_k, max_new],
        outputs=[chat, state, gallery, sources_out],  # UPDATED
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=msg,
    )

if __name__ == "__main__":
    # Launch on localhost only
    demo.launch(server_name="127.0.0.1", server_port=7860)