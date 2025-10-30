import os
import requests
import argparse
from app.storage.faiss_store import LocalFaissStore

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/generate")
COLLECTION = os.getenv("FAISS_COLLECTION", "colpali_documents")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3000"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "256"))

def build_prompt(question: str, contexts: list, max_context_chars: int = MAX_CONTEXT_CHARS):
    # select and truncate most relevant contexts (preserve order)
    selected = []
    total = 0
    for c in contexts:
        txt = c["text"].strip()
        if total + len(txt) > max_context_chars:
            remaining = max_context_chars - total
            if remaining <= 0:
                break
            txt = txt[:remaining]
            selected.append({"text": txt, "metadata": c.get("metadata", {})})
            break
        selected.append({"text": txt, "metadata": c.get("metadata", {})})
        total += len(txt)

    # build numbered context with simple citation keys [1], [2], ...
    ctx_parts = []
    for i, c in enumerate(selected, start=1):
        src = c.get("metadata", {}).get("source", "unknown")
        chunk_idx = c.get("metadata", {}).get("chunk")
        ctx_parts.append(f"[{i}] Source: {src} (chunk={chunk_idx})\n{c['text']}")
    ctx_text = "\n\n---\n\n".join(ctx_parts)

    prompt = (
        "You are a concise assistant. Use ONLY the CONTEXT snippets below to answer the QUESTION.\n\n"
        "CONTEXT:\n" + ctx_text + "\n\n"
        "QUESTION:\n" + question + "\n\n"
        "INSTRUCTIONS: Answer concisely in one paragraph. Cite sources inline by bracket number (e.g. [1]). "
        "Do not invent sources. If the answer is not supported by the context, say 'I don't know based on the provided context.'"
    )
    return prompt

def call_llm(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    # payload keys depend on your LLM server; include deterministic params
    payload = {"prompt": prompt, "max_new_tokens": int(max_new_tokens)}
    # many adapters support temperature; include if accepted
    payload["temperature"] = float(temperature)
    try:
        resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        # common shapes
        if isinstance(data, dict) and "text" in data:
            return data["text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)
    except requests.exceptions.RequestException as e:
        body = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.text
            except Exception:
                body = "<could not read response body>"
        raise RuntimeError(f"LLM request failed: {e}\nResponse body: {body}") from e

def query(question: str):
    store = LocalFaissStore()
    contexts = store.search_texts(COLLECTION, question, top_k=TOP_K)
    if not contexts:
        print("No context found in collection:", COLLECTION)
        return
    prompt = build_prompt(question, contexts)
    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("LLM call failed:", e)
        return

    print("\n--- ANSWER ---\n")
    print(answer)
    print("\n--- SOURCES ---\n")
    # print mapping for cited indices
    for i, c in enumerate(contexts[:TOP_K], start=1):
        print(f"[{i}] id={c['id']} score={c['score']:.4f} source={c['metadata'].get('source')} chunk={c['metadata'].get('chunk')}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+", help="Question to ask")
    args = p.parse_args()
    q = " ".join(args.question)
    query(q)