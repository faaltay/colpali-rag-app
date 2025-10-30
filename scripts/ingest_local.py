import argparse
from pathlib import Path
from typing import List

from app.storage.faiss_store import LocalFaissStore
import PyPDF2

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="PDF file or folder to ingest")
    parser.add_argument("--collection", "-c", default="colpali_documents")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.path)
    store = LocalFaissStore()
    store.create_collection(args.collection)

    files = [path] if path.is_file() else list(path.glob("*.pdf"))
    total = 0
    for f in files:
        print(f"Ingesting {f}")
        text = extract_text_from_pdf(f)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        metadatas = [{"source": str(f), "chunk": i} for i in range(len(chunks))]
        store.add_texts(args.collection, chunks, metadatas)
        total += len(chunks)
    print(f"Ingested {total} chunks into collection '{args.collection}'")

if __name__ == "__main__":
    main()