import argparse
from app.storage.faiss_store import LocalFaissStore, EMBED_DIM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", "-c", default="colpali_documents", help="collection name")
    parser.add_argument("--dim", type=int, default=EMBED_DIM, help="embedding dimension (optional)")
    args = parser.parse_args()

    store = LocalFaissStore()
    store.create_collection(args.collection, dim=args.dim)
    print(f"Created FAISS collection '{args.collection}' (dim={args.dim}).")

if __name__ == "__main__":
    main()