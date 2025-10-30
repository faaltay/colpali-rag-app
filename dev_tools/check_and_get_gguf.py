import os
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download

REPO_ID = "vidore/colqwen2.5-v0.2"

# prefer project-local models folder; keep behavior robust across environments
OUT_DIR = os.path.abspath(r"G:\My Drive\Projects\colpali-rag-app\models")
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except Exception:
    OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(OUT_DIR, exist_ok=True)

# token from env (set HUGGINGFACE_HUB_TOKEN) or None for anonymous access
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

print("Using model output directory:", OUT_DIR)
print("Listing files in repo:", REPO_ID)
files = list_repo_files(REPO_ID)
for f in files:
    print(f)

ggufs = [f for f in files if f.lower().endswith(".gguf")]
if ggufs:
    fname = ggufs[0]
    print("\nFound .gguf file in repo:", fname)
    print("Downloading", fname, "to", OUT_DIR)
    path = hf_hub_download(repo_id=REPO_ID, filename=fname, cache_dir=OUT_DIR, token=HF_TOKEN)
    print("Downloaded to:", path)
else:
    print("\nNo .gguf files found. Downloading repo snapshot for conversion.")
    snap = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=OUT_DIR,
        local_dir=os.path.join(OUT_DIR, "repo_snapshot"),
        allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.txt", "*.model"],
        token=HF_TOKEN
    )
    print("Downloaded repo snapshot to:", snap)